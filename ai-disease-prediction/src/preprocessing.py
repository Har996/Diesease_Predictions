"""Dataset loading, cleaning and feature encoding.

Two CSV layouts are understood and converted to one internal "wide" layout:

1. WIDE   : one 0/1 column per symptom + a ``disease`` column (+ optional ``age``/``gender``).
2. LONG   : ``Disease, Symptom_1, Symptom_2, ...`` where each cell holds a symptom *name*
            (the layout of the popular Kaggle "Disease Symptom Prediction" dataset).
            A target column called ``prognosis`` (Kaggle "Disease Prediction Using ML") is accepted too.

Design notes on data leakage
----------------------------
* Everything that *learns* from data (imputation values, scaling, one-hot categories) lives inside a
  scikit-learn ``ColumnTransformer`` that is placed in a ``Pipeline`` with the classifier. It is therefore
  fitted on training rows only - never on validation or test rows.
* Exact duplicate rows are removed *before* splitting, so an identical record cannot appear in both
  the training and the test set (a common, silent source of inflated accuracy).
* The cleaning steps in this file use no information from the test set for fitting anything.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET = "disease"
AGE = "age"
GENDER = "gender"
VALID_GENDERS = ("female", "male", "other")

_TARGET_CANDIDATES = ("disease", "prognosis", "diagnosis", "condition", "label")
_GENDER_CANDIDATES = ("gender", "sex")
_IGNORED_COLUMNS = {"id", "index", "patient_id", "unnamed_0"}
_TRUE_WORDS = {"1", "yes", "y", "true", "t"}
_FALSE_WORDS = {"0", "no", "n", "false", "f"}
_LONG_SYMPTOM_RE = re.compile(r"^symptom_?\d+$")


def normalize_name(value: object) -> str:
    """'  Skin Rash ' -> 'skin_rash'. Used for column names, symptoms and user input alike."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")


def humanize(name: str) -> str:
    """'skin_rash' -> 'Skin rash' (display label)."""
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


# --------------------------------------------------------------------------- schema
@dataclass
class DatasetSchema:
    """Describes which columns of the cleaned dataset are model inputs."""

    symptom_cols: list[str]
    has_age: bool = False
    has_gender: bool = False
    target: str = TARGET

    @property
    def numeric_cols(self) -> list[str]:
        return [AGE] if self.has_age else []

    @property
    def categorical_cols(self) -> list[str]:
        return [GENDER] if self.has_gender else []

    @property
    def feature_cols(self) -> list[str]:
        return self.symptom_cols + self.numeric_cols + self.categorical_cols

    def to_dict(self) -> dict:
        return {"symptoms": self.symptom_cols, "has_age": self.has_age, "has_gender": self.has_gender}


@dataclass
class SplitData:
    """Stratified train / validation / test split (``dev`` = train + validation)."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    X_dev: pd.DataFrame = field(init=False)
    y_dev: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.X_dev = pd.concat([self.X_train, self.X_val])
        self.y_dev = pd.concat([self.y_train, self.y_val])


# --------------------------------------------------------------------------- loading
def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV and normalise column names. Long layouts are converted to wide."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Place a CSV at data/disease_symptoms.csv "
            "or run `python -m src.generate_sample_data` for a synthetic demo file."
        )
    df = pd.read_csv(path, skipinitialspace=True)
    if df.empty:
        raise ValueError(f"Dataset {path} is empty.")

    df.columns = [normalize_name(c) or f"unnamed_{i}" for i, c in enumerate(df.columns)]
    if df.columns.duplicated().any():
        dupes = sorted(set(df.columns[df.columns.duplicated()]))
        raise ValueError(f"Duplicate column names after normalisation: {dupes}")
    return _long_to_wide_if_needed(df)


def _find_target(columns: list[str]) -> str:
    for candidate in _TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"No target column found. Expected one of {_TARGET_CANDIDATES}.")


def _long_to_wide_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    long_cols = [c for c in df.columns if _LONG_SYMPTOM_RE.match(c)]
    if not long_cols:
        return df
    target = _find_target([c for c in df.columns if c not in long_cols])
    stacked = df[long_cols].stack().map(normalize_name)
    stacked = stacked[stacked != ""]
    dummies = pd.get_dummies(stacked)
    wide = dummies.groupby(level=0).max().astype(int).reindex(df.index, fill_value=0)
    wide[target] = df[target]
    logger.info("Converted long-format dataset (%d symptom columns) to wide format.", len(long_cols))
    return wide


# --------------------------------------------------------------------------- cleaning
def _to_binary(series: pd.Series) -> pd.Series:
    """Coerce a column to 0.0 / 1.0 (NaN stays NaN). Accepts numbers and yes/no/true/false text."""
    if series.dtype.kind in "biuf":
        numeric = series.astype(float)
    else:
        text = series.astype("string").str.strip().str.lower()
        mapped = text.map(lambda v: v if pd.isna(v) else ("1" if v in _TRUE_WORDS else "0" if v in _FALSE_WORDS else v))
        numeric = pd.to_numeric(mapped, errors="coerce").astype(float)
    return (numeric > 0).astype(float).where(numeric.notna())


def _clean_gender(series: pd.Series) -> pd.Series:
    mapping = {"m": "male", "male": "male", "man": "male", "f": "female", "female": "female", "woman": "female"}
    text = series.astype("string").str.strip().str.lower()
    cleaned = text.map(lambda v: v if pd.isna(v) else ("" if v == "" else mapping.get(v, "other")))
    return cleaned.replace("", pd.NA).astype(object).where(cleaned.notna() & (cleaned != ""), np.nan)


def clean_dataset(df: pd.DataFrame, min_samples_per_class: int = 10) -> tuple[pd.DataFrame, DatasetSchema, dict]:
    """Clean a loaded dataset.

    Returns ``(clean_df, schema, report)`` where ``report`` documents every decision taken
    (rows removed, columns dropped, missing values found) so it can be shown on the dashboard.

    Missing-value policy
    --------------------
    * symptom cell empty/NaN -> treated as "not reported" (0). Imputed inside the pipeline.
    * age empty/NaN/out of range -> median of the *training* rows (inside the pipeline).
    * gender empty/NaN -> most frequent *training* category (inside the pipeline).
    * target empty -> row dropped (cannot learn from it).
    """
    df = df.copy()
    report: dict = {"rows_raw": int(len(df))}

    df = df.dropna(how="all").drop(columns=[c for c in df.columns if c in _IGNORED_COLUMNS])
    target_src = _find_target(list(df.columns))
    df = df.rename(columns={target_src: TARGET})

    # --- target
    labels = df[TARGET].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    keep = labels.notna() & (labels != "")
    report["rows_missing_target"] = int((~keep).sum())
    df = df.loc[keep].copy()
    df[TARGET] = labels.loc[keep].astype(object)

    # --- optional demographics
    gender_src = next((c for c in _GENDER_CANDIDATES if c in df.columns), None)
    if gender_src:
        df = df.rename(columns={gender_src: GENDER})
        df[GENDER] = _clean_gender(df[GENDER])
        if df[GENDER].isna().all():
            df = df.drop(columns=[GENDER])
    if AGE in df.columns:
        age = pd.to_numeric(df[AGE], errors="coerce").astype(float)
        age = age.where((age >= 0) & (age <= 120))
        df[AGE] = age
        if age.isna().all():
            df = df.drop(columns=[AGE])

    # --- symptoms = every remaining column
    symptom_cols = [c for c in df.columns if c not in (TARGET, AGE, GENDER)]
    if not symptom_cols:
        raise ValueError("No symptom columns found in the dataset.")
    for c in symptom_cols:
        df[c] = _to_binary(df[c])

    report["missing_values"] = {
        "symptoms": int(df[symptom_cols].isna().sum().sum()),
        "age": int(df[AGE].isna().sum()) if AGE in df.columns else None,
        "gender": int(df[GENDER].isna().sum()) if GENDER in df.columns else None,
    }

    # --- symptoms that are never present can never influence a prediction
    never_present = [c for c in symptom_cols if df[c].fillna(0).sum() == 0]
    if never_present:
        df = df.drop(columns=never_present)
        symptom_cols = [c for c in symptom_cols if c not in never_present]
    report["dropped_empty_symptom_columns"] = never_present

    # --- exact duplicates (removed BEFORE splitting to prevent train/test leakage)
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    # --- classes too small to split / evaluate
    counts = df[TARGET].value_counts()
    rare = counts[counts < min_samples_per_class]
    report["dropped_rare_classes"] = {str(k): int(v) for k, v in rare.items()}
    if len(rare):
        df = df[~df[TARGET].isin(rare.index)]
    if df[TARGET].nunique() < 2:
        raise ValueError("Fewer than two diseases with enough samples remain after cleaning.")

    df = df.reset_index(drop=True)
    schema = DatasetSchema(symptom_cols=symptom_cols, has_age=AGE in df.columns, has_gender=GENDER in df.columns)
    counts = df[TARGET].value_counts()
    report.update(
        rows=int(len(df)),
        n_classes=int(counts.size),
        n_symptoms=len(symptom_cols),
        class_counts={str(k): int(v) for k, v in counts.items()},
        imbalance_ratio=float(counts.max() / counts.min()),
    )
    # Column order: features then target
    df = df[schema.feature_cols + [TARGET]]
    return df, schema, report


# --------------------------------------------------------------------------- features + split
def build_preprocessor(schema: DatasetSchema) -> ColumnTransformer:
    """Feature encoder; fitted only when the surrounding Pipeline is fitted (no leakage)."""
    transformers = [("symptoms", SimpleImputer(strategy="constant", fill_value=0.0), schema.symptom_cols)]
    if schema.has_age:
        transformers.append(("age", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), schema.numeric_cols))
    if schema.has_gender:
        transformers.append(("gender", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), schema.categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def split_data(df: pd.DataFrame, schema: DatasetSchema, test_size: float = 0.2,
               val_size: float = 0.2, seed: int = 42) -> SplitData:
    """Reproducible, stratified split: (train | validation) | test."""
    X, y = df[schema.feature_cols], df[schema.target]
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_size, stratify=y_dev, random_state=seed)
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)

"""Generate a SYNTHETIC demonstration dataset.

IMPORTANT
---------
The data produced here is *invented*. Symptom probabilities were written by hand
to look roughly plausible and are NOT derived from clinical records. The file
exists only so the application pipeline (cleaning -> training -> evaluation ->
web app) can be demonstrated end to end.

Any accuracy measured on this data says nothing about real-world performance.
For meaningful evaluation, replace ``data/disease_symptoms.csv`` with a real,
public dataset (see README) and run ``python -m src.train`` again.

Usage:
    python -m src.generate_sample_data            # writes data/disease_symptoms.csv
    python -m src.generate_sample_data --rows 6000 --seed 7
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_META_PATH, DATA_PATH

# Probability that a symptom is present, given the disease. Profiles intentionally
# overlap (fever, cough, fatigue ...) so the task is not trivially separable.
PROFILES: dict[str, dict[str, float]] = {
    "Common Cold": {"runny_nose": .90, "nasal_congestion": .85, "sneezing": .80, "sore_throat": .70,
                    "cough": .60, "headache": .25, "fatigue": .30, "fever": .15},
    "Influenza": {"fever": .85, "high_fever": .40, "chills": .70, "muscle_pain": .80, "fatigue": .85,
                  "headache": .60, "cough": .75, "dry_cough": .50, "sore_throat": .40,
                  "loss_of_appetite": .50, "runny_nose": .30},
    "Allergic Rhinitis": {"sneezing": .90, "itchy_eyes": .85, "runny_nose": .80, "nasal_congestion": .70,
                          "headache": .20, "cough": .20, "fatigue": .20},
    "Migraine": {"severe_headache": .95, "nausea": .70, "light_sensitivity": .80, "vomiting": .35,
                 "dizziness": .35, "fatigue": .30},
    "Gastroenteritis": {"diarrhea": .90, "vomiting": .80, "nausea": .85, "abdominal_pain": .80,
                        "fever": .40, "fatigue": .50, "loss_of_appetite": .60, "muscle_pain": .20},
    "Urinary Tract Infection": {"burning_urination": .92, "frequent_urination": .85,
                                "lower_abdominal_pain": .70, "fever": .25, "nausea": .15, "fatigue": .25},
    "Pneumonia": {"cough": .90, "productive_cough": .75, "fever": .80, "high_fever": .35,
                  "shortness_of_breath": .70, "chest_pain": .50, "chills": .55, "fatigue": .60,
                  "sweating": .40},
    "Dengue": {"high_fever": .90, "headache": .70, "pain_behind_eyes": .60, "joint_pain": .70,
               "muscle_pain": .70, "rash": .50, "nausea": .45, "vomiting": .30, "fatigue": .60,
               "loss_of_appetite": .40},
    "Typhoid": {"high_fever": .80, "abdominal_pain": .60, "headache": .55, "constipation": .40,
                "diarrhea": .30, "loss_of_appetite": .60, "fatigue": .60, "nausea": .40, "rash": .20,
                "chills": .30},
    "Asthma": {"wheezing": .90, "shortness_of_breath": .85, "chest_tightness": .80, "dry_cough": .60,
               "fatigue": .20},
}

# Deliberately imbalanced class frequencies (real data usually is too).
CLASS_WEIGHTS: dict[str, float] = {
    "Common Cold": .22, "Influenza": .15, "Allergic Rhinitis": .12, "Migraine": .12,
    "Gastroenteritis": .11, "Urinary Tract Infection": .09, "Pneumonia": .07,
    "Dengue": .05, "Typhoid": .04, "Asthma": .03,
}

# (mean, std) of age per disease - arbitrary, for demonstration of the optional age field.
AGE_PARAMS: dict[str, tuple[float, float]] = {
    "Common Cold": (28, 18), "Influenza": (35, 20), "Allergic Rhinitis": (30, 15), "Migraine": (36, 11),
    "Gastroenteritis": (30, 20), "Urinary Tract Infection": (42, 18), "Pneumonia": (55, 20),
    "Dengue": (30, 14), "Typhoid": (26, 12), "Asthma": (22, 15),
}

# Probability of "female" per disease - arbitrary, for demonstration of the optional gender field.
FEMALE_SHARE: dict[str, float] = {"Urinary Tract Infection": .80, "Migraine": .70}

BACKGROUND_NOISE = 0.03     # chance of any unrelated symptom being present
MISSING_AGE_RATE = 0.03     # a few blanks, to exercise missing-value handling
MISSING_GENDER_RATE = 0.02


def generate_sample_dataframe(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic dataset in the 'wide' format used by the project."""
    rng = np.random.default_rng(seed)
    diseases = list(CLASS_WEIGHTS)
    weights = np.array([CLASS_WEIGHTS[d] for d in diseases], dtype=float)
    weights /= weights.sum()
    labels = rng.choice(diseases, size=n_rows, p=weights)

    symptoms = sorted({s for profile in PROFILES.values() for s in profile})
    col = {s: i for i, s in enumerate(symptoms)}
    matrix = np.zeros((n_rows, len(symptoms)), dtype=np.int8)
    age = np.zeros(n_rows)
    female = np.zeros(n_rows, dtype=bool)

    for disease in diseases:
        rows = np.where(labels == disease)[0]
        probs = np.full(len(symptoms), BACKGROUND_NOISE)
        for symptom, p in PROFILES[disease].items():
            probs[col[symptom]] = p
        matrix[rows] = (rng.random((len(rows), len(symptoms))) < probs).astype(np.int8)

        # Every record should report at least one symptom.
        empty = rows[matrix[rows].sum(axis=1) == 0]
        matrix[empty, col[max(PROFILES[disease], key=PROFILES[disease].get)]] = 1

        mean, std = AGE_PARAMS[disease]
        age[rows] = np.clip(rng.normal(mean, std, len(rows)), 1, 95).round()
        female[rows] = rng.random(len(rows)) < FEMALE_SHARE.get(disease, .5)

    df = pd.DataFrame(matrix, columns=symptoms)
    df["age"] = pd.array(age.astype(int), dtype="Int64")
    df["gender"] = np.where(female, "female", "male").astype(object)
    df["disease"] = labels

    df.loc[rng.random(n_rows) < MISSING_AGE_RATE, "age"] = pd.NA
    df.loc[rng.random(n_rows) < MISSING_GENDER_RATE, "gender"] = np.nan
    return df


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file; used to detect whether data/disease_symptoms.csv is still the demo file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sample_dataset(path: str | Path = DATA_PATH, meta_path: str | Path = DATA_META_PATH,
                         n_rows: int = 4000, seed: int = 42) -> Path:
    """Write the synthetic CSV plus a small metadata file marking it as synthetic."""
    path, meta_path = Path(path), Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_dataframe(n_rows, seed).to_csv(path, index=False)
    meta = {
        "synthetic": True,
        "sha256": file_sha256(path),
        "rows": n_rows,
        "seed": seed,
        "note": "Invented demonstration data. NOT clinical data. Replace with a real public dataset.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic demo dataset.")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    out = write_sample_dataset(args.out, DATA_META_PATH if args.out == DATA_PATH else args.out.with_suffix(".meta.json"),
                               args.rows, args.seed)
    print(f"Wrote synthetic demo dataset to {out}")


if __name__ == "__main__":
    main()

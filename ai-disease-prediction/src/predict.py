"""Load the saved model bundle and turn user input into a prediction.

Contains:
* input validation (``validate_input``) - raises ``ValidationError`` with a user-friendly message;
* ``DiseasePredictor`` - wraps the joblib bundle produced by ``src.train``;
* helpers for the educational disease descriptions.

Security note: ``joblib.load`` unpickles data. Only load model files that you trained yourself.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.preprocessing import GENDER, AGE, VALID_GENDERS, humanize, normalize_name

logger = logging.getLogger(__name__)

MAX_SYMPTOMS = 40
LOW_CONFIDENCE = 0.40
DISCLAIMER = ("This is an educational demo. The output is a statistical estimate from a machine-learning model, "
              "NOT a medical diagnosis. Please consult a qualified healthcare professional.")


class ValidationError(ValueError):
    """Raised for invalid user input (maps to HTTP 400)."""


class ModelNotFoundError(RuntimeError):
    """Raised when the trained model file does not exist yet."""


# --------------------------------------------------------------------------- validation
def validate_input(payload: object, known_symptoms: list[str], uses_age: bool = False,
                   uses_gender: bool = False, max_symptoms: int = MAX_SYMPTOMS) -> dict:
    """Validate and normalise a request payload.

    Accepts: ``{"symptoms": [...], "age": 34 (optional), "gender": "female" (optional)}``
    Returns: ``{"symptoms": [...normalised, unique...], "age": float|None, "gender": str|None}``
    """
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    raw = payload.get("symptoms")
    if raw is None:
        raise ValidationError("Missing required field 'symptoms'.")
    if not isinstance(raw, list):
        raise ValidationError("'symptoms' must be a list of symptom names.")
    if len(raw) == 0:
        raise ValidationError("Select at least one symptom.")
    if len(raw) > max_symptoms * 2:
        raise ValidationError(f"Too many symptoms submitted (limit {max_symptoms}).")

    cleaned: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValidationError("Every symptom must be a string.")
        name = normalize_name(item[:100])
        if name and name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise ValidationError("Select at least one symptom.")
    if len(cleaned) > max_symptoms:
        raise ValidationError(f"Too many symptoms selected (limit {max_symptoms}).")

    known = set(known_symptoms)
    unknown = [s for s in cleaned if s not in known]
    if unknown:
        shown = ", ".join(unknown[:5]) + ("..." if len(unknown) > 5 else "")
        raise ValidationError(f"Unknown symptom(s): {shown}.")

    age = None
    raw_age = payload.get("age")
    if raw_age not in (None, ""):
        if isinstance(raw_age, bool):
            raise ValidationError("'age' must be a number.")
        try:
            age = float(raw_age)
        except (TypeError, ValueError):
            raise ValidationError("'age' must be a number.") from None
        if not np.isfinite(age) or not 0 <= age <= 120:
            raise ValidationError("'age' must be between 0 and 120.")

    gender = None
    raw_gender = payload.get("gender")
    if raw_gender not in (None, ""):
        if not isinstance(raw_gender, str):
            raise ValidationError("'gender' must be text.")
        gender = normalize_name(raw_gender)
        if gender in ("prefer_not_to_say", "unknown", "na", "n_a"):
            gender = None
        elif gender not in VALID_GENDERS:
            raise ValidationError(f"'gender' must be one of: {', '.join(VALID_GENDERS)}.")

    return {"symptoms": cleaned,
            "age": age if uses_age else None,
            "gender": gender if uses_gender else None}


# --------------------------------------------------------------------------- educational text
def load_disease_info(path: str | Path = config.DISEASE_INFO_PATH) -> dict:
    """Load general educational descriptions (keyed by normalised disease name)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Disease info file %s could not be read; using generic text.", path)
        return {}
    return {normalize_name(k): v for k, v in data.items()}


GENERIC_INFO = {
    "summary": ("No educational summary is available for this condition in this demo. "
                "A healthcare professional can explain what it means and how it is confirmed."),
    "typical_features": [],
    "seek_care": ["Symptoms are severe, getting worse or not improving.",
                  "You have difficulty breathing, chest pain, confusion or fainting."],
    "general_notes": ["Only a qualified clinician can diagnose a condition after an examination and, if needed, tests."],
}


def get_disease_info(disease_info: dict, disease: str) -> dict:
    return disease_info.get(normalize_name(disease), GENERIC_INFO)


# --------------------------------------------------------------------------- predictor
class DiseasePredictor:
    """Thin wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model_path: str | Path = config.MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}. Run `python -m src.train` first.")
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.classes: list[str] = list(bundle["classes"])
        self.symptoms: list[str] = list(bundle["symptoms"])
        self.uses_age: bool = bool(bundle.get("uses_age"))
        self.uses_gender: bool = bool(bundle.get("uses_gender"))
        self.model_name: str = bundle.get("model_name", "unknown")
        self.model_version: str = bundle.get("model_version", "unknown")
        self.trained_at: str = bundle.get("trained_at", "")
        self.is_synthetic: bool = bool(bundle.get("dataset_is_synthetic"))

    # -- helpers
    @property
    def symptom_options(self) -> list[dict]:
        return [{"value": s, "label": humanize(s)} for s in self.symptoms]

    def info(self) -> dict:
        return {"model": self.model_name, "version": self.model_version, "trained_at": self.trained_at,
                "n_diseases": len(self.classes), "n_symptoms": len(self.symptoms),
                "uses_age": self.uses_age, "uses_gender": self.uses_gender,
                "trained_on_synthetic_data": self.is_synthetic}

    def _frame(self, selected_sets: list[set[str]], age, gender) -> pd.DataFrame:
        rows = []
        for selected in selected_sets:
            row = {s: 1.0 if s in selected else 0.0 for s in self.symptoms}
            if self.uses_age:
                row[AGE] = np.nan if age is None else float(age)
            if self.uses_gender:
                row[GENDER] = np.nan if gender is None else gender
            rows.append(row)
        frame = pd.DataFrame(rows)
        if self.uses_gender:
            frame[GENDER] = frame[GENDER].astype(object)
        return frame

    # -- main entry point
    def predict(self, symptoms: list[str], age: float | None = None, gender: str | None = None,
                top_k: int = 5, explain: bool = True) -> dict:
        """Predict from already-validated input (see ``validate_input``)."""
        selected = list(symptoms)
        # Row 0 = the user's input. Rows 1..n = same input with ONE symptom removed (for explanations).
        variants = [set(selected)]
        if explain:
            variants += [set(selected) - {s} for s in selected]
        probs = self.pipeline.predict_proba(self._frame(variants, age, gender))
        classes = list(self.pipeline.classes_)

        full = probs[0]
        order = np.argsort(full)[::-1][:max(1, top_k)]
        top = [{"disease": str(classes[i]), "probability": round(float(full[i]), 4)} for i in order]
        best_idx = int(order[0])
        confidence = float(full[best_idx])

        explanation = []
        if explain:
            for symptom, without in zip(selected, probs[1:]):
                explanation.append({"symptom": symptom, "label": humanize(symptom),
                                    "effect": round(float(full[best_idx] - without[best_idx]), 4)})
            explanation.sort(key=lambda e: abs(e["effect"]), reverse=True)

        return {
            "prediction": top[0]["disease"],
            "confidence": round(confidence, 4),
            "top_predictions": top,
            "selected_symptoms": selected,
            "explanation": explanation,
            "low_confidence": confidence < LOW_CONFIDENCE,
            "model_version": self.model_version,
            "trained_on_synthetic_data": self.is_synthetic,
            "disclaimer": DISCLAIMER,
        }

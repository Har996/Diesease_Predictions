"""Central place for file paths and default settings.

Every other module imports paths from here so that file locations stay consistent.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Data -------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "disease_symptoms.csv"
DATA_META_PATH = DATA_DIR / "disease_symptoms.meta.json"  # written only for the synthetic demo data
DISEASE_INFO_PATH = DATA_DIR / "disease_info.json"

# --- Model artefacts --------------------------------------------------------
MODEL_DIR = ROOT_DIR / "models"
MODEL_FILENAME = "disease_model.joblib"
METRICS_FILENAME = "metrics.json"
REGISTRY_FILENAME = "registry.json"
VERSIONS_DIRNAME = "versions"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
METRICS_PATH = MODEL_DIR / METRICS_FILENAME

# --- Generated evaluation figures (served by Flask from /static) ------------
REPORT_DIR = ROOT_DIR / "static" / "img"

# --- Web app storage --------------------------------------------------------
DB_PATH = ROOT_DIR / "instance" / "app.sqlite3"

# --- Training defaults ------------------------------------------------------
RANDOM_STATE = 42      # fixed seed -> reproducible split and models
TEST_SIZE = 0.20       # share of all rows held out for the final, untouched test set
VAL_SIZE = 0.20        # share of the remaining rows used for model selection
MIN_SAMPLES_PER_CLASS = 10

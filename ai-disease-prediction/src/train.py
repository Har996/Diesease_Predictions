"""Train, compare and persist the disease classifier.

Run from the project root:

    python -m src.train                       # uses data/disease_symptoms.csv
    python -m src.train --data my.csv --seed 7 --cv-folds 5

Workflow
--------
1. Load + clean the dataset (``src.preprocessing``).
2. Stratified split: train / validation / test  (64 / 16 / 20 % by default, fixed seed).
3. Fit every candidate model on *train*; score on *validation*.
4. Choose the model with the best validation macro-F1 (accuracy breaks ties).
   (Macro-F1 gives every disease equal weight, which matters when classes are imbalanced.)
5. Refit the chosen pipeline on train + validation, then evaluate ONCE on the untouched test set.
6. Save the model bundle, versioned copy, registry entry, metrics JSON and evaluation figures.

Nothing in this file hard-codes a result: every number shown in the dashboard is computed here.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend - no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from src import config  # noqa: E402
from src.generate_sample_data import file_sha256, write_sample_dataset  # noqa: E402
from src.preprocessing import (  # noqa: E402
    DatasetSchema, build_preprocessor, clean_dataset, load_dataset, split_data,
)

logger = logging.getLogger("train")


# --------------------------------------------------------------------------- candidates
def build_candidates(seed: int) -> dict:
    """Models to compare. ``class_weight`` counteracts class imbalance."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=seed),
    }


def make_pipeline(schema: DatasetSchema, estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(schema)), ("clf", estimator)])


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    """All metrics are JSON-serialisable. 'macro' = unweighted mean over diseases."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "precision_weighted": float(pw), "recall_weighted": float(rw), "f1_weighted": float(fw),
        "per_class": {
            label: {"precision": float(report[label]["precision"]), "recall": float(report[label]["recall"]),
                    "f1": float(report[label]["f1-score"]), "support": int(report[label]["support"])}
            for label in labels
        },
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def native_importance(pipeline: Pipeline) -> list[dict] | None:
    """Model-specific importance (tree impurity / |coefficients|) if the estimator provides it."""
    clf = pipeline.named_steps["clf"]
    names = pipeline.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_).mean(axis=0)
    else:
        return None
    order = np.argsort(values)[::-1][:25]
    return [{"feature": str(names[i]), "importance": float(values[i])} for i in order]


def permutation_importances(pipeline: Pipeline, X_test, y_test, seed: int, top: int = 25) -> list[dict]:
    """Model-agnostic: drop in test macro-F1 when one input column is shuffled."""
    result = permutation_importance(pipeline, X_test, y_test, scoring="f1_macro", n_repeats=5,
                                    random_state=seed, n_jobs=1)
    order = np.argsort(result.importances_mean)[::-1][:top]
    return [{"feature": str(X_test.columns[i]), "importance": float(result.importances_mean[i]),
             "std": float(result.importances_std[i])} for i in order]


# --------------------------------------------------------------------------- figures
def save_figures(report_dir: Path, test_metrics: dict, candidates: list[dict], importance: list[dict]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    labels = test_metrics["labels"]
    cm = np.array(test_metrics["confusion_matrix"])

    size = max(6, 0.55 * len(labels) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    sns.heatmap(cm, annot=len(labels) <= 25, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Test samples"}, ax=ax)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("Actual disease")
    ax.set_title("Confusion matrix (held-out test set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [c["name"] for c in candidates]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [c["val"]["accuracy"] for c in candidates], 0.4, label="Accuracy", color="#4636C9")
    ax.bar(x + 0.2, [c["val"]["f1_macro"] for c in candidates], 0.4, label="Macro F1", color="#9A8FEA")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the validation set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "model_comparison.png", dpi=140)
    plt.close(fig)

    top = importance[:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([t["feature"] for t in top], [t["importance"] for t in top], color="#4636C9")
    ax.set_xlabel("Drop in test macro-F1 when shuffled (permutation importance)")
    ax.set_title("Most influential input features")
    fig.tight_layout()
    fig.savefig(report_dir / "feature_importance.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- training
def _is_synthetic(data_path: Path, meta_path: Path) -> bool:
    """True only if the CSV is byte-identical to the demo file we generated."""
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return bool(meta.get("synthetic")) and meta.get("sha256") == file_sha256(data_path)
    except (OSError, ValueError):
        return False


def run_training(data_path: Path = config.DATA_PATH, model_dir: Path = config.MODEL_DIR,
                 report_dir: Path = config.REPORT_DIR, meta_path: Path = config.DATA_META_PATH,
                 seed: int = config.RANDOM_STATE, test_size: float = config.TEST_SIZE,
                 val_size: float = config.VAL_SIZE, cv_folds: int = 5,
                 min_samples_per_class: int = config.MIN_SAMPLES_PER_CLASS, make_plots: bool = True,
                 generate_sample_if_missing: bool = True) -> dict:
    """Run the full pipeline and return the metrics dictionary that was written to disk."""
    data_path, model_dir, report_dir = Path(data_path), Path(model_dir), Path(report_dir)

    if not data_path.exists():
        if not generate_sample_if_missing:
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        logger.warning("No dataset at %s - generating a SYNTHETIC demo dataset. "
                       "Use a real public dataset for meaningful results.", data_path)
        write_sample_dataset(data_path, meta_path)

    # 1. load + clean -----------------------------------------------------------------
    raw = load_dataset(data_path)
    df, schema, data_report = clean_dataset(raw, min_samples_per_class)
    logger.info("Dataset: %d rows, %d diseases, %d symptoms (imbalance ratio %.1f:1)",
                data_report["rows"], data_report["n_classes"], data_report["n_symptoms"],
                data_report["imbalance_ratio"])

    # 2. split -------------------------------------------------------------------------
    split = split_data(df, schema, test_size, val_size, seed)
    labels = sorted(df[schema.target].unique())

    # 3-4. compare candidates on the validation set ---------------------------------------
    candidates = []
    fitted = {}
    folds = min(cv_folds, int(split.y_dev.value_counts().min())) if cv_folds else 0
    for name, estimator in build_candidates(seed).items():
        pipe = make_pipeline(schema, estimator)
        started = time.perf_counter()
        pipe.fit(split.X_train, split.y_train)
        fit_seconds = time.perf_counter() - started
        val_metrics = compute_metrics(split.y_val, pipe.predict(split.X_val), labels)
        entry = {"name": name, "fit_seconds": round(fit_seconds, 2), "val": val_metrics}
        if folds >= 2:  # extra stability check on train+validation only (never the test set)
            cv = cross_val_score(clone(pipe), split.X_dev, split.y_dev, scoring="f1_macro", n_jobs=1,
                                 cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
            entry.update(cv_folds=folds, cv_f1_macro_mean=float(cv.mean()), cv_f1_macro_std=float(cv.std()))
        candidates.append(entry)
        fitted[name] = pipe
        logger.info("%-20s val acc=%.3f  macro-F1=%.3f", name, val_metrics["accuracy"], val_metrics["f1_macro"])

    best = max(candidates, key=lambda c: (c["val"]["f1_macro"], c["val"]["accuracy"]))
    logger.info("Selected model: %s", best["name"])

    # 5. refit on train+validation, evaluate once on the test set -------------------------------
    final = clone(fitted[best["name"]])
    final.fit(split.X_dev, split.y_dev)
    test_metrics = compute_metrics(split.y_test, final.predict(split.X_test), labels)
    logger.info("TEST  acc=%.3f  macro-F1=%.3f", test_metrics["accuracy"], test_metrics["f1_macro"])

    permutation = permutation_importances(final, split.X_test, split.y_test, seed)

    # 6. persist ---------------------------------------------------------------------------
    now = datetime.now(timezone.utc)
    version = now.strftime("%Y%m%d-%H%M%S")
    synthetic = _is_synthetic(data_path, meta_path)
    notes = [
        "Metrics come from a single stratified hold-out split; they carry sampling uncertainty.",
        "High accuracy on a dataset does not make a model medically reliable.",
    ]
    if synthetic:
        notes.insert(0, "The dataset is SYNTHETIC demo data. These metrics say nothing about real-world accuracy.")
    if data_report["imbalance_ratio"] >= 3:
        notes.append(f"Classes are imbalanced ({data_report['imbalance_ratio']:.1f}:1); class weights were "
                     "used and macro-averaged metrics are reported.")

    metrics = {
        "version": version,
        "trained_at": now.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": seed,
        "dataset": {**data_report, "path": str(data_path.name), "uses_age": schema.has_age,
                    "uses_gender": schema.has_gender, "is_synthetic": synthetic},
        "split": {"train": len(split.X_train), "validation": len(split.X_val), "test": len(split.X_test),
                  "scheme": f"stratified, seed={seed}, test={test_size:.0%} of all rows, "
                            f"validation={val_size:.0%} of the remainder"},
        "candidates": candidates,
        "selected_model": best["name"],
        "selection_criterion": "highest validation macro-F1 (accuracy breaks ties)",
        "test": test_metrics,
        "feature_importance": {"permutation": permutation, "native": native_importance(final)},
        "notes": notes,
    }

    bundle = {
        "pipeline": final,
        "classes": [str(c) for c in final.classes_],
        "symptoms": schema.symptom_cols,
        "uses_age": schema.has_age,
        "uses_gender": schema.has_gender,
        "model_name": best["name"],
        "model_version": version,
        "trained_at": metrics["trained_at"],
        "sklearn_version": sklearn.__version__,
        "dataset_is_synthetic": synthetic,
        "random_state": seed,
    }

    versions_dir = model_dir / config.VERSIONS_DIRNAME
    versions_dir.mkdir(parents=True, exist_ok=True)
    versioned_model = versions_dir / f"disease_model_{version}.joblib"
    joblib.dump(bundle, versioned_model)
    shutil.copyfile(versioned_model, model_dir / config.MODEL_FILENAME)  # "latest" = what the web app loads

    (model_dir / config.METRICS_FILENAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (versions_dir / f"metrics_{version}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    registry_path = model_dir / config.REGISTRY_FILENAME
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({
        "version": version, "model": best["name"], "trained_at": metrics["trained_at"],
        "file": f"{config.VERSIONS_DIRNAME}/{versioned_model.name}",
        "test_accuracy": test_metrics["accuracy"], "test_f1_macro": test_metrics["f1_macro"],
        "n_classes": data_report["n_classes"], "n_symptoms": data_report["n_symptoms"],
        "dataset_is_synthetic": synthetic, "sklearn_version": sklearn.__version__,
    })
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    if make_plots:
        save_figures(report_dir, test_metrics, candidates, permutation)

    logger.info("Saved model %s (version %s) to %s", best["name"], version, model_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare disease classifiers.")
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=config.MODEL_DIR)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--cv-folds", type=int, default=5, help="0 disables cross-validation")
    parser.add_argument("--min-samples", type=int, default=config.MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-sample-fallback", action="store_true",
                        help="fail instead of generating synthetic data when the CSV is missing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = run_training(args.data, args.model_dir, seed=args.seed, cv_folds=args.cv_folds,
                           min_samples_per_class=args.min_samples, make_plots=not args.no_plots,
                           generate_sample_if_missing=not args.no_sample_fallback)
    t = metrics["test"]
    print(f"\nSelected: {metrics['selected_model']}  |  test accuracy {t['accuracy']:.3f}  |  "
          f"macro precision {t['precision_macro']:.3f}  recall {t['recall_macro']:.3f}  F1 {t['f1_macro']:.3f}")
    if metrics["dataset"]["is_synthetic"]:
        print("NOTE: trained on SYNTHETIC demo data - these numbers are not meaningful for real-world use.")


if __name__ == "__main__":
    main()

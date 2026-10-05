"""Unit and API tests.

Run either way from the project root:
    python -m unittest discover -s tests -v
    pytest -q

A small model is trained once (module fixture) on a synthetic dataset inside a temporary
directory, so the tests never touch your real data/ or models/ folders.
"""
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import create_app  # noqa: E402
from src import config  # noqa: E402
from src.generate_sample_data import generate_sample_dataframe, write_sample_dataset  # noqa: E402
from src.predict import DiseasePredictor, ValidationError, validate_input  # noqa: E402
from src.preprocessing import (AGE, build_preprocessor, clean_dataset, load_dataset,  # noqa: E402
                               normalize_name, split_data)
from src.train import run_training  # noqa: E402

TMP: Path
MODEL_DIR: Path
METRICS: dict


def setUpModule():
    """Train one small model for all tests (fast: 1,500 synthetic rows, no CV, no plots)."""
    global TMP, MODEL_DIR, METRICS
    TMP = Path(tempfile.mkdtemp(prefix="dp-tests-"))
    MODEL_DIR = TMP / "models"
    data = TMP / "data.csv"
    write_sample_dataset(data, TMP / "data.meta.json", n_rows=1500, seed=1)
    METRICS = run_training(data, MODEL_DIR, TMP / "img", TMP / "data.meta.json", seed=7,
                           cv_folds=0, make_plots=False)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


def csrf_from(client, path="/login"):
    html = client.get(path).get_data(as_text=True)
    return re.search(r'name="_csrf" value="([^"]+)"', html).group(1)


# =============================================================================== preprocessing
class PreprocessingTests(unittest.TestCase):
    def test_normalize_name(self):
        self.assertEqual(normalize_name("  Skin Rash "), "skin_rash")
        self.assertEqual(normalize_name("Joint-Pain!"), "joint_pain")

    def test_clean_removes_duplicates_missing_target_and_rare_classes(self):
        rows = [{"fever": 1, "cough": 0, "disease": "A"}] * 3          # exact duplicates
        rows += [{"fever": i % 2, "cough": (i // 2) % 2, "disease": "A"} for i in range(40)]
        rows += [{"fever": 0, "cough": 1, "disease": "B"} for _ in range(2)]  # rare class + dupes
        rows += [{"fever": 1, "cough": 1, "disease": None}]            # missing target
        rows += [{"fever": 1, "cough": i % 2, "disease": "C"} for i in range(30)]
        df = pd.DataFrame(rows)
        # make class C distinguishable so duplicates do not collapse it below the threshold
        df.loc[df.disease == "C", "age"] = np.arange(30)
        clean, schema, report = clean_dataset(df, min_samples_per_class=2)
        self.assertEqual(report["rows_missing_target"], 1)
        self.assertGreaterEqual(report["duplicates_removed"], 3)
        self.assertNotIn(None, clean["disease"].tolist())
        self.assertEqual(set(schema.symptom_cols), {"fever", "cough"})
        self.assertTrue(schema.has_age)

    def test_clean_drops_rare_classes(self):
        df = generate_sample_dataframe(800, seed=2)
        rare = df.head(3).copy()
        rare["disease"] = "Very Rare Disease"
        _, schema, report = clean_dataset(pd.concat([df, rare], ignore_index=True), min_samples_per_class=5)
        self.assertIn("Very Rare Disease", report["dropped_rare_classes"])
        self.assertNotIn("Very Rare Disease", report["class_counts"])

    def test_long_format_is_converted_to_wide(self):
        path = TMP / "long.csv"
        rows = ["Disease,Symptom_1,Symptom_2,Symptom_3"]
        for i in range(30):
            rows.append("Flu, fever, cough ,")
            rows.append("Migraine, severe headache, nausea, " + ("vomiting" if i % 2 else ""))
        path.write_text("\n".join(rows), encoding="utf-8")
        wide = load_dataset(path)
        self.assertTrue({"fever", "cough", "severe_headache", "nausea", "vomiting"} <= set(wide.columns))
        self.assertEqual(int(wide.loc[0, "fever"]), 1)

    def test_missing_file_raises_helpful_error(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(TMP / "does_not_exist.csv")

    def test_split_is_reproducible_stratified_and_disjoint(self):
        df, schema, _ = clean_dataset(generate_sample_dataframe(1200, seed=3))
        a = split_data(df, schema, seed=11)
        b = split_data(df, schema, seed=11)
        self.assertEqual(list(a.X_test.index), list(b.X_test.index))
        train, val, test = set(a.X_train.index), set(a.X_val.index), set(a.X_test.index)
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train | val | test), len(df))
        self.assertEqual(set(a.y_test.unique()), set(df["disease"].unique()))

    def test_preprocessor_learns_only_from_training_rows(self):
        df, schema, _ = clean_dataset(generate_sample_dataframe(1200, seed=3))
        split = split_data(df, schema, seed=5)
        prep = build_preprocessor(schema).fit(split.X_train)
        learned = prep.named_transformers_["age"].named_steps["impute"].statistics_[0]
        self.assertAlmostEqual(learned, split.X_train[AGE].median())


# =============================================================================== training
class TrainingTests(unittest.TestCase):
    def test_artifacts_are_written_and_versioned(self):
        self.assertTrue((MODEL_DIR / "disease_model.joblib").exists())
        self.assertTrue((MODEL_DIR / "metrics.json").exists())
        registry = json.loads((MODEL_DIR / "registry.json").read_text())
        self.assertEqual(registry[-1]["version"], METRICS["version"])
        self.assertTrue((MODEL_DIR / registry[-1]["file"]).exists())

    def test_all_candidates_compared_and_best_validation_model_selected(self):
        names = [c["name"] for c in METRICS["candidates"]]
        self.assertEqual(len(names), 3)
        best = max(METRICS["candidates"], key=lambda c: (c["val"]["f1_macro"], c["val"]["accuracy"]))
        self.assertEqual(METRICS["selected_model"], best["name"])

    def test_metrics_are_consistent(self):
        t = METRICS["test"]
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            self.assertGreaterEqual(t[key], 0.0)
            self.assertLessEqual(t[key], 1.0)
        cm = np.array(t["confusion_matrix"])
        self.assertEqual(cm.shape, (METRICS["dataset"]["n_classes"],) * 2)
        self.assertEqual(int(cm.sum()), METRICS["split"]["test"])
        self.assertAlmostEqual(np.trace(cm) / cm.sum(), t["accuracy"], places=6)
        self.assertTrue(METRICS["dataset"]["is_synthetic"])

    def test_training_is_reproducible(self):
        again = run_training(TMP / "data.csv", TMP / "models_again", TMP / "img2", TMP / "data.meta.json",
                             seed=7, cv_folds=0, make_plots=False)
        self.assertEqual(again["test"]["confusion_matrix"], METRICS["test"]["confusion_matrix"])


# =============================================================================== validation + predictor
class ValidationTests(unittest.TestCase):
    KNOWN = ["fever", "cough", "fatigue"]

    def test_normalises_and_deduplicates(self):
        out = validate_input({"symptoms": ["Fever", " cough ", "FEVER"]}, self.KNOWN)
        self.assertEqual(out["symptoms"], ["fever", "cough"])

    def test_rejects_bad_payloads(self):
        bad = [None, [], "fever", {}, {"symptoms": "fever"}, {"symptoms": []}, {"symptoms": [""]},
               {"symptoms": [1]}, {"symptoms": ["nope"]}, {"symptoms": ["fever"] * 1000}]
        for payload in bad:
            with self.subTest(payload=str(payload)[:40]), self.assertRaises(ValidationError):
                validate_input(payload, self.KNOWN)

    def test_age_and_gender_rules(self):
        ok = validate_input({"symptoms": ["fever"], "age": "34", "gender": "Female"}, self.KNOWN, True, True)
        self.assertEqual((ok["age"], ok["gender"]), (34.0, "female"))
        for age in (-1, 121, "abc", True, float("nan")):
            with self.subTest(age=age), self.assertRaises(ValidationError):
                validate_input({"symptoms": ["fever"], "age": age}, self.KNOWN, True, True)
        with self.assertRaises(ValidationError):
            validate_input({"symptoms": ["fever"], "gender": "robot"}, self.KNOWN, True, True)
        blank = validate_input({"symptoms": ["fever"], "age": "", "gender": "prefer not to say"}, self.KNOWN, True, True)
        self.assertEqual((blank["age"], blank["gender"]), (None, None))

    def test_age_ignored_when_model_does_not_use_it(self):
        out = validate_input({"symptoms": ["fever"], "age": 30, "gender": "male"}, self.KNOWN, False, False)
        self.assertEqual((out["age"], out["gender"]), (None, None))


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = DiseasePredictor(MODEL_DIR / "disease_model.joblib")

    def test_output_structure_and_probabilities(self):
        res = self.predictor.predict(["fever", "cough"], top_k=len(self.predictor.classes))
        probs = [t["probability"] for t in res["top_predictions"]]
        self.assertAlmostEqual(sum(probs), 1.0, places=2)
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(res["prediction"], res["top_predictions"][0]["disease"])
        self.assertEqual(res["confidence"], probs[0])
        self.assertIn("not a medical diagnosis", res["disclaimer"].lower())

    def test_prediction_is_deterministic(self):
        a = self.predictor.predict(["fever", "cough", "fatigue"], age=40, gender="male")
        b = self.predictor.predict(["fever", "cough", "fatigue"], age=40, gender="male")
        self.assertEqual(a, b)

    def test_optional_fields_may_be_missing(self):
        res = self.predictor.predict(["wheezing"], age=None, gender=None)
        self.assertIn(res["prediction"], self.predictor.classes)

    def test_pipeline_sanity_on_synthetic_profile(self):
        # Pipeline-wiring check on the synthetic data only - NOT a clinical claim.
        res = self.predictor.predict(["wheezing", "chest_tightness", "shortness_of_breath"])
        self.assertEqual(res["prediction"], "Asthma")

    def test_explanation_covers_each_selected_symptom(self):
        res = self.predictor.predict(["fever", "cough"])
        self.assertEqual({e["symptom"] for e in res["explanation"]}, {"fever", "cough"})

    def test_missing_model_file_raises(self):
        from src.predict import ModelNotFoundError
        with self.assertRaises(ModelNotFoundError):
            DiseasePredictor(TMP / "nothing.joblib")


# =============================================================================== web app / API
class AppTestCase(unittest.TestCase):
    extra_config: dict = {}

    def setUp(self):
        self.db_dir = Path(tempfile.mkdtemp(prefix="dp-db-"))
        cfg = {"TESTING": True, "SECRET_KEY": "test", "DATABASE": str(self.db_dir / "t.sqlite3"),
               "MODEL_DIR": str(MODEL_DIR), **self.extra_config}
        self.app = create_app(cfg)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.db_dir, ignore_errors=True)

    def predict(self, payload, client=None):
        return (client or self.client).post("/predict", json=payload)


class ApiTests(AppTestCase):
    def test_health_and_pages(self):
        self.assertEqual(self.client.get("/health").json["model_loaded"], True)
        for path in ("/", "/predict", "/dashboard", "/history", "/api/docs", "/login", "/register"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_home_shows_title_and_disclaimer(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("AI Disease Prediction System", html)
        self.assertIn("not a medical diagnosis", html.lower())

    def test_symptoms_and_model_endpoints(self):
        syms = self.client.get("/api/symptoms").json
        self.assertEqual(syms["count"], len(syms["symptoms"]))
        self.assertIn("n_diseases", self.client.get("/api/model").json)
        self.assertEqual(self.client.get("/api/openapi.json").json["openapi"], "3.0.3")

    def test_predict_success_matches_documented_shape(self):
        res = self.predict({"symptoms": ["fever", "cough", "fatigue"], "age": 30, "gender": "female"})
        self.assertEqual(res.status_code, 200)
        body = res.json
        for key in ("prediction", "confidence", "top_predictions", "disclaimer"):
            self.assertIn(key, body)
        self.assertTrue(0 <= body["confidence"] <= 1)
        self.assertEqual(set(body["top_predictions"][0]), {"disease", "probability"})

    def test_predict_rejects_invalid_input_gracefully(self):
        cases = [{"symptoms": []}, {}, {"symptoms": ["not_a_symptom"]}, {"symptoms": "fever"},
                 {"symptoms": ["fever"], "age": 500}, {"symptoms": ["fever"], "gender": "x"}, [1, 2]]
        for payload in cases:
            with self.subTest(payload=str(payload)):
                res = self.predict(payload)
                self.assertEqual(res.status_code, 400)
                self.assertIn("error", res.json)

    def test_predict_rejects_malformed_json_and_oversized_body(self):
        bad = self.client.post("/predict", data="{not json", content_type="application/json")
        self.assertEqual(bad.status_code, 400)
        big = self.client.post("/predict", json={"symptoms": ["fever"], "pad": "x" * 100_000})
        self.assertEqual(big.status_code, 413)

    def test_save_false_skips_history(self):
        res = self.predict({"symptoms": ["fever"], "save": False})
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("id", res.json)

    def test_unknown_route_returns_json_for_api_paths(self):
        res = self.client.get("/api/does-not-exist")
        self.assertEqual(res.status_code, 404)
        self.assertIn("error", res.json)

    def test_security_headers(self):
        res = self.client.get("/")
        self.assertEqual(res.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(res.headers["X-Frame-Options"], "DENY")


class MissingModelTests(AppTestCase):
    extra_config = {"MODEL_DIR": "/nonexistent-model-dir"}

    def test_api_returns_503_and_pages_still_render(self):
        self.assertEqual(self.predict({"symptoms": ["fever"]}).status_code, 503)
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/predict").status_code, 200)
        self.assertEqual(self.client.get("/dashboard").status_code, 200)
        self.assertEqual(self.client.get("/api/symptoms").status_code, 503)


class ResultAndHistoryTests(AppTestCase):
    def test_result_page_and_report(self):
        res = self.predict({"symptoms": ["wheezing", "chest_tightness"]}).json
        page = self.client.get(res["result_url"])
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn(res["prediction"], html)
        self.assertIn("not a diagnosis", html.lower())
        self.assertIn("healthcare professional", html.lower())
        report = self.client.get(res["result_url"] + "/report")
        self.assertIn("attachment", report.headers["Content-Disposition"])

    def test_results_are_private_to_their_owner(self):
        res = self.predict({"symptoms": ["fever"]}).json
        stranger = self.app.test_client()
        self.assertEqual(stranger.get(res["result_url"]).status_code, 404)
        self.assertEqual(stranger.get(res["result_url"] + "/report").status_code, 404)

    def test_history_lists_and_clears_with_csrf(self):
        self.predict({"symptoms": ["fever"]})
        self.assertIn("Open", self.client.get("/history").get_data(as_text=True))
        self.assertEqual(self.client.post("/history/clear").status_code, 400)       # no CSRF token
        token = csrf_from(self.client, "/history")
        self.client.post("/history/clear", data={"_csrf": token})
        self.assertIn("No predictions saved yet", self.client.get("/history").get_data(as_text=True))

    def test_html_form_flow_redirects_to_result(self):
        token = csrf_from(self.client, "/predict")
        res = self.client.post("/predict", data={"_csrf": token, "symptoms": ["fever", "cough"]})
        self.assertEqual(res.status_code, 302)
        self.assertIn("/result/", res.headers["Location"])

    def test_html_form_errors_are_shown(self):
        token = csrf_from(self.client, "/predict")
        res = self.client.post("/predict", data={"_csrf": token})
        self.assertEqual(res.status_code, 400)
        self.assertIn("Select at least one symptom", res.get_data(as_text=True))
        self.assertEqual(self.client.post("/predict", data={"symptoms": ["fever"]}).status_code, 400)  # no CSRF


class AuthTests(AppTestCase):
    def register(self, username="alice", password="correct horse", confirm=None, client=None):
        client = client or self.client
        return client.post("/register", data={"_csrf": csrf_from(client, "/register"), "username": username,
                                              "password": password, "confirm": confirm or password})

    def test_register_login_logout(self):
        self.assertEqual(self.register().status_code, 302)
        self.assertIn("alice", self.client.get("/").get_data(as_text=True))
        self.client.post("/logout", data={"_csrf": csrf_from(self.client, "/")})
        self.assertNotIn("Signed in as", self.client.get("/").get_data(as_text=True))
        ok = self.client.post("/login", data={"_csrf": csrf_from(self.client), "username": "ALICE", "password": "correct horse"})
        self.assertEqual(ok.status_code, 302)

    def test_rejects_bad_registrations(self):
        self.assertIn("at least 8", self.register(password="short", confirm="short").get_data(as_text=True))
        self.assertIn("do not match", self.register(password="longenough1", confirm="different1").get_data(as_text=True))
        self.assertIn("Username must be", self.register(username="a b").get_data(as_text=True))
        self.register()
        other = self.app.test_client()
        self.assertIn("already taken", self.register(client=other).get_data(as_text=True))

    def test_wrong_password_gives_generic_error(self):
        self.register()
        other = self.app.test_client()
        res = other.post("/login", data={"_csrf": csrf_from(other), "username": "alice", "password": "wrong-pass"})
        self.assertIn("Incorrect username or password", res.get_data(as_text=True))
        res = other.post("/login", data={"_csrf": csrf_from(other), "username": "nobody", "password": "wrong-pass"})
        self.assertIn("Incorrect username or password", res.get_data(as_text=True))

    def test_login_requires_csrf(self):
        self.assertEqual(self.client.post("/login", data={"username": "a", "password": "b"}).status_code, 400)

    def test_anonymous_history_is_adopted_on_registration(self):
        res = self.predict({"symptoms": ["fever"]}).json
        self.register()
        self.assertEqual(self.client.get(res["result_url"]).status_code, 200)


class DashboardAccessTests(AppTestCase):
    extra_config = {"DASHBOARD_REQUIRE_ADMIN": True, "ADMIN_USERNAME": "boss", "ADMIN_PASSWORD": "admin-password-1"}

    def test_dashboard_requires_admin(self):
        self.assertEqual(self.client.get("/dashboard").status_code, 302)           # -> login
        user = self.app.test_client()
        user.post("/register", data={"_csrf": csrf_from(user, "/register"), "username": "bob",
                                     "password": "bobs-password", "confirm": "bobs-password"})
        self.assertEqual(user.get("/dashboard").status_code, 403)
        admin = self.app.test_client()
        admin.post("/login", data={"_csrf": csrf_from(admin), "username": "boss", "password": "admin-password-1"})
        page = admin.get("/dashboard")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Confusion matrix", page.get_data(as_text=True))

    def test_admin_username_cannot_be_registered(self):
        c = self.app.test_client()
        res = c.post("/register", data={"_csrf": csrf_from(c, "/register"), "username": "Boss",
                                        "password": "whatever-123", "confirm": "whatever-123"})
        self.assertIn("reserved", res.get_data(as_text=True))


class DashboardContentTests(AppTestCase):
    def test_dashboard_reports_computed_metrics(self):
        html = self.client.get("/dashboard").get_data(as_text=True)
        for text in ("Accuracy", "Precision", "Recall", "F1-score", "Confusion matrix", "Feature importance",
                     "synthetic"):
            self.assertIn(text, html)
        self.assertIn(METRICS["selected_model"], html)


if __name__ == "__main__":
    unittest.main(verbosity=2)

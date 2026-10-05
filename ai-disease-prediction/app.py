"""Flask application: pages, JSON API, authentication, prediction history and model dashboard.

Run (development):   python app.py        or     flask --app app run --debug
Run (production):    gunicorn "app:create_app()" --bind 0.0.0.0:8000

Educational demo only - predictions are NOT medical diagnoses.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import secrets
from functools import wraps
from pathlib import Path

from flask import (Flask, abort, flash, g, jsonify, make_response, redirect, render_template,
                   request, session, url_for)
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from src import config, database
from src.api_spec import OPENAPI_SPEC
from src.predict import (DISCLAIMER, DiseasePredictor, ModelNotFoundError, ValidationError,
                         get_disease_info, load_disease_info, validate_input)
from src.preprocessing import humanize

logger = logging.getLogger(__name__)
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        DATABASE=os.environ.get("DATABASE_PATH", str(config.DB_PATH)),
        MODEL_DIR=os.environ.get("MODEL_DIR", str(config.MODEL_DIR)),
        DISEASE_INFO_PATH=str(config.DISEASE_INFO_PATH),
        MAX_CONTENT_LENGTH=64 * 1024,                 # reject oversized request bodies
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=_env_flag("SESSION_COOKIE_SECURE"),
        DASHBOARD_REQUIRE_ADMIN=_env_flag("DASHBOARD_REQUIRE_ADMIN"),
        ADMIN_USERNAME=os.environ.get("ADMIN_USERNAME", ""),
        ADMIN_PASSWORD=os.environ.get("ADMIN_PASSWORD", ""),
        TESTING=False,
    )
    if test_config:
        app.config.update(test_config)
    if app.config["SECRET_KEY"] == "dev-only-change-me" and not app.config["TESTING"]:
        logger.warning("Using the default SECRET_KEY. Set the SECRET_KEY environment variable in production.")

    database.init_app(app)
    app.extensions["predictor"] = None
    app.extensions["disease_info"] = load_disease_info(app.config["DISEASE_INFO_PATH"])
    _ensure_admin(app)

    # ------------------------------------------------------------------ helpers
    def model_dir() -> Path:
        return Path(app.config["MODEL_DIR"])

    def get_predictor() -> DiseasePredictor | None:
        """Load the model lazily so the app starts (and shows guidance) before the first training run."""
        if app.extensions["predictor"] is None:
            try:
                app.extensions["predictor"] = DiseasePredictor(model_dir() / config.MODEL_FILENAME)
            except ModelNotFoundError:
                return None
        return app.extensions["predictor"]

    def load_metrics() -> dict | None:
        try:
            return json.loads((model_dir() / config.METRICS_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def load_registry() -> list:
        try:
            return json.loads((model_dir() / config.REGISTRY_FILENAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def wants_json() -> bool:
        if request.is_json or request.path.startswith("/api/") or request.path == "/health":
            return True
        best = request.accept_mimetypes.best_match(["application/json", "text/html"])
        return best == "application/json" and request.accept_mimetypes[best] > request.accept_mimetypes["text/html"]

    def owner() -> tuple[int | None, str | None]:
        user_id = g.user["id"] if g.user else None
        return user_id, session.get("anon_id")

    # ------------------------------------------------------------------ request hooks
    @app.before_request
    def load_user_and_session() -> None:
        g.user = None
        if "user_id" in session:
            g.user = database.get_user(session["user_id"])
            if g.user is None:
                session.pop("user_id", None)

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    @app.context_processor
    def inject_globals() -> dict:
        predictor = app.extensions["predictor"] or get_predictor()
        return {
            "current_user": g.get("user"),
            "csrf_token": csrf_token,
            "disclaimer": DISCLAIMER,
            "model_synthetic": bool(predictor and predictor.is_synthetic),
            "humanize": humanize,
        }

    def csrf_token() -> str:
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_urlsafe(32)
        return session["_csrf"]

    def require_csrf() -> None:
        sent = request.form.get("_csrf", "")
        if not sent or not hmac.compare_digest(sent, session.get("_csrf", "")):
            abort(400, description="Your form session expired. Please go back and try again.")

    def admin_required(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if app.config["DASHBOARD_REQUIRE_ADMIN"]:
                if g.user is None:
                    flash("Sign in as the administrator to view the dashboard.", "warning")
                    return redirect(url_for("login"))
                if g.user["username"].lower() != app.config["ADMIN_USERNAME"].lower():
                    abort(403, description="The dashboard is restricted to the administrator.")
            return view(*args, **kwargs)
        return wrapper

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return render_template("index.html", predictor=get_predictor())

    @app.get("/predict")
    def predict_page():
        predictor = get_predictor()
        return render_template("predict.html", predictor=predictor)

    @app.post("/predict")
    def predict():
        """JSON API (Content-Type: application/json) and HTML form fallback share this endpoint."""
        is_api = request.is_json
        predictor = get_predictor()

        def fail(message: str, status: int):
            if is_api:
                return jsonify(error=message), status
            flash(message, "danger")
            return render_template("predict.html", predictor=predictor, selected=request.form.getlist("symptoms"),
                                   form=request.form), status

        if predictor is None:
            return fail("The model has not been trained yet. Run `python -m src.train`.", 503)

        if is_api:
            payload = request.get_json(silent=True)
            if payload is None:
                return jsonify(error="Request body must be valid JSON."), 400
            save = payload.get("save", True) if isinstance(payload, dict) else True
        else:
            if not request.form.get("_csrf") or not hmac.compare_digest(request.form["_csrf"], session.get("_csrf", "")):
                return fail("Your form session expired. Please try again.", 400)
            payload = {"symptoms": request.form.getlist("symptoms"), "age": request.form.get("age"),
                       "gender": request.form.get("gender")}
            save = True

        try:
            cleaned = validate_input(payload, predictor.symptoms, predictor.uses_age, predictor.uses_gender)
            result = predictor.predict(**cleaned)
        except ValidationError as exc:
            return fail(str(exc), 400)
        except Exception:  # model/pipeline failure - never leak internals to the user
            logger.exception("Prediction failed")
            return fail("Something went wrong while running the model.", 500)

        if save:
            session.setdefault("anon_id", secrets.token_hex(16))
            user_id, anon_id = owner()
            result["id"] = database.save_prediction(result, cleaned, user_id, anon_id)
            result["result_url"] = url_for("result_page", pred_id=result["id"])

        if is_api:
            return jsonify(result)
        return redirect(result["result_url"])

    @app.get("/result/<int:pred_id>")
    def result_page(pred_id: int):
        row = database.get_prediction(pred_id, *owner())
        if row is None:
            abort(404)
        record = database.row_to_record(row)
        info = get_disease_info(app.extensions["disease_info"], record["prediction"])
        return render_template("result.html", r=record, info=info)

    @app.get("/result/<int:pred_id>/report")
    def download_report(pred_id: int):
        row = database.get_prediction(pred_id, *owner())
        if row is None:
            abort(404)
        record = database.row_to_record(row)
        info = get_disease_info(app.extensions["disease_info"], record["prediction"])
        resp = make_response(render_template("report.html", r=record, info=info))
        resp.headers["Content-Disposition"] = f'attachment; filename="prediction-report-{pred_id}.html"'
        return resp

    @app.get("/history")
    def history():
        rows = [database.row_to_record(r) for r in database.list_predictions(*owner())]
        return render_template("history.html", records=rows)

    @app.post("/history/clear")
    def clear_history():
        require_csrf()
        removed = database.delete_predictions(*owner())
        flash(f"Deleted {removed} saved prediction(s).", "success")
        return redirect(url_for("history"))

    @app.get("/dashboard")
    @admin_required
    def dashboard():
        return render_template("dashboard.html", metrics=load_metrics(), registry=load_registry()[::-1])

    # ------------------------------------------------------------------ authentication
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            require_csrf()
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            error = None
            if not USERNAME_RE.match(username):
                error = "Username must be 3-32 characters: letters, digits, dot, dash or underscore."
            elif len(password) < 8:
                error = "Password must be at least 8 characters."
            elif password != request.form.get("confirm", ""):
                error = "Passwords do not match."
            elif app.config["ADMIN_USERNAME"] and username.lower() == app.config["ADMIN_USERNAME"].lower():
                error = "That username is reserved."
            else:
                user_id = database.create_user(username, generate_password_hash(password))
                if user_id is None:
                    error = "That username is already taken."
                else:
                    _start_session(user_id)
                    flash("Account created. Your prediction history is now saved to it.", "success")
                    return redirect(url_for("index"))
            flash(error, "danger")
        return render_template("auth.html", mode="register")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            require_csrf()
            user = database.get_user_by_username(request.form.get("username", "").strip())
            if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
                _start_session(user["id"])
                flash("Signed in.", "success")
                return redirect(url_for("index"))
            flash("Incorrect username or password.", "danger")  # same message for both cases
        return render_template("auth.html", mode="login")

    @app.post("/logout")
    def logout():
        require_csrf()
        session.clear()
        flash("Signed out.", "success")
        return redirect(url_for("index"))

    def _start_session(user_id: int) -> None:
        anon = session.get("anon_id")
        session.clear()                      # new session on login (prevents fixation)
        session["user_id"] = user_id
        if anon:
            database.claim_anonymous_predictions(anon, user_id)

    # ------------------------------------------------------------------ JSON API
    @app.get("/api/symptoms")
    def api_symptoms():
        predictor = get_predictor()
        if predictor is None:
            return jsonify(error="Model not trained yet."), 503
        return jsonify(symptoms=predictor.symptom_options, count=len(predictor.symptoms))

    @app.get("/api/model")
    def api_model():
        predictor = get_predictor()
        if predictor is None:
            return jsonify(error="Model not trained yet."), 503
        return jsonify(predictor.info())

    @app.get("/health")
    def health():
        return jsonify(status="ok", model_loaded=get_predictor() is not None)

    @app.get("/api/openapi.json")
    def openapi():
        return jsonify(OPENAPI_SPEC)

    @app.get("/api/docs")
    def api_docs():
        return render_template("api_docs.html")

    # ------------------------------------------------------------------ errors
    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        if wants_json():
            return jsonify(error=exc.description or exc.name), exc.code
        return render_template("error.html", code=exc.code, name=exc.name, message=exc.description), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception):
        logger.exception("Unhandled error")
        if wants_json():
            return jsonify(error="Internal server error."), 500
        return render_template("error.html", code=500, name="Internal Server Error",
                               message="Something went wrong on our side."), 500

    return app


def _ensure_admin(app: Flask) -> None:
    """If ADMIN_USERNAME + ADMIN_PASSWORD are set, make sure that account exists."""
    username, password = app.config["ADMIN_USERNAME"], app.config["ADMIN_PASSWORD"]
    if not (username and password):
        return
    with app.app_context():
        if database.get_user_by_username(username) is None:
            database.create_user(username, generate_password_hash(password))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
                     debug=_env_flag("FLASK_DEBUG"))

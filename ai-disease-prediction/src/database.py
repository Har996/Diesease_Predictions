"""Small SQLite layer for users and prediction history (standard-library ``sqlite3`` only)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app, g

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER REFERENCES users(id) ON DELETE CASCADE,
    anon_id       TEXT,
    created_at    TEXT NOT NULL,
    symptoms      TEXT NOT NULL,   -- JSON list
    age           REAL,
    gender        TEXT,
    prediction    TEXT NOT NULL,
    confidence    REAL NOT NULL,
    top_json      TEXT NOT NULL,   -- JSON list of {disease, probability}
    explain_json  TEXT NOT NULL,   -- JSON list of {symptom, label, effect}
    model_version TEXT NOT NULL,
    synthetic     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_pred_anon ON predictions(anon_id);
"""


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


def close_db(_exc=None) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    Path(current_app.config["DATABASE"]).parent.mkdir(parents=True, exist_ok=True)
    get_db().executescript(SCHEMA)
    get_db().commit()


def init_app(app) -> None:
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- users
def create_user(username: str, password_hash: str) -> int | None:
    """Insert a user; returns the new id, or None if the username already exists."""
    db = get_db()
    try:
        cur = db.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                         (username, password_hash, _now()))
        db.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None


def get_user(user_id: int):
    return get_db().execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()


def get_user_by_username(username: str):
    return get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()


# --------------------------------------------------------------------------- predictions
def save_prediction(result: dict, cleaned: dict, user_id: int | None, anon_id: str | None) -> int:
    db = get_db()
    cur = db.execute(
        """INSERT INTO predictions (user_id, anon_id, created_at, symptoms, age, gender, prediction,
                                    confidence, top_json, explain_json, model_version, synthetic)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, anon_id, _now(), json.dumps(result["selected_symptoms"]), cleaned.get("age"),
         cleaned.get("gender"), result["prediction"], result["confidence"],
         json.dumps(result["top_predictions"]), json.dumps(result["explanation"]),
         result["model_version"], int(result["trained_on_synthetic_data"])))
    db.commit()
    return cur.lastrowid


def _owner_clause(user_id, anon_id) -> tuple[str, tuple]:
    if user_id is not None:
        return "user_id = ?", (user_id,)
    return "user_id IS NULL AND anon_id = ?", (anon_id,)


def get_prediction(pred_id: int, user_id, anon_id):
    clause, params = _owner_clause(user_id, anon_id)
    return get_db().execute(f"SELECT * FROM predictions WHERE id = ? AND {clause}", (pred_id, *params)).fetchone()


def list_predictions(user_id, anon_id, limit: int = 100):
    clause, params = _owner_clause(user_id, anon_id)
    return get_db().execute(
        f"SELECT * FROM predictions WHERE {clause} ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()


def delete_predictions(user_id, anon_id) -> int:
    clause, params = _owner_clause(user_id, anon_id)
    db = get_db()
    cur = db.execute(f"DELETE FROM predictions WHERE {clause}", params)
    db.commit()
    return cur.rowcount


def claim_anonymous_predictions(anon_id: str, user_id: int) -> None:
    """After login, attach this browser's anonymous history to the account."""
    db = get_db()
    db.execute("UPDATE predictions SET user_id = ? WHERE user_id IS NULL AND anon_id = ?", (user_id, anon_id))
    db.commit()


def row_to_record(row) -> dict:
    """Convert a predictions row into a template-friendly dict."""
    return {
        "id": row["id"], "created_at": row["created_at"], "prediction": row["prediction"],
        "confidence": row["confidence"], "age": row["age"], "gender": row["gender"],
        "symptoms": json.loads(row["symptoms"]), "top_predictions": json.loads(row["top_json"]),
        "explanation": json.loads(row["explain_json"]), "model_version": row["model_version"],
        "synthetic": bool(row["synthetic"]),
    }

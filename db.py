"""
db.py

All database interaction: connection, schema creation, seeding, and queries.
Imported by main.py, trainer.py, and dashboard.py.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH    = "library.db"
BOOKS_JSON = "books.json"

REQUIRED_BOOK_FIELDS = {
    "id", "title", "course", "semester",
    "difficulty", "duration_suitability", "past_usage_score", "label",
}


# -------------------------
# Connection
# -------------------------
def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a sqlite3 connection (creates file if it does not exist)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# -------------------------
# Schema
# -------------------------
def _create_tables(cur: sqlite3.Cursor) -> None:
    """Create books and models tables if they do not already exist."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            id                   INTEGER PRIMARY KEY,
            title                TEXT,
            course               TEXT,
            semester             INTEGER,
            difficulty           TEXT,
            duration_suitability TEXT,
            past_usage_score     INTEGER,
            label                TEXT,
            last_updated         TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path       TEXT,
            created_at      TEXT,
            model_name      TEXT,
            feature_columns TEXT,
            feature_version TEXT,
            metrics_json    TEXT,
            is_best         INTEGER DEFAULT 0
        )
        """
    )


# -------------------------
# Seeding
# -------------------------
def init_db(db_path: str = DB_PATH, seed_json: str = BOOKS_JSON) -> None:
    """
    Create tables and load seed data from books.json.
    Re-runnable: skips books already present (by id).
    Validates required fields and logs a data summary.
    """
    conn = get_db_connection(db_path)
    cur  = conn.cursor()
    _create_tables(cur)
    conn.commit()

    if not os.path.exists(seed_json):
        logger.warning("Seed JSON not found at %s — skipping data load.", seed_json)
        conn.close()
        return

    with open(seed_json, "r", encoding="utf-8") as f:
        books = json.load(f)

    # Validate
    valid_books, skipped = [], 0
    for book in books:
        missing = REQUIRED_BOOK_FIELDS - set(book.keys())
        if missing:
            logger.warning("Book id=%s missing fields %s — skipped.", book.get("id", "?"), missing)
            skipped += 1
        else:
            valid_books.append(book)
    if skipped:
        logger.warning("%d book(s) skipped due to validation errors.", skipped)

    # Insert
    now_ts = datetime.now(timezone.utc).isoformat()
    inserted = 0
    for book in valid_books:
        cur.execute("SELECT 1 FROM books WHERE id = ?", (book["id"],))
        if cur.fetchone():
            continue
        cur.execute(
            """
            INSERT INTO books
                (id, title, course, semester, difficulty,
                 duration_suitability, past_usage_score, label, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                book["id"], book["title"], book["course"], book["semester"],
                book["difficulty"], book["duration_suitability"],
                book["past_usage_score"], book["label"], now_ts,
            ),
        )
        inserted += 1

    conn.commit()
    logger.info("Init DB: inserted %d new books from %s.", inserted, seed_json)
    _log_data_summary(cur)
    conn.close()


def _log_data_summary(cur: sqlite3.Cursor) -> None:
    """Log total books and per-course counts."""
    cur.execute("SELECT COUNT(*) AS total FROM books")
    total = cur.fetchone()["total"]
    logger.info("Total books in DB: %d", total)
    cur.execute("SELECT course, COUNT(*) AS cnt FROM books GROUP BY course ORDER BY cnt DESC")
    for row in cur.fetchall():
        logger.info("  %-22s %d", row["course"], row["cnt"])


# -------------------------
# Queries
# -------------------------
def load_books_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all books as a pandas DataFrame."""
    df = pd.read_sql_query("SELECT * FROM books", conn)
    if df.empty:
        logger.error("No books in DB. Run: python main.py --init-db")
    return df


def load_latest_model(conn: sqlite3.Connection) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Load the best (or most recent) model entry from the DB.
    Returns (model_object, feature_columns, metrics_dict).
    Returns (None, [], {}) if no model is found or the file is missing.
    """
    import joblib

    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM models WHERE is_best = 1 ORDER BY created_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        cur.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
    if not row:
        return None, [], {}

    model_path      = row["file_path"]
    feature_columns = json.loads(row["feature_columns"])
    metrics         = json.loads(row["metrics_json"])

    if not os.path.exists(model_path):
        logger.error("Model file missing at %s", model_path)
        return None, feature_columns, metrics

    model = joblib.load(model_path)
    return model, feature_columns, metrics


# -------------------------
# Statistics helper (for --show-stats CLI)
# -------------------------
def print_stats(db_path: str = DB_PATH) -> None:
    """Print a detailed dataset statistics report to stdout."""
    conn = get_db_connection(db_path)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM books")
    row = cur.fetchone()
    if not row or row["total"] == 0:
        print("No books in DB. Run: python main.py --init-db")
        conn.close()
        return

    print(f"\n{'='*44}")
    print("  DATASET STATISTICS")
    print(f"{'='*44}")
    print(f"Total books      : {row['total']}")

    for label, query in [
        ("Books per course",     "SELECT course     AS name, COUNT(*) AS cnt FROM books GROUP BY course     ORDER BY cnt DESC"),
        ("Books per difficulty", "SELECT difficulty AS name, COUNT(*) AS cnt FROM books GROUP BY difficulty ORDER BY cnt DESC"),
        ("Books per label",      "SELECT label      AS name, COUNT(*) AS cnt FROM books GROUP BY label      ORDER BY cnt DESC"),
    ]:
        print(f"\n{label}:")
        cur.execute(query)
        for r in cur.fetchall():
            print(f"  {r['name']:<26} {r['cnt']}")

    cur.execute(
        "SELECT AVG(past_usage_score) AS avg, "
        "MIN(past_usage_score) AS mn, MAX(past_usage_score) AS mx FROM books"
    )
    r = cur.fetchone()
    print(f"\nPast usage score — avg: {r['avg']:.1f}, min: {r['mn']}, max: {r['mx']}")
    print(f"{'='*44}\n")
    conn.close()

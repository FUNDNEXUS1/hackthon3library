"""
main.py

Core application logic for:
 - loading a seed JSON dataset (books.json),
 - storing/reading records in a local SQLite DB (library.db),
 - training simple ML models (Logistic Regression, Decision Tree, Random Forest),
 - saving the chosen model and metadata,
 - providing a recommend_books(...) function to be used by a dashboard or CLI.

Usage examples:
  python main.py --init-db         # create DB and load books.json into it
  python main.py --train          # train models and persist best model
  python main.py --quick-test     # run a quick end-to-end demo in console
"""

import argparse
import json
import sqlite3
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib  # for model serialization

# -------------------------
# Config / constants
# -------------------------
DB_PATH = "library.db"
MODEL_DIR = "models"
BOOKS_JSON = "books.json"

# Label mappings: textual label to numeric for model training
LABEL_TO_INT = {"Not Recommended": 0, "Recommended": 1, "Highly Recommended": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# -------------------------
# Utility: Database helpers
# -------------------------
def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Returns a sqlite3 connection (creates file if not exists)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH, seed_json: str = BOOKS_JSON) -> None:
    """
    Create database tables (books, models) and load seed data from JSON.
    Re-runnable: will skip inserting duplicates based on id.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # Create books table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY,
            title TEXT,
            course TEXT,
            semester INTEGER,
            difficulty TEXT,
            duration_suitability TEXT,
            past_usage_score INTEGER,
            label TEXT
        )
        """
    )

    # Create models metadata table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT,
            created_at TEXT,
            model_name TEXT,
            feature_columns TEXT,
            metrics_json TEXT
        )
        """
    )

    conn.commit()

    # Load JSON and insert (if not present)
    if os.path.exists(seed_json):
        with open(seed_json, "r", encoding="utf-8") as f:
            books = json.load(f)

        inserted = 0
        for book in books:
            cur.execute("SELECT 1 FROM books WHERE id = ?", (book["id"],))
            if cur.fetchone():
                continue

            cur.execute(
                """
                INSERT INTO books (id, title, course, semester, difficulty, duration_suitability, past_usage_score, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    book["id"],
                    book["title"],
                    book["course"],
                    book["semester"],
                    book["difficulty"],
                    book["duration_suitability"],
                    book["past_usage_score"],
                    book["label"],
                ),
            )
            inserted += 1

        conn.commit()
        print(f"Init DB: inserted {inserted} new books from {seed_json}")
    else:
        print(f"Init DB: seed JSON not found at {seed_json}")

    conn.close()


# -------------------------
# Data utilities
# -------------------------
def load_books_from_db(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all books from DB as a pandas DataFrame."""
    return pd.read_sql_query("SELECT * FROM books", conn)


def preprocess_for_model(
    df: pd.DataFrame, feature_columns: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert the books DataFrame into numeric features for model training/prediction.

    Strategy:
      - semester: numeric
      - difficulty: map Introductory->0, Intermediate->1, Advanced->2
      - duration_suitability: one-hot
      - course: one-hot
      - past_usage_score: numeric
    """
    df_work = df.copy()

    # Map difficulty to numbers
    difficulty_map = {"Introductory": 0, "Intermediate": 1, "Advanced": 2}
    df_work["difficulty_num"] = df_work["difficulty"].map(difficulty_map).fillna(0).astype(int)

    # Ensure numeric types
    df_work["semester"] = pd.to_numeric(df_work["semester"], errors="coerce").fillna(1).astype(int)
    df_work["past_usage_score"] = pd.to_numeric(
        df_work["past_usage_score"], errors="coerce"
    ).fillna(0).astype(int)

    # One-hot encode categorical fields
    course_dummies = pd.get_dummies(df_work["course"], prefix="course")
    duration_dummies = pd.get_dummies(df_work["duration_suitability"], prefix="dur")

    x_features = pd.concat(
        [
            df_work[["semester", "difficulty_num", "past_usage_score"]],
            course_dummies,
            duration_dummies,
        ],
        axis=1,
    )

    # If feature_columns provided (during prediction), align columns
    if feature_columns is not None:
        for col in feature_columns:
            if col not in x_features.columns:
                x_features[col] = 0
        x_features = x_features[feature_columns].copy()
        feature_columns = list(x_features.columns)
    else:
        feature_columns = list(x_features.columns)

    return x_features, feature_columns


# -------------------------
# Model training & selection
# -------------------------
def train_and_select_model(df: pd.DataFrame, out_dir: str = MODEL_DIR) -> Dict[str, Any]:
    """
    Train three simple models, evaluate them, select the best by macro F1,
    and persist model + metadata.
    """
    os.makedirs(out_dir, exist_ok=True)

    df_work = df.copy()
    df_work["label_num"] = df_work["label"].map(LABEL_TO_INT)
    df_work = df_work.dropna(subset=["label_num"])

    x_features, feature_cols = preprocess_for_model(df_work)
    y_labels = df_work["label_num"]

    x_train, x_test, y_train, y_test = train_test_split(
        x_features, y_labels, test_size=0.25, random_state=42, stratify=y_labels
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results: Dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        results[name] = {
            "model": model,
            "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1},
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

        print(f"[train] {name} — f1_macro: {f1:.4f}, acc: {acc:.4f}")

    # Select best model by macro F1
    best_name = max(results.keys(), key=lambda n: results[n]["metrics"]["f1_macro"])
    best_entry = results[best_name]
    best_model = best_entry["model"]
    best_metrics = best_entry["metrics"]

    # Timezone-aware UTC timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_filename = f"{best_name}_{timestamp}.joblib"
    model_path = os.path.join(out_dir, model_filename)

    joblib.dump(best_model, model_path)

    metadata = {
        "file_path": model_path,
        "created_at": timestamp,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "metrics": best_metrics,
    }

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO models (file_path, created_at, model_name, feature_columns, metrics_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            metadata["file_path"],
            metadata["created_at"],
            metadata["model_name"],
            json.dumps(metadata["feature_columns"]),
            json.dumps(metadata["metrics"]),
        ),
    )
    conn.commit()
    conn.close()

    print(f"[train] Best model: {best_name} saved to {model_path}")
    return metadata


# -------------------------
# Model loading & recommendation
# -------------------------
def load_latest_model(
    conn: sqlite3.Connection = None,
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """Load the latest model entry from DB."""
    if conn is None:
        conn = get_db_connection()

    cur = conn.cursor()
    cur.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None, [], {}

    model_path = row["file_path"]
    feature_columns = json.loads(row["feature_columns"])
    metrics = json.loads(row["metrics_json"])

    if not os.path.exists(model_path):
        print(f"[load_model] model file missing at {model_path}")
        return None, feature_columns, metrics

    model = joblib.load(model_path)
    return model, feature_columns, metrics


def recommend_books(
    course: str,
    semester: int,
    study_duration: str,
    top_n: int = 5,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Recommend books from the DB for given course, semester, and study_duration.
    """
    conn = get_db_connection(db_path)
    model, feature_cols, _ = load_latest_model(conn)
    if model is None:
        raise RuntimeError("No trained model found. Run: python main.py --train")

    books_df = load_books_from_db(conn)

    x_all, _ = preprocess_for_model(books_df, feature_columns=feature_cols)

    # Predict probability for "Highly Recommended" (class 2) if possible
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_all)
        classes = list(model.classes_)
        if 2 in classes:
            idx = classes.index(2)
            prob_high = probs[:, idx]
        else:
            prob_high = probs.max(axis=1)
    else:
        preds = model.predict(x_all)
        # Build a pandas Series explicitly to avoid type checker warnings
        prob_high = pd.Series(preds == 2, index=books_df.index).replace({True: 1.0, False: 0.0})

    books_df = books_df.copy()
    books_df["pred_prob_highly_recommended"] = prob_high

    # Combine probability with popularity score
    weight_prob = 0.6
    weight_pop = 0.4
    books_df["final_score"] = (
        weight_prob * books_df["pred_prob_highly_recommended"]
        + weight_pop * (books_df["past_usage_score"] / 100.0)
    )

    # Filter by course and semester proximity
    sem_lower = max(1, int(semester) - 1)
    sem_upper = int(semester) + 1

    filtered = books_df[
        (books_df["course"].str.lower() == course.lower())
        & (books_df["semester"] >= sem_lower)
        & (books_df["semester"] <= sem_upper)
    ].copy()

    # Duration match bonus (build a pandas Series explicitly to keep type checkers happy)
    duration_matches = filtered["duration_suitability"].str.lower() == study_duration.lower()
    filtered["duration_match"] = pd.Series(duration_matches, index=filtered.index).apply(
        lambda v: 1 if v else 0
    )

    filtered["final_score_adj"] = filtered["final_score"] + 0.05 * filtered["duration_match"]

    top = filtered.sort_values(
        by=["final_score_adj", "pred_prob_highly_recommended"], ascending=False
    ).head(top_n)

    output: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        output.append(
            {
                "id": int(row["id"]),
                "title": row["title"],
                "course": row["course"],
                "semester": int(row["semester"]),
                "difficulty": row["difficulty"],
                "duration_suitability": row["duration_suitability"],
                "past_usage_score": int(row["past_usage_score"]),
                "pred_prob_highly_recommended": float(row["pred_prob_highly_recommended"]),
                "final_score": float(row["final_score_adj"]),
                "label": row["label"],
            }
        )

    conn.close()
    return output


# -------------------------
# Small CLI for local testing
# -------------------------
def quick_demo():
    print("Quick demo: initializing DB (books.json) ...")
    init_db()

    conn = get_db_connection()
    df = load_books_from_db(conn)
    conn.close()

    print(f"Loaded {len(df)} books from DB.")
    print("Training models...")
    meta = train_and_select_model(df)
    print("Model metrics:", meta["metrics"])

    print("Sample recommendation for Computer Science, semester 2, Short-term:")
    recs = recommend_books(course="Computer Science", semester=2, study_duration="Short-term", top_n=5)
    for i, rec in enumerate(recs, start=1):
        print(f"{i}. {rec['title']} — score {rec['final_score']:.3f}")


def main_cli():
    parser = argparse.ArgumentParser(description="Library Book Recommender - main.py")
    parser.add_argument("--init-db", action="store_true", help="Initialize DB and load books.json")
    parser.add_argument("--train", action="store_true", help="Train models and persist best model")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick end-to-end demo")
    args = parser.parse_args()

    if args.init_db:
        init_db()
    elif args.train:
        conn = get_db_connection()
        df = load_books_from_db(conn)
        conn.close()
        if df.empty:
            print("No books in DB. Run --init-db first.")
            return
        train_and_select_model(df)
    elif args.quick_test:
        quick_demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()

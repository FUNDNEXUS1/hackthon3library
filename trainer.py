"""
trainer.py

Training orchestrator — separate from main.py to keep concerns clean.

Responsibilities:
  - Load data from DB
  - Preprocess features
  - Loop over all model modules in models/
  - Evaluate each model
  - Select the best by macro F1
  - Persist all models and mark the best in DB

Usage:
  python trainer.py           # train all models, auto-select best
  python trainer.py --list    # list all previously trained models
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from db import get_db_connection, load_books_from_db
from preprocessor import preprocess_for_model, log_features, FEATURE_VERSION
from models import CANDIDATE_MODULES

# -------------------------
# Config
# -------------------------
SAVED_MODELS_DIR = "saved_models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LABEL_TO_INT = {"Not Recommended": 0, "Recommended": 1, "Highly Recommended": 2}


# -------------------------
# Core training function
# -------------------------
def train_all_models(db_path: str = "library.db") -> Dict[str, Any]:
    """
    Train all models in models/CANDIDATE_MODULES, evaluate each,
    save all to disk and DB, mark the best one.

    Returns metadata dict for the best model.
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    conn = get_db_connection(db_path)
    df = load_books_from_db(conn)
    conn.close()

    if df.empty:
        logger.error("No books in DB. Run: python main.py --init-db")
        sys.exit(1)

    # Prepare labels
    df = df.copy()
    df["label_num"] = df["label"].map(LABEL_TO_INT)
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    x_all, feature_cols = preprocess_for_model(df)
    y_all = df["label_num"]

    log_features(feature_cols)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.25, random_state=42, stratify=y_all
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results: Dict[str, Dict[str, Any]] = {}

    conn = get_db_connection(db_path)
    cur = conn.cursor()

    for module in CANDIDATE_MODULES:
        name = module.MODEL_NAME
        logger.info("Training %s ...", name)

        model = module.train(x_train, y_train)
        metrics = module.evaluate(model, x_train, x_test, y_train, y_test)

        f1 = metrics["f1_macro"]
        acc = metrics["accuracy"]
        cv_mean = metrics.get("cv_f1_macro_mean")

        log_line = f"[{name}] f1_macro={f1:.4f}  acc={acc:.4f}"
        if cv_mean is not None:
            log_line += f"  cv_f1={cv_mean:.4f}"
        logger.info(log_line)

        # Persist model file
        filename = f"{name}_{timestamp}.joblib"
        filepath = os.path.join(SAVED_MODELS_DIR, filename)
        joblib.dump(model, filepath)

        # Strip classification_report from DB storage (verbose, not needed there)
        db_metrics = {k: v for k, v in metrics.items() if k != "classification_report"}

        cur.execute(
            """
            INSERT INTO models
                (file_path, created_at, model_name, feature_columns,
                 feature_version, metrics_json, is_best)
            VALUES (?, ?, ?, ?, ?, ?, 0)
            """,
            (
                filepath,
                timestamp,
                name,
                json.dumps(feature_cols),
                FEATURE_VERSION,
                json.dumps(db_metrics),
            ),
        )

        results[name] = {
            "model":    model,
            "filepath": filepath,
            "metrics":  metrics,
        }

    conn.commit()

    # Select best model by macro F1
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1_macro"])
    best = results[best_name]
    best_filepath = best["filepath"]
    best_metrics  = best["metrics"]

    cur.execute(
        "UPDATE models SET is_best = 1 WHERE file_path = ?",
        (best_filepath,),
    )
    conn.commit()
    conn.close()

    logger.info("Best model: %s  (f1_macro=%.4f)", best_name, best_metrics["f1_macro"])

    return {
        "model_name":      best_name,
        "file_path":       best_filepath,
        "created_at":      timestamp,
        "feature_columns": feature_cols,
        "feature_version": FEATURE_VERSION,
        "metrics":         best_metrics,
    }


# -------------------------
# list all trained models
# -------------------------
def list_models(db_path: str = "library.db") -> None:
    """Print a comparison table of all models stored in the DB."""
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT model_name, created_at, feature_version, metrics_json, is_best "
        "FROM models ORDER BY created_at DESC"
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No trained models found. Run: python trainer.py")
        return

    print(f"\n{'='*90}")
    print("  TRAINED MODELS")
    print(f"{'='*90}")
    header = (
        f"{'Model':<22} {'Trained At':<18} {'FeatVer':<9} "
        f"{'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'CV F1':>7} {'Best'}"
    )
    print(header)
    print("-" * 90)
    for row in rows:
        m = json.loads(row["metrics_json"])
        best_marker = "  ✓" if row["is_best"] else ""
        cv = f"{m['cv_f1_macro_mean']:.3f}" if "cv_f1_macro_mean" in m else "  -  "
        print(
            f"{row['model_name']:<22} {row['created_at']:<18} "
            f"{(row['feature_version'] or '-'):<9} "
            f"{m.get('accuracy',0):>6.3f} {m.get('precision',0):>6.3f} "
            f"{m.get('recall',0):>6.3f} {m.get('f1_macro',0):>6.3f} "
            f"{cv:>7}{best_marker}"
        )
    print(f"{'='*90}\n")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train all book recommendation models.")
    parser.add_argument("--list", action="store_true", help="List all trained models")
    parser.add_argument("--db", default="library.db", help="Path to SQLite DB")
    args = parser.parse_args()

    if args.list:
        list_models(args.db)
    else:
        train_all_models(args.db)


if __name__ == "__main__":
    main()

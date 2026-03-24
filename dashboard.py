"""
main.py

Entry point and recommendation engine.
Training is now handled by trainer.py; DB logic by db.py; features by preprocessor.py.

Usage:
  python main.py --init-db        # create library.db and load books.json
  python main.py --train          # train all models and select the best
  python main.py --quick-test     # end-to-end console demo
  python main.py --show-stats     # print dataset statistics
  python main.py --list-models    # show all trained models and metrics
"""

import argparse
import logging
from typing import Any, Dict, List

import pandas as pd

from db import get_db_connection, load_books_from_db, load_latest_model, init_db, print_stats
from preprocessor import preprocess_for_model, get_usage_bucket, FEATURE_VERSION
from trainer import train_all_models, list_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DB_PATH = "library.db"


def _get_confidence_label(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    else:
        return "Low"


def _build_explanation(row: pd.Series, input_semester: int, study_duration: str) -> str:
    reasons = []
    bucket = get_usage_bucket(row["past_usage_score"])
    if bucket == "High":
        reasons.append("high popularity")
    elif bucket == "Medium":
        reasons.append("moderate popularity")

    sem_diff = abs(int(row["semester"]) - input_semester)
    if sem_diff == 0:
        reasons.append("matches your semester exactly")
    elif sem_diff == 1:
        reasons.append("close to your semester")

    if str(row["duration_suitability"]).lower() == study_duration.lower():
        reasons.append(f"matches your {study_duration} study plan")

    if row.get("label") == "Highly Recommended":
        reasons.append("historically highly recommended")

    return "Recommended because: " + (", ".join(reasons) if reasons else "good overall fit")


def recommend_books(
    course: str,
    semester: int,
    study_duration: str,
    top_n: int = 5,
    only_highly_recommended: bool = False,
    sort_by: str = "score",
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    model, feature_cols, _ = load_latest_model(conn)

    if model is None:
        raise RuntimeError("No trained model found. Run: python main.py --train")

    books_df = load_books_from_db(conn)
    conn.close()

    if books_df.empty:
        return []

    x_all, _ = preprocess_for_model(books_df, input_semester=semester, feature_columns=feature_cols)

    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(x_all)
        classes = list(model.classes_)
        idx     = classes.index(2) if 2 in classes else None
        prob_high = probs[:, idx] if idx is not None else probs.max(axis=1)
    else:
        preds     = model.predict(x_all)
        prob_high = pd.Series(preds == 2, index=books_df.index).replace({True: 1.0, False: 0.0})

    books_df = books_df.copy()
    books_df["pred_prob_highly_recommended"] = prob_high
    books_df["final_score"] = (
        0.6 * books_df["pred_prob_highly_recommended"]
        + 0.4 * (books_df["past_usage_score"] / 100.0)
    )

    sem_lower = max(1, int(semester) - 1)
    sem_upper = int(semester) + 1
    filtered = books_df[
        (books_df["course"].str.lower() == course.lower())
        & (books_df["semester"] >= sem_lower)
        & (books_df["semester"] <= sem_upper)
    ].copy()

    dur_match = filtered["duration_suitability"].str.lower() == study_duration.lower()
    filtered["duration_match"] = pd.Series(dur_match, index=filtered.index).astype(int)
    filtered["final_score_adj"] = filtered["final_score"] + 0.05 * filtered["duration_match"]

    if only_highly_recommended:
        filtered = filtered[filtered["label"] == "Highly Recommended"]

    sort_map = {
        "score":      ("final_score_adj", False),
        "popularity": ("past_usage_score", False),
        "difficulty": ("difficulty",       True),
    }
    sort_col, sort_asc = sort_map.get(sort_by, ("final_score_adj", False))
    top = filtered.sort_values(by=[sort_col], ascending=sort_asc).head(top_n)

    output: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        prob = float(row["pred_prob_highly_recommended"])
        output.append({
            "id":                           int(row["id"]),
            "title":                        row["title"],
            "course":                       row["course"],
            "semester":                     int(row["semester"]),
            "difficulty":                   row["difficulty"],
            "duration_suitability":         row["duration_suitability"],
            "past_usage_score":             int(row["past_usage_score"]),
            "usage_bucket":                 get_usage_bucket(row["past_usage_score"]),
            "pred_prob_highly_recommended": prob,
            "confidence":                   _get_confidence_label(prob),
            "final_score":                  float(row["final_score_adj"]),
            "label":                        row["label"],
            "explanation":                  _build_explanation(row, int(semester), study_duration),
        })

    return output


def quick_demo() -> None:
    print("Quick demo: initializing DB ...")
    init_db()
    train_all_models()
    print("\nSample — Computer Science, semester 2, Short-term:")
    recs = recommend_books("Computer Science", semester=2, study_duration="Short-term", top_n=5)
    for i, rec in enumerate(recs, start=1):
        print(f"  {i}. {rec['title']}  score={rec['final_score']:.3f}  [{rec['confidence']} confidence]")
        print(f"     {rec['explanation']}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Library Book Recommender")
    parser.add_argument("--init-db",     action="store_true", help="Initialize DB from books.json")
    parser.add_argument("--train",       action="store_true", help="Train all models, save best")
    parser.add_argument("--quick-test",  action="store_true", help="End-to-end console demo")
    parser.add_argument("--show-stats",  action="store_true", help="Print dataset statistics")
    parser.add_argument("--list-models", action="store_true", help="List all trained models")
    args = parser.parse_args()

    if args.init_db:
        init_db()
    elif args.train:
        train_all_models()
    elif args.quick_test:
        quick_demo()
    elif args.show_stats:
        print_stats()
    elif args.list_models:
        list_models()
    else:
        parser.print_help()


if __name__ == "__main__":
    main_cli()

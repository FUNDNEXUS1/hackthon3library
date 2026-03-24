"""
preprocessor.py

All feature engineering lives here.
Imported by trainer.py and main.py so both training and prediction
use exactly the same transformation logic.

[Feature version: v2]
Changes vs v1:
  - past_usage_score bucketed into Low/Medium/High (one-hot)
  - is_same_semester derived feature
  - feature_version tracked
"""

import logging
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Bump this string whenever the feature set changes
FEATURE_VERSION = "v2"
FEATURE_LOG_FILE = "feature_log.txt"


# -------------------------
# Usage score bucketing
# -------------------------
def get_usage_bucket(score: float) -> str:
    """Classify a past_usage_score into Low / Medium / High."""
    if score <= 59:
        return "Low"
    elif score <= 74:
        return "Medium"
    else:
        return "High"


# -------------------------
# Main preprocessing function
# -------------------------
def preprocess_for_model(
    df: pd.DataFrame,
    input_semester: int = None,
    feature_columns: List[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert the books DataFrame into numeric features for training / prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Books dataframe from the DB.
    input_semester : int, optional
        The user's target semester — enables the is_same_semester feature.
        Pass None during training (feature defaults to 0).
    feature_columns : list of str, optional
        If provided, align output columns to this list (used at prediction time
        to ensure the same column set the model was trained on).

    Returns
    -------
    x_features : pd.DataFrame
    feature_columns : list of str
    """
    df_work = df.copy()

    # --- Numeric: semester, past_usage_score ---
    df_work["semester"] = (
        pd.to_numeric(df_work["semester"], errors="coerce").fillna(1).astype(int)
    )
    df_work["past_usage_score"] = (
        pd.to_numeric(df_work["past_usage_score"], errors="coerce").fillna(0).astype(int)
    )

    # --- Ordinal: difficulty ---
    difficulty_map = {"Introductory": 0, "Intermediate": 1, "Advanced": 2}
    df_work["difficulty_num"] = (
        df_work["difficulty"].map(difficulty_map).fillna(0).astype(int)
    )

    # --- Derived: is_same_semester ---
    if input_semester is not None:
        df_work["is_same_semester"] = (
            df_work["semester"] == int(input_semester)
        ).astype(int)
    else:
        df_work["is_same_semester"] = 0

    # --- Bucketed usage score (one-hot) ---
    df_work["usage_bucket"] = df_work["past_usage_score"].apply(get_usage_bucket)
    usage_dummies = pd.get_dummies(df_work["usage_bucket"], prefix="usage")

    # --- One-hot: course, duration_suitability ---
    course_dummies   = pd.get_dummies(df_work["course"],               prefix="course")
    duration_dummies = pd.get_dummies(df_work["duration_suitability"], prefix="dur")

    # --- Assemble feature matrix ---
    x_features = pd.concat(
        [
            df_work[["semester", "difficulty_num", "past_usage_score", "is_same_semester"]],
            usage_dummies,
            course_dummies,
            duration_dummies,
        ],
        axis=1,
    )

    # --- Column alignment (prediction mode) ---
    if feature_columns is not None:
        for col in feature_columns:
            if col not in x_features.columns:
                x_features[col] = 0
        x_features = x_features[feature_columns].copy()
    else:
        feature_columns = list(x_features.columns)

    return x_features, list(feature_columns)


# -------------------------
# Feature logging
# -------------------------
def log_features(feature_columns: List[str]) -> None:
    """Append the current feature list to feature_log.txt."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with open(FEATURE_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{timestamp}] feature_version={FEATURE_VERSION}\n")
        for col in feature_columns:
            f.write(f"  {col}\n")
    logger.info("Feature list written to %s", FEATURE_LOG_FILE)

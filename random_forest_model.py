"""
models/random_forest_model.py

Random Forest classifier for book recommendation.

Why Random Forest for this project:
  - Handles mixed numeric + one-hot features without scaling
  - Robust to small datasets (avoids overfitting via bagging)
  - Naturally outputs class probabilities (predict_proba)
  - Handles class imbalance reasonably with class_weight
  - Easy to interpret via feature_importances_
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


MODEL_NAME = "RandomForest"


def build_model() -> RandomForestClassifier:
    """Return a configured (untrained) Random Forest model."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",   # handles label imbalance in small dataset
        random_state=42,
    )


def train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Fit the model on training data and return it."""
    model = build_model()
    model.fit(x_train, y_train)
    return model


def evaluate(
    model: RandomForestClassifier,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate the model on the test set and run 3-fold cross-validation.
    Returns a metrics dict compatible with the trainer and DB storage.
    """
    y_pred = model.predict(x_test)

    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec   = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1    = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 3-fold cross-validation on full training data
    cv_scores = cross_val_score(
        build_model(), x_train, y_train, cv=3, scoring="f1_macro"
    )
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())

    # Feature importances (top 10)
    fi = sorted(
        zip(x_train.columns, model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    )[:10]
    top_features = {col: round(float(imp), 4) for col, imp in fi}

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1_macro":         f1,
        "cv_f1_macro_mean": cv_mean,
        "cv_f1_macro_std":  cv_std,
        "top_features":     top_features,
        "classification_report": report,
    }

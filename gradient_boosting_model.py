"""
models/gradient_boosting_model.py

Gradient Boosting classifier for book recommendation.

Why Gradient Boosting for this project:
  - Best-in-class accuracy on structured/tabular data
  - Builds trees sequentially, correcting previous errors — ideal for the
    nuanced scoring differences between "Recommended" and "Highly Recommended"
  - Handles mixed feature types without preprocessing beyond encoding
  - Works well even on small datasets (no need for large training sets)
  - Outputs calibrated probabilities via predict_proba
"""

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import pandas as pd
from typing import Dict, Any


MODEL_NAME = "GradientBoosting"


def build_model() -> GradientBoostingClassifier:
    """Return a configured (untrained) Gradient Boosting model."""
    return GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,        # stochastic GB — reduces overfitting on small data
        random_state=42,
    )


def train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> GradientBoostingClassifier:
    """Fit the model on training data and return it."""
    model = build_model()
    model.fit(x_train, y_train)
    return model


def evaluate(
    model: GradientBoostingClassifier,
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

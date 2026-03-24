"""
models/knn_model.py

K-Nearest Neighbors classifier for book recommendation.

Why KNN for this project:
  - Recommendation systems are naturally similarity-based — KNN directly
    captures "books similar to what worked for students like you"
  - Requires no training phase; fast to update when new books are added
  - Non-parametric: makes no assumptions about the data distribution
  - Works well on small datasets where deep models would overfit
  - Intuitive to explain: "This book was recommended because students with
    similar course/semester profiles found it useful"

Note: Features must be scaled before KNN — handled in preprocess_for_model
via MinMaxScaler applied here so the caller doesn't have to worry about it.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
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


MODEL_NAME = "KNN"


def build_model() -> Pipeline:
    """
    Return a Pipeline of MinMaxScaler + KNeighborsClassifier.
    Scaling is embedded in the pipeline so prediction is one-call safe.
    """
    return Pipeline([
        ("scaler", MinMaxScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean",
            weights="distance",   # closer neighbors count more
        )),
    ])


def train(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the pipeline on training data and return it."""
    model = build_model()
    model.fit(x_train, y_train)
    return model


def evaluate(
    model: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate the model on the test set and run 3-fold cross-validation.
    Returns a metrics dict compatible with the trainer and DB storage.
    KNN has no feature_importances_, so top_features is omitted.
    """
    y_pred = model.predict(x_test)

    acc   = float(accuracy_score(y_test, y_pred))
    prec  = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec   = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1    = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True)

    # 3-fold cross-validation
    cv_scores = cross_val_score(
        build_model(), x_train, y_train, cv=3, scoring="f1_macro"
    )
    cv_mean = float(cv_scores.mean())
    cv_std  = float(cv_scores.std())

    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1_macro":         f1,
        "cv_f1_macro_mean": cv_mean,
        "cv_f1_macro_std":  cv_std,
        "classification_report": report,
    }

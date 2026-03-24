"""
models package

Each module exposes:
  MODEL_NAME  — string identifier stored in the DB
  build_model()  — returns an untrained model/pipeline
  train(x_train, y_train)  — fits and returns the model
  evaluate(model, x_train, x_test, y_train, y_test)  — returns metrics dict

Adding a new model:
  1. Create models/my_model.py following the same interface above.
  2. Import it in trainer.py's CANDIDATE_MODULES list.
  That's it — the trainer loop picks it up automatically.
"""

from models import random_forest_model, gradient_boosting_model, knn_model

CANDIDATE_MODULES = [
    random_forest_model,
    gradient_boosting_model,
    knn_model,
]

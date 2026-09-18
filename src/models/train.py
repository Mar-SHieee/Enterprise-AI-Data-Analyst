"""Training utilities for classical ML models.

Models are saved with joblib and a JSON metadata file so the deployment
layer can identify the artifact and the feature schema.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.evaluation.metrics import classification_metrics


DEFAULT_FEATURES = [
    "first_order_revenue",
    "first_order_quantity",
    "first_order_avg_price",
    "first_order_unique_products",
    "first_order_lines",
    "log1p_first_order_revenue",
    "log1p_first_order_quantity",
    "log1p_first_order_avg_price",
    "log1p_first_order_lines",
]


def build_models(random_state: int = 42):
    """Return baseline pipelines for comparison."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", numeric, DEFAULT_FEATURES)], remainder="drop")

    logistic = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        )),
    ])
    forest = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=-1
        )),
    ])
    return {"logistic_regression": logistic, "random_forest": forest}


def tune_random_forest(X, y, random_state: int = 42):
    """Tune RF with stratified CV; returns fitted GridSearchCV."""
    model = build_models(random_state)["random_forest"]
    grid = {
        "model__max_depth": [3, 5, None],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", 0.8],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        model, grid, scoring="average_precision", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X, y)
    return search


def evaluate_fitted(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return classification_metrics(y_test, pred, proba)


def save_artifact(model, feature_names, metrics, output_dir="models", name="classical_repeat_purchase"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / f"{name}.joblib"
    meta_path = output / f"{name}.json"
    joblib.dump(model, model_path)
    metadata = {
        "model_name": name,
        "task": "repeat_purchase_prediction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(feature_names),
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf8")
    return model_path, meta_path

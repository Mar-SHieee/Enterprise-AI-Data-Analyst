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


# ============================================================
# Member 4 — Deep Learning functions
# Model: MLP for repeat-purchase prediction.
# Trained on the EXACT SAME feature table / target / train-test split as the
# classical baselines above (see build_repeat_purchase_dataset + DEFAULT_FEATURES),
# so the ML-vs-DL comparison is apples-to-apples.
# ============================================================

try:
    import torch
    import torch.nn as nn
except ImportError:  # torch is only required for the DL functions below
    torch = None
    nn = None


class ChurnMLP(nn.Module if nn is not None else object):
    """Simple MLP for binary classification on tabular features."""

    def __init__(self, n_features: int, hidden=(32, 16), dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    hidden=(32, 16),
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 300,
    patience: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
):
    """Train ChurnMLP with class-weighted BCE loss, early stopping and
    ReduceLROnPlateau learning-rate scheduling.

    Returns (model, history) where history has 'train_loss' / 'val_loss'
    per epoch (required training-curve deliverable).
    """
    if torch is None:
        raise ImportError("torch is required for train_mlp_classifier(); pip install torch")
    torch.manual_seed(seed)

    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xval = torch.tensor(X_val, dtype=torch.float32, device=device)
    yval = torch.tensor(y_val, dtype=torch.float32, device=device)

    model = ChurnMLP(X_train.shape[1], hidden=hidden, dropout=dropout).to(device)

    n_pos = max((ytr == 1).sum().item(), 1)
    n_neg = max((ytr == 0).sum().item(), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    n = Xtr.shape[0]
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xval), yval).item()

        scheduler.step(val_loss)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def predict_proba_mlp(model: "ChurnMLP", X, device: str = "cpu"):
    if torch is None:
        raise ImportError("torch is required for predict_proba_mlp(); pip install torch")
    with torch.no_grad():
        model.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return torch.sigmoid(model(Xt)).cpu().numpy()


def save_dl_artifact(
    model: "ChurnMLP",
    feature_names,
    scaler_mean,
    scaler_scale,
    metrics: dict,
    output_dir="models",
    name="dl_mlp_repeat_purchase_v1",
):
    """Save the DL artifact using the SAME naming/metadata convention as
    Member 3's `save_artifact()` above, so Member 6 can load either model
    type the same way."""
    import json as _json
    from pathlib import Path as _Path
    from datetime import datetime, timezone

    out = _Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{name}.pt")

    metadata = {
        "model_name": name,
        "task": "repeat_purchase_prediction",
        "model_type": "ChurnMLP",
        "hidden": list(model.hidden),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(feature_names),
        "scaler_mean": list(map(float, scaler_mean)),
        "scaler_scale": list(map(float, scaler_scale)),
        "metrics": metrics,
    }
    with open(out / f"{name}.json", "w", encoding="utf8") as f:
        _json.dump(metadata, f, indent=2)

    return out / f"{name}.pt", out / f"{name}.json"


def load_dl_artifact(output_dir="models", name="dl_mlp_repeat_purchase_v1", device="cpu"):
    """Load a saved ChurnMLP artifact back for inference (used by predict.py)."""
    import json as _json
    from pathlib import Path as _Path

    in_dir = _Path(output_dir)
    with open(in_dir / f"{name}.json") as f:
        meta = _json.load(f)

    model = ChurnMLP(n_features=len(meta["features"]), hidden=tuple(meta["hidden"]))
    model.load_state_dict(torch.load(in_dir / f"{name}.pt", map_location=device))
    model.eval()
    return model, meta

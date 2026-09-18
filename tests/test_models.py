"""
Tests for the modelling and serving layer (Members 3, 4 and 6).

These cover the things that actually broke in this project: feature leakage,
the serving contract between a saved artifact and the prediction tool, and the
prediction log. They use synthetic data and a temporary SQLite database, so
they need no network, no trained artifact and no real warehouse.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.engineering import build_repeat_purchase_dataset  # noqa: E402
from src.models.train import DEFAULT_FEATURES, build_models  # noqa: E402
from src.utils.logger import PredictionLogger  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def make_transactions() -> pd.DataFrame:
    """Two customers: one repeats, one does not."""
    rows = [
        # customer 1 — first order, then a second order (repeat)
        ("A1", "P1", 2, "2011-01-05 10:00:00", 5.0, 1, "UK"),
        ("A1", "P2", 1, "2011-01-05 10:00:00", 10.0, 1, "UK"),
        ("A2", "P1", 3, "2011-03-05 10:00:00", 5.0, 1, "UK"),
        # customer 2 — one order only
        ("B1", "P3", 4, "2011-02-01 09:00:00", 2.5, 2, "France"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["Invoice", "StockCode", "Quantity", "InvoiceDate",
                 "Price", "Customer ID", "Country"],
    )
    frame["Revenue"] = frame["Quantity"] * frame["Price"]
    return frame


@pytest.fixture
def warehouse(tmp_path) -> Path:
    """A minimal warehouse matching sql/schema.sql."""
    db_path = tmp_path / "test_retail.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, country TEXT);
        CREATE TABLE products (stock_code TEXT PRIMARY KEY, description TEXT);
        CREATE TABLE orders (invoice_no TEXT PRIMARY KEY, customer_id INTEGER,
                             invoice_date DATETIME, is_cancelled INTEGER DEFAULT 0);
        CREATE TABLE order_items (item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                  invoice_no TEXT, stock_code TEXT,
                                  quantity INTEGER, unit_price REAL, revenue REAL);
        INSERT INTO customers VALUES (1, 'UK'), (2, 'France');
        INSERT INTO products VALUES ('P1','a'), ('P2','b'), ('P3','c');
        INSERT INTO orders VALUES
            ('A1', 1, '2011-01-05 10:00:00', 0),
            ('A2', 1, '2011-03-05 10:00:00', 0),
            ('B1', 2, '2011-02-01 09:00:00', 0);
        INSERT INTO order_items (invoice_no, stock_code, quantity, unit_price, revenue)
        VALUES ('A1','P1',2,5.0,10.0), ('A1','P2',1,10.0,10.0),
               ('A2','P1',3,5.0,15.0), ('B1','P3',4,2.5,10.0);
        """
    )
    conn.commit()
    conn.close()
    return db_path


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_repeat_target_is_correct():
    dataset = build_repeat_purchase_dataset(make_transactions())
    by_customer = dataset.set_index("Customer ID")["repeat_customer"].to_dict()
    assert by_customer[1] == 1
    assert by_customer[2] == 0


def test_features_use_first_order_only():
    """Customer 1's second order (revenue 15) must not enter the features."""
    dataset = build_repeat_purchase_dataset(make_transactions())
    first = dataset.set_index("Customer ID").loc[1]
    assert first["first_order_revenue"] == pytest.approx(20.0)  # 10 + 10, not 35
    assert first["first_order_lines"] == 2


def test_no_leaking_columns_in_feature_list():
    """observed_order_count defines the target and must never be a predictor."""
    assert "observed_order_count" not in DEFAULT_FEATURES
    assert "repeat_customer" not in DEFAULT_FEATURES


def test_dataset_has_no_nan_or_inf():
    dataset = build_repeat_purchase_dataset(make_transactions())
    numeric = dataset.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Classical models
# --------------------------------------------------------------------------- #
def test_classical_pipelines_fit_and_score():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, len(DEFAULT_FEATURES))),
                     columns=DEFAULT_FEATURES)
    y = (X[DEFAULT_FEATURES[0]] > 0).astype(int)

    for name, model in build_models().items():
        model.fit(X, y)
        proba = model.predict_proba(X)[:, 1]
        assert len(proba) == len(X), name
        assert ((proba >= 0) & (proba <= 1)).all(), name


# --------------------------------------------------------------------------- #
# Deep-learning track
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch", reason="DL track needs PyTorch")


def test_mlp_trains_and_serves(tmp_path):
    from src.models.dl import save_dl_artifact, train_mlp, tune_threshold

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, len(DEFAULT_FEATURES))),
                     columns=DEFAULT_FEATURES)
    y = (X[DEFAULT_FEATURES[0]] + rng.normal(scale=0.3, size=n) > 0.8).astype(int)

    split = int(n * 0.7)
    model, history = train_mlp(
        X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:],
        max_epochs=15, patience=5, verbose=False,
    )

    proba = model.predict_proba(X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert len(history.train_loss) >= 1

    threshold = tune_threshold(model, X.iloc[split:], y.iloc[split:])
    assert 0.0 < threshold < 1.0

    model_path, meta_path = save_dl_artifact(
        model, {"average_precision": 0.5}, history, output_dir=tmp_path
    )
    assert model_path.exists() and meta_path.exists()


def test_saved_mlp_reloads_identically(tmp_path):
    """The joblib round-trip must not change a single prediction."""
    import joblib

    from src.models.dl import save_dl_artifact, train_mlp

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(80, len(DEFAULT_FEATURES))),
                     columns=DEFAULT_FEATURES)
    y = (X[DEFAULT_FEATURES[0]] > 0).astype(int)
    model, _ = train_mlp(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:],
                         max_epochs=10, patience=5, verbose=False)

    before = model.predict_proba(X)[:, 1]
    path, _ = save_dl_artifact(model, {}, None, output_dir=tmp_path)
    after = joblib.load(path).predict_proba(X)[:, 1]
    assert np.allclose(before, after)


# --------------------------------------------------------------------------- #
# Serving contract
# --------------------------------------------------------------------------- #
def test_prediction_tool_serves_a_real_customer(tmp_path, warehouse):
    """A DL artifact must be loadable by the same tool that serves the
    classical model — that is the whole point of the wrapper."""
    import joblib

    from src.models.dl import save_dl_artifact, train_mlp
    from src.models.predict import PredictionTool

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(80, len(DEFAULT_FEATURES))),
                     columns=DEFAULT_FEATURES)
    y = (X[DEFAULT_FEATURES[0]] > 0).astype(int)
    model, _ = train_mlp(X.iloc[:60], y.iloc[:60], X.iloc[60:], y.iloc[60:],
                         max_epochs=10, patience=5, verbose=False)
    save_dl_artifact(model, {}, None, output_dir=tmp_path, name="mlp_test")

    tool = PredictionTool.from_artifact(tmp_path / "mlp_test", db_path=warehouse)
    result = tool.predict(customer_id="1")

    assert result.status == "success"
    assert 0.0 <= (result.probability or 0.0) <= 1.0
    # features come from the FIRST invoice (A1: revenue 10 + 10), never A2
    assert result.features_used["first_order_revenue"] == pytest.approx(20.0)
    assert result.features_used["first_order_lines"] == 2
    assert joblib is not None


def test_prediction_tool_reports_missing_customer(tmp_path, warehouse):
    import joblib

    from src.models.predict import PredictionTool
    from src.models.train import build_models

    model = build_models()["logistic_regression"]
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(40, len(DEFAULT_FEATURES))),
                     columns=DEFAULT_FEATURES)
    model.fit(X, (X[DEFAULT_FEATURES[0]] > 0).astype(int))
    joblib.dump(model, tmp_path / "m.joblib")
    (tmp_path / "m.json").write_text(
        '{"features": %s, "model_version": "test"}' % list(DEFAULT_FEATURES).__repr__().replace("'", '"')
    )

    tool = PredictionTool.from_artifact(tmp_path / "m", db_path=warehouse)
    assert tool.predict(customer_id="9999").status == "error"
    # no id at all must also fail cleanly, not score something arbitrary
    assert tool.predict().status == "error"


# --------------------------------------------------------------------------- #
# Prediction logging (MLOps)
# --------------------------------------------------------------------------- #
def test_prediction_logger_round_trip(tmp_path):
    logger = PredictionLogger(tmp_path / "predictions.jsonl")
    logger.log_prediction(
        "17841",
        {"model_version": "v1", "prediction": 1, "probability": 0.77,
         "label": "repeat", "status": "success", "features_used": {"a": 1.0},
         "challenger": {"model_version": "dl_v1", "probability": 0.61}},
    )
    logger.log_answer("total revenue?", {"trace_id": "abc", "status": "success",
                                         "tools_used": ["sql"], "latency_ms": 12.0})

    records = logger.read_all()
    assert len(records) == 2
    prediction = records[0]
    assert prediction["model_version"] == "v1"
    assert prediction["challenger_probability"] == 0.61
    assert prediction["schema_version"] == PredictionLogger.SCHEMA_VERSION
    assert records[1]["trace_id"] == "abc"


def test_prediction_logger_satisfies_audit_contract(tmp_path):
    """The same object must be usable as the agent's audit sink."""
    logger = PredictionLogger(tmp_path / "audit.jsonl")
    event = logger.log({"trace_id": "t1", "event": "route", "status": "success"})
    assert event["event"] == "route"
    assert "timestamp" in event

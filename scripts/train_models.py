"""
Train the classical baseline and the deep-learning upgrade, then write the
ML vs DL comparison table.

This is the reproducible, non-notebook path to every model artifact the app
loads. Notebooks 04 and 05 show the analysis; this script guarantees the
artifacts can be rebuilt in one command inside Docker or CI.

Usage (from the repo root):
    python scripts/train_models.py
    python scripts/train_models.py --csv data/sample.csv --skip-dl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.features.engineering import build_repeat_purchase_dataset  # noqa: E402
from src.models.train import (  # noqa: E402
    DEFAULT_FEATURES,
    build_models,
    evaluate_fitted,
    save_artifact,
    tune_random_forest,
)

TARGET = "repeat_customer"


def load_dataset(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    dataset = build_repeat_purchase_dataset(raw)
    print(f"customers: {len(dataset)} | repeat rate: {dataset[TARGET].mean():.3f}")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/sample.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.csv))
    X = dataset[DEFAULT_FEATURES].copy()
    y = dataset[TARGET].copy()

    # One stratified split reused by every model, so the comparison is fair.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.seed
    )
    # The DL model needs its own validation split for early stopping; it is
    # carved out of the training data so the test set stays untouched.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2,
        stratify=y_train_full, random_state=args.seed,
    )
    print(f"train {X_train.shape} | val {X_val.shape} | test {X_test.shape}")

    comparison: list[dict] = []

    # ---- classical baselines --------------------------------------------- #
    for name, model in build_models(args.seed).items():
        model.fit(X_train_full, y_train_full)
        metrics = evaluate_fitted(model, X_test, y_test)
        comparison.append({"model": name, "family": "classical", **metrics})
        print(f"{name:<24} AP={metrics.get('average_precision', float('nan')):.4f}")

    # ---- tuned champion --------------------------------------------------- #
    search = tune_random_forest(X_train_full, y_train_full, args.seed)
    best_rf = search.best_estimator_
    rf_metrics = evaluate_fitted(best_rf, X_test, y_test)
    comparison.append({"model": "random_forest_tuned", "family": "classical", **rf_metrics})
    print(f"{'random_forest_tuned':<24} AP={rf_metrics.get('average_precision', float('nan')):.4f}")
    print("best params:", search.best_params_)

    model_path, meta_path = save_artifact(
        best_rf, DEFAULT_FEATURES, rf_metrics,
        output_dir=args.models_dir, name="classical_repeat_purchase_rf_v1",
    )
    print("saved:", model_path)

    # ---- deep-learning upgrade -------------------------------------------- #
    if not args.skip_dl:
        try:
            from src.models.dl import (
                evaluate_mlp, save_dl_artifact, train_mlp, tune_threshold,
            )

            mlp, history = train_mlp(
                X_train, y_train, X_val, y_val, random_state=args.seed, verbose=True
            )
            threshold = tune_threshold(mlp, X_val, y_val)
            dl_metrics = evaluate_mlp(mlp, X_test, y_test)
            comparison.append({"model": "mlp_v1", "family": "deep_learning", **dl_metrics})
            print(f"{'mlp_v1':<24} AP={dl_metrics.get('average_precision', float('nan')):.4f} "
                  f"(threshold {threshold})")
            dl_model_path, _ = save_dl_artifact(
                mlp, dl_metrics, history, output_dir=args.models_dir
            )
            print("saved:", dl_model_path)
        except ImportError as exc:
            print(f"[train] deep-learning track skipped: {exc}")

    # ---- comparison table -------------------------------------------------- #
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(comparison).sort_values(
        "average_precision", ascending=False, na_position="last"
    )
    frame.to_csv(reports / "model_comparison.csv", index=False)
    (reports / "model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print("\n" + frame.to_string(index=False))
    print(f"\ncomparison written to {reports / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()

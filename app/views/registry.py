"""Model Registry — artifacts, versions, metrics, ML vs DL comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.data.access import read_json
from app.views.base import Page
from src.utils import config


class RegistryPage(Page):
    title = "Model registry"

    def render(self) -> None:
        st.header(self.title)
        models_dir = Path(config.MODELS_DIR)
        reports_dir = Path(config.REPORTS_DIR)

        self._render_comparison(reports_dir)
        st.divider()
        self._render_artifacts(models_dir)
        self._render_training_curves(models_dir)

    def _render_comparison(self, reports_dir: Path) -> None:
        comparison = read_json(reports_dir / "model_comparison.json")
        if not comparison:
            st.info("Run `python scripts/train_models.py` to generate the comparison.")
            return

        st.subheader("Classical ML vs deep learning")
        frame = pd.DataFrame(comparison)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        if "average_precision" in frame:
            st.bar_chart(frame.set_index("model")["average_precision"])
        st.caption(
            "Average precision (PR-AUC) is the headline metric: the target is "
            "imbalanced, so accuracy is not informative."
        )

    def _render_artifacts(self, models_dir: Path) -> None:
        st.subheader("Registered artifacts")
        rows = []
        for meta_file in sorted(models_dir.glob("*.json")):
            meta = read_json(meta_file, {})
            artifact = meta_file.with_suffix(".joblib")
            rows.append(
                {
                    "artifact": meta_file.stem,
                    "family": meta.get("model_family", "classical"),
                    "version": meta.get("model_version", "v1"),
                    "created_utc": meta.get("created_at_utc", "—"),
                    "n_features": len(meta.get("features", [])),
                    "average_precision": (meta.get("metrics") or {}).get(
                        "average_precision"
                    ),
                    "size_kb": round(artifact.stat().st_size / 1024, 1)
                    if artifact.exists()
                    else None,
                }
            )
        if not rows:
            st.info("No model artifacts found in `models/`.")
            return

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Inspect metadata", [r["artifact"] for r in rows])
        st.json(read_json(models_dir / f"{chosen}.json", {}))

    def _render_training_curves(self, models_dir: Path) -> None:
        history = models_dir / "dl_repeat_purchase_mlp_v1_history.csv"
        if not history.exists():
            return
        st.subheader("Deep-learning training curves")
        curves = pd.read_csv(history).set_index("epoch")
        c1, c2 = st.columns(2)
        c1.line_chart(curves[["train_loss", "val_loss"]])
        c2.line_chart(curves[["val_average_precision", "val_roc_auc"]])

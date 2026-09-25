"""Monitoring — audit log, prediction log, latency and block rates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from app.data.access import get_resources, read_json
from app.views.base import Page
from src.utils import config


class MonitoringPage(Page):
    title = "Monitoring"

    def render(self) -> None:
        _, prediction_log = get_resources()
        st.header(self.title)

        self._render_agent_evaluation()
        st.divider()
        self._render_audit_log()
        st.divider()
        self._render_prediction_log(prediction_log)

    def _render_agent_evaluation(self) -> None:
        agent_report = read_json(Path(config.REPORTS_DIR) / "agent_evaluation.json")
        if not agent_report or not agent_report.get("summary"):
            return
        st.subheader("Agent evaluation (offline)")
        summary = agent_report["summary"]
        cols = st.columns(4)
        cols[0].metric(
            "Tool selection", f"{summary.get('tool_selection_accuracy', 0):.0%}"
        )
        cols[1].metric("Groundedness", f"{summary.get('groundedness_rate', 0):.0%}")
        cols[2].metric("Hallucination", f"{summary.get('hallucination_rate', 0):.0%}")
        cols[3].metric(
            "Task completion", f"{summary.get('task_completion_rate', 0):.0%}"
        )

    def _render_audit_log(self) -> None:
        st.subheader("Live audit log")
        audit_path = Path(config.AUDIT_LOG_PATH)
        if not audit_path.exists():
            st.info("No audit events yet — ask the analyst a question first.")
            return

        records = []
        for line in audit_path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        audit = pd.DataFrame(records)
        if audit.empty:
            return

        answers = audit[audit["event"] == "answer"] if "event" in audit else audit
        cols = st.columns(4)
        cols[0].metric("Events", len(audit))
        cols[1].metric("Blocked", int((audit.get("status") == "blocked").sum()))
        cols[2].metric("Errors", int((audit.get("status") == "error").sum()))
        if "latency_ms" in answers and not answers.empty:
            cols[3].metric(
                "Median latency", f"{answers['latency_ms'].median():.0f} ms"
            )
        show = [
            c
            for c in ["timestamp", "event", "status", "tools_used", "reason", "latency_ms"]
            if c in audit.columns
        ]
        st.dataframe(
            audit[show].tail(50).iloc[::-1], use_container_width=True, hide_index=True
        )

    def _render_prediction_log(self, prediction_log) -> None:
        st.subheader("Prediction log")
        predictions = prediction_log.to_dataframe()
        if predictions.empty:
            st.info("No predictions served yet.")
            return

        served = (
            predictions[predictions.get("event") == "prediction"]
            if "event" in predictions
            else predictions
        )
        show = [
            c
            for c in [
                "timestamp",
                "customer_id",
                "model_version",
                "probability",
                "label",
                "challenger_model_version",
                "challenger_probability",
                "status",
            ]
            if c in served.columns
        ]
        st.dataframe(
            served[show].tail(50).iloc[::-1], use_container_width=True, hide_index=True
        )
        if "probability" in served and served["probability"].notna().any():
            st.caption(
                "Distribution of served probabilities — a shift here is the "
                "first sign of input drift."
            )
            buckets = pd.cut(
                served["probability"].dropna(),
                bins=[i / 10 for i in range(11)],
                labels=[f"{i/10:.1f}–{(i+1)/10:.1f}" for i in range(10)],
                include_lowest=True,
            )
            st.bar_chart(buckets.value_counts().sort_index())

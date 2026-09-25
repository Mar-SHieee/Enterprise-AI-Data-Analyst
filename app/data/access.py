"""Resources (built once per session) and small data-access helpers.

Design rule carried over from the agent: **the app never computes a business
number in Python**. Every figure comes from a query executed through the
read-only SQL tool, so what the manager sees and what the agent answers can
never disagree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from rag import build_agent
from src.utils import config
from src.utils.logger import PredictionLogger, get_logger

logger = get_logger("streamlit_app")


@st.cache_resource(show_spinner="Starting the analytics agent…")
def get_resources():
    """Build (once) and cache the agent + the prediction logger."""
    prediction_log = PredictionLogger(
        Path(config.REPORTS_DIR) / "predictions.jsonl"
    )
    agent = build_agent()
    return agent, prediction_log


@st.cache_data(ttl=300, show_spinner=False)
def run_sql(sql: str) -> dict:
    """Execute a query through the guarded tool and cache the result."""
    agent, _ = get_resources()
    if agent.sql_tool is None:
        return {"status": "error", "reason": "No database connected.", "rows": []}
    return agent.sql_tool.execute(sql)


def sql_frame(sql: str) -> pd.DataFrame:
    """Run `sql` and return the rows as a DataFrame, warning on failure."""
    result = run_sql(sql)
    if result.get("status") != "success":
        st.warning(f"Query {result.get('status')}: {result.get('reason')}")
        return pd.DataFrame()
    return pd.DataFrame(result.get("rows", []))


def scalar(sql: str, default="—"):
    """Run `sql` and return its single first cell, or `default` if empty."""
    frame = sql_frame(sql)
    if frame.empty:
        return default
    return frame.iloc[0, 0]


def read_json(path: Path, default=None):
    """Best-effort JSON read that never raises — returns `default` instead."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default

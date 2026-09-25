"""
Enterprise AI Data Analyst — Streamlit application entry point.

Five pages, mapping one-to-one onto the rubric's deployment requirements:

  Executive Dashboard  — deterministic KPIs, straight from SQL.
  Ask the Analyst      — the controlled RAG + SQL + prediction agent.
  Customer Prediction  — champion (classical) vs challenger (deep MLP).
  Model Registry       — artifacts, versions, metrics, ML vs DL comparison.
  Monitoring           — audit log, prediction log, latency and block rates.

All the actual page logic lives under app/views/, the shared SQL and cached
resources under app/data/, and the sidebar under app/ui/. This file only
wires them together.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui.sidebar import render_sidebar  # noqa: E402
from app.views import PAGES  # noqa: E402

st.set_page_config(
    page_title="Enterprise AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    choice = render_sidebar(PAGES)
    PAGES[choice].render()


if __name__ == "__main__":
    main()

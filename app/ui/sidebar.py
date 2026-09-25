"""Sidebar chrome: page picker + system status panel."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.data.access import get_resources
from app.views import Page
from src.utils import config


def render_sidebar(pages: dict[str, Page]) -> str:
    """Draw the sidebar and return the title of the page the user picked."""
    st.sidebar.title("Enterprise AI Data Analyst")
    st.sidebar.caption("Project 10 — SQL + Spark + ML/DL + RAG + agent")
    choice = st.sidebar.radio("Page", list(pages))

    st.sidebar.divider()
    _render_status()

    return choice


def _render_status() -> None:
    agent, _ = get_resources()
    st.sidebar.markdown("**System status**")
    st.sidebar.write(
        {
            "LLM": getattr(agent.llm, "name", "unknown"),
            "tools": agent.available_tools or ["none"],
            "database": Path(config.DB_PATH).name,
        }
    )
    if getattr(agent.llm, "name", "") == "rule":
        st.sidebar.info(
            "Running in offline rule mode: routing and SQL come from "
            "deterministic templates and answers show raw tool evidence. "
            "Set `LLM_PROVIDER` and `LLM_API_KEY` in `.env` for generated prose."
        )
    if agent.sql_tool is None:
        st.sidebar.error("No database. Run `python scripts/build_database.py` first.")

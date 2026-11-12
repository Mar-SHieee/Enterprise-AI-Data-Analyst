"""Top navigation bar — part of the page, not a separate panel."""

from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from app.data.access import get_resources
from app.views import Page
from src.utils import config

LABELS = {
    "AI Analyst": "AI Analyst",
    "Executive dashboard": "Dashboard",
    "Customer prediction": "Predictions",
    "Model registry": "Model Registry",
    "Monitoring": "Monitoring",
}


def _go(name: str) -> None:
    st.session_state["nova_page"] = name


def render_topnav(pages: dict[str, Page]) -> str:
    agent, _ = get_resources()
    online = agent.sql_tool is not None
    tooltip = escape(
        f"Model: {getattr(agent.llm, 'name', 'unknown')} · "
        f"Database: {Path(config.DB_PATH).name} · "
        f"Tools: {len(agent.available_tools or [])}",
        quote=True,
    )

    names = list(pages)
    if st.session_state.get("nova_page") not in names:
        st.session_state["nova_page"] = names[0]

    with st.container(key="topnav"):
        cols = st.columns([3.4] + [1.35] * len(names) + [1.1])
        cols[0].markdown(
            '<div class="app-brand">✦ Enterprise AI Data Analyst</div>',
            unsafe_allow_html=True,
        )
        for col, name in zip(cols[1:-1], names):
            active = st.session_state["nova_page"] == name
            col.button(
                LABELS.get(name, name),
                key=f"nav_{name}",
                type="primary" if active else "secondary",
                width="stretch",
                on_click=_go,
                args=(name,),
            )
        cols[-1].markdown(
            f'<div style="text-align:right"><span class="nova-pill{"" if online else " warn"}" '
            f'title="{tooltip}">● {"Online" if online else "Offline"}</span></div>',
            unsafe_allow_html=True,
        )

    return st.session_state["nova_page"]

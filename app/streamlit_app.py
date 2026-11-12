"""Enterprise AI Data Analyst — Streamlit application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui.theme import inject_theme  # noqa: E402
from app.ui.topnav import render_topnav  # noqa: E402
from app.views import PAGES  # noqa: E402

st.set_page_config(
    page_title="Enterprise AI Data Analyst",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main() -> None:
    inject_theme()
    choice = render_topnav(PAGES)
    PAGES[choice].render()


if __name__ == "__main__":
    main()

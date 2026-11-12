"""Global theme — one light, pink/mauve palette with a top navigation bar."""

from __future__ import annotations

import streamlit as st

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #FBF7FA; --card: #FFFFFF; --border: #EBDCE6;
    --text: #4A2F45; --muted: #8A7586;
    --pink: #C982A8; --deep: #B0688F; --soft: #F6E3EE; --soft-text: #8E4A72;
}

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg); color: var(--text); }

/* chrome: no sidebar, no default toolbar */
#MainMenu, footer, [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 7rem; max-width: 1180px; }

h1, h2, h3, h4 { color: var(--text) !important; font-weight: 700; letter-spacing: -0.2px; }
h3 { font-size: 1.1rem !important; }
p, label { color: var(--muted); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
    color: var(--muted) !important; opacity: 1 !important;
}

/* ---------- top navigation ---------- */
.st-key-topnav {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 10px 18px; margin-bottom: 26px;
}
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items: center; }
.app-brand { font-size: 16px; font-weight: 800; color: var(--soft-text); letter-spacing: -0.2px; }
.st-key-topnav .stButton > button {
    background: transparent; border: none; box-shadow: none;
    color: var(--muted); font-weight: 600; border-radius: 10px; padding: 8px 10px;
}
.st-key-topnav .stButton > button p { color: inherit; font-size: 14px; }
.st-key-topnav .stButton > button:hover { background: var(--soft); color: var(--soft-text); }
.st-key-topnav .stButton > button[kind="primary"],
.st-key-topnav .stButton > button[data-testid="stBaseButton-primary"] {
    background: var(--deep); color: #fff;
}
.st-key-topnav .stButton > button[kind="primary"] p,
.st-key-topnav .stButton > button[data-testid="stBaseButton-primary"] p { color: #fff !important; }

/* ---------- page pieces ---------- */
.nova-page-title { font-size: 28px; font-weight: 800; color: var(--text); letter-spacing: -0.4px; }
.nova-page-sub { font-size: 14px; color: var(--muted); margin: 4px 0 22px 0; }

[class*="st-key-card_"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 18px 22px; margin-bottom: 6px;
}
.st-key-card_champion [data-testid="stMetric"],
.st-key-card_challenger [data-testid="stMetric"] {
    border: none; padding: 0; background: transparent;
}

/* buttons */
.stButton > button {
    background: var(--card); color: var(--text);
    border: 1px solid var(--border); border-radius: 12px;
    padding: .6rem 1rem; transition: all .2s ease;
}
.stButton > button:hover { border-color: var(--pink); color: var(--soft-text); background: #FDF3F8; }
.stButton > button[kind="primary"] { background: var(--deep); color: #fff; border: none; }
.stButton > button[kind="primary"]:hover { background: #9C577D; color: #fff; }
.stButton > button[kind="primary"] p { color: #fff !important; }

[data-baseweb="input"], [data-baseweb="select"] > div {
    background: var(--card) !important; border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* chat */
[data-testid="stChatMessage"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 16px 20px; margin-bottom: 14px;
}
[data-testid="stChatMessage"] p { color: var(--text); font-size: 15px; line-height: 1.7; }
[data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatInput"] {
    background: var(--card); border: 1px solid var(--pink);
    border-radius: 18px; box-shadow: 0 8px 30px rgba(201,130,168,0.18);
}
[data-testid="stBottom"] > div { background: var(--bg); }

/* metrics */
[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px 18px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    color: var(--muted) !important; font-size: 12.5px !important;
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    color: var(--text) !important; font-weight: 700 !important; font-size: 1.6rem !important;
    white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important;
}

/* expanders, tables, alerts (all one palette: no red / yellow / green) */
[data-testid="stExpander"] {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; overflow: hidden;
}
[data-testid="stExpander"] summary p { color: var(--text); font-weight: 600; font-size: 13px; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
hr { border-color: var(--border); }
[data-testid="stAlert"] {
    background: #FBEFF5 !important; border: 1px solid #EBCFDF !important; border-radius: 12px;
}
[data-testid="stAlert"] * { color: var(--text) !important; }
p code, li code {
    background: var(--soft) !important; color: var(--soft-text) !important;
    border-radius: 6px; padding: 2px 6px; font-family: 'JetBrains Mono', monospace;
}

/* code blocks */
div.nova-code {
    background: #FAF3F7 !important; border: 1px solid var(--border);
    border-left: 3px solid var(--pink); border-radius: 12px;
    padding: 14px 16px; margin: 6px 0 12px 0;
    white-space: normal; word-break: break-word;
    font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
    line-height: 1.75; color: var(--text);
}
.tok-k { color: #A23B72; font-weight: 600; }
.tok-s { color: #8E4A72; }
.tok-n { color: #B0688F; }
.tok-f { color: #7A4F8E; }
.tok-key { color: #7A4F8E; }

/* retrieved-document cards */
[class*="st-key-src_"] {
    background: #FDF8FB; border: 1px solid var(--border);
    border-left: 3px solid var(--pink); border-radius: 12px;
    padding: 14px 18px; margin: 10px 0;
}
.src-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px; }
.src-title { font-size: 15px; font-weight: 700; color: var(--text); }
.src-crumb { font-size: 12px; color: var(--muted); margin-top: 2px; }
.src-chips { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }

/* chat / answer pieces */
.nova-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--soft); color: var(--soft-text);
    border-radius: 999px; padding: 3px 12px; font-size: 12px; font-weight: 600;
}
.nova-pill.warn { background: #F1D3E3; color: #7A2E5A; }
.nova-pill.neutral { background: #F3EAF0; color: var(--muted); }
.nova-greeting { font-size: 30px; font-weight: 700; color: var(--text); margin: 4px 0 4px 0; }
.nova-greeting-sub { font-size: 15px; color: var(--muted); margin-bottom: 22px; }
.nova-msg-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
.nova-msg-who { font-size: 12px; font-weight: 600; letter-spacing: .6px; color: var(--muted); }
.nova-msg-who.bot { font-size: 15px; letter-spacing: 0; color: var(--text); font-weight: 700; }
.nova-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.nova-chip {
    background: #FAF3F7; border: 1px solid var(--border); color: var(--muted);
    border-radius: 999px; padding: 3px 11px; font-size: 11.5px;
}
.nova-chip b { color: var(--soft-text); font-weight: 600; }
.nova-kpi {
    display: inline-block; min-width: 240px; background: #FDF6FA;
    border: 1px solid var(--border); border-left: 4px solid var(--pink);
    border-radius: 12px; padding: 14px 20px; margin: 6px 0 10px 0;
}
.nova-kpi-label { font-size: 12px; color: var(--muted); letter-spacing: .4px; }
.nova-kpi-value { font-size: 30px; font-weight: 800; color: var(--text); line-height: 1.2; }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)

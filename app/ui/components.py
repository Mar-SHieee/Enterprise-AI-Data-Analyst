"""Small reusable UI pieces so every page shares one look."""

from __future__ import annotations

import json
import re
from html import escape

import streamlit as st

BRAND = "#C982A8"
BRAND_DARK = "#8A4F7D"


# --------------------------------------------------------------- layout
def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="nova-page-sub">{escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="nova-page-title">{escape(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def card(key: str):
    """A white rounded card; style lives in theme.py (class st-key-card_*)."""
    return st.container(key=f"card_{key}")


# ----------------------------------------------------------- code blocks
_SQL_WORDS = (
    "SELECT|FROM|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP|ORDER|BY|HAVING|"
    "LIMIT|OFFSET|AS|AND|OR|NOT|IN|IS|NULL|DISTINCT|CASE|WHEN|THEN|ELSE|END|"
    "WITH|UNION|ALL|DESC|ASC|LIKE|BETWEEN"
)
_SQL_TOKEN = re.compile(
    r"('(?:[^']|'')*')"  # 1 string
    r"|(\b\d+(?:\.\d+)?\b)"  # 2 number
    rf"|(\b(?:{_SQL_WORDS})\b)"  # 3 keyword
    r"|(\b[A-Za-z_]+(?=\())",  # 4 function
    re.IGNORECASE,
)
_JSON_TOKEN = re.compile(
    r'("(?:\\.|[^"\\])*")(\s*:)?'
    r"|\b(true|false|null)\b"
    r"|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


def format_sql(sql: str) -> str:
    """Put each major clause on its own line."""
    text = re.sub(r"\s+", " ", str(sql).strip())
    text = re.sub(
        r"\s+((?:(?:LEFT|RIGHT|INNER|OUTER)\s+)?JOIN|FROM|WHERE|GROUP BY|"
        r"ORDER BY|HAVING|LIMIT|UNION)\b",
        r"\n\1",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+(AND|OR)\s", r"\n  \1 ", text, flags=re.IGNORECASE)


def _highlight_sql(text: str) -> str:
    def repl(m: re.Match) -> str:
        kind = "s" if m.group(1) else "n" if m.group(2) else "k" if m.group(3) else "f"
        return f'<span class="tok-{kind}">{m.group(0)}</span>'

    return _SQL_TOKEN.sub(repl, escape(text, quote=False))


def _highlight_json(text: str) -> str:
    def repl(m: re.Match) -> str:
        if m.group(1):
            kind = "key" if m.group(2) else "s"
            return f'<span class="tok-{kind}">{m.group(1)}</span>{m.group(2) or ""}'
        kind = "k" if m.group(3) else "n"
        return f'<span class="tok-{kind}">{m.group(0)}</span>'

    return _JSON_TOKEN.sub(repl, escape(text, quote=False))


def code_block(text: str, language: str = "sql") -> None:
    if language == "sql":
        body = _highlight_sql(format_sql(text))
    elif language == "json":
        body = _highlight_json(text)
    else:
        body = escape(text, quote=False)
    # Streamlit re-renders <pre> itself and turns nested spans into
    # "[object Object]", so use a <div> with explicit line breaks instead.
    body = re.sub(r"(?m)^( +)", lambda m: "&nbsp;" * len(m.group(1)), body)
    body = body.replace("\n", "<br>")
    st.markdown(f'<div class="nova-code">{body}</div>', unsafe_allow_html=True)


def json_block(obj) -> None:
    code_block(json.dumps(obj, indent=2, ensure_ascii=False, default=str), "json")

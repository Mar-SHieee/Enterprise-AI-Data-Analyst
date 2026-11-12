"""AI Analyst — conversational RAG + SQL + prediction interface."""

from __future__ import annotations

import json
import numbers
import re
from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

from app.data.access import get_resources, logger
from app.data.queries import EXAMPLE_QUESTIONS
from app.ui.components import code_block, json_block
from app.ui.rag_format import parse_rag_answer, split_blocks
from app.views.base import Page

# The last example is a prompt-injection demo; keep it off the landing buttons.
SUGGESTIONS = EXAMPLE_QUESTIONS[:3]
RULE_NOTE = re.compile(r"\(\s*Deterministic mode[^)]*\)\s*", re.IGNORECASE)


def _greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning."
    if hour < 18:
        return "Good afternoon."
    return "Good evening."


class AnalystPage(Page):
    title = "AI Analyst"

    # ----------------------------------------------------------- callbacks
    @staticmethod
    def _reset() -> None:
        st.session_state.chat_history = []

    @staticmethod
    def _queue(text: str) -> None:
        st.session_state["pending_question"] = text

    # --------------------------------------------------------------- page
    def render(self) -> None:
        agent, prediction_log = get_resources()
        history = st.session_state.setdefault("chat_history", [])
        pending = st.session_state.pop("pending_question", None)

        left, right = st.columns([5, 1])
        left.markdown(
            f'<div class="nova-greeting">{_greeting()}</div>'
            '<div class="nova-greeting-sub">'
            "What would you like to understand about your business?</div>",
            unsafe_allow_html=True,
        )
        right.button("＋ New chat", width="stretch", on_click=self._reset)

        if not history and not pending:
            cols = st.columns(len(SUGGESTIONS))
            for i, (col, text) in enumerate(zip(cols, SUGGESTIONS)):
                col.button(
                    text,
                    key=f"suggestion_{i}",
                    width="stretch",
                    on_click=self._queue,
                    args=(text,),
                )

        for idx, message in enumerate(history):
            self._render_message(message, idx)

        question = (
            st.chat_input("Ask anything about your business data...") or pending or ""
        ).strip()
        if question:
            self._handle_question(agent, prediction_log, question)

    # ------------------------------------------------------------ handling
    def _handle_question(self, agent, prediction_log, question: str) -> None:
        history = st.session_state.chat_history

        user_msg = {"role": "user", "content": question}
        history.append(user_msg)
        self._render_message(user_msg, len(history) - 1)

        slot = st.empty()
        with slot.container():
            with st.spinner("Analyzing your question..."):
                reply = self._ask(agent, prediction_log, question)
        slot.empty()

        history.append(reply)
        self._render_message(reply, len(history) - 1)

    @staticmethod
    def _ask(agent, prediction_log, question: str) -> dict:
        try:
            result = agent.run(question)
            prediction_log.log_answer(question, result)
            logger.info(
                "answered trace_id=%s status=%s tools=%s",
                result.get("trace_id"),
                result.get("status"),
                result.get("tools_used"),
            )
            return {
                "role": "assistant",
                "content": result.get("answer", "No answer returned."),
                "result": result,
            }
        except Exception:
            logger.exception("Analyst request failed")
            return {
                "role": "assistant",
                "content": "Something went wrong while processing your question. Please try again.",
                "result": {
                    "status": "error",
                    "answer": "Unexpected application error.",
                },
            }

    # ------------------------------------------------------------ messages
    def _render_message(self, message: dict, idx: int) -> None:
        role = message["role"]
        result = message.get("result") or {}
        content = message["content"]

        if role == "assistant":
            content = RULE_NOTE.sub("", content)

        with st.chat_message(role):
            if role == "user":
                st.markdown(
                    '<div class="nova-msg-head"><span class="nova-msg-who">YOU</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="nova-msg-head">'
                    '<span class="nova-msg-who bot">Enterprise AI Data Analyst</span>'
                    f"{self._badge(result)}</div>",
                    unsafe_allow_html=True,
                )

            intro, sources = (
                parse_rag_answer(content) if role == "assistant" else (content, [])
            )
            if sources:
                if intro:
                    st.markdown(intro)
                self._render_sources(sources, idx)
            else:
                text, table, rest = self._split_query_result(content)
                if text:
                    st.markdown(text)
                if table is not None and not result.get("data"):
                    self._render_data(table)
                if rest:
                    st.markdown(rest)

            if role == "assistant" and result:
                self._render_result_details(result)

    @staticmethod
    def _render_sources(sources: list, idx: int) -> None:
        n = len(sources)
        st.markdown(
            f"Found **{n}** relevant passage{'s' if n != 1 else ''} in your documentation."
        )
        for i, src in enumerate(sources):
            with st.container(key=f"src_{idx}_{i}"):
                crumb = (
                    f'<div class="src-crumb">{escape(src["crumb"])}</div>'
                    if src["crumb"]
                    else ""
                )
                st.markdown(
                    '<div class="src-head"><div>'
                    f'<div class="src-title">{escape(src["title"])}</div>{crumb}</div>'
                    '<div class="src-chips">'
                    f'<span class="nova-chip">{escape(src["file"])}</span>'
                    f'<span class="nova-chip">Match <b>{src["match"]}%</b></span>'
                    "</div></div>",
                    unsafe_allow_html=True,
                )
                for kind, lang, text in split_blocks(src["body"]):
                    if kind == "code":
                        code_block(text, "sql" if lang in ("", "sql") else "text")
                    else:
                        st.markdown(text)

    @staticmethod
    def _split_query_result(content: str):
        """Turn a raw 'Query result: [ {...} ]' text blob into a real table."""
        marker = "Query result:"
        if marker not in content:
            return content, None, ""
        head, _, tail = content.partition(marker)
        tail = tail.strip()
        try:
            rows, end = json.JSONDecoder().raw_decode(tail)
        except ValueError:
            return content, None, ""
        if isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
            return head.strip(), pd.DataFrame(rows), tail[end:].strip()
        return content, None, ""

    @staticmethod
    def _render_data(rows) -> None:
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        if df.shape == (1, 1):
            label = str(df.columns[0]).replace("_", " ").title()
            value = df.iat[0, 0]
            if isinstance(value, numbers.Real) and not isinstance(value, bool):
                shown = (
                    f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"
                )
            else:
                shown = str(value)
            st.markdown(
                '<div class="nova-kpi">'
                f'<div class="nova-kpi-label">{escape(label)}</div>'
                f'<div class="nova-kpi-value">{escape(shown)}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(df, width="stretch", hide_index=True)

    @staticmethod
    def _badge(result: dict) -> str:
        status = result.get("status", "unknown")
        if status == "blocked":
            return '<span class="nova-pill warn">Blocked</span>'
        if status == "error":
            return '<span class="nova-pill warn">Error</span>'
        if result.get("sql") or result.get("sources") or result.get("data"):
            return '<span class="nova-pill">Grounded answer</span>'
        return '<span class="nova-pill neutral">Answer</span>'

    def _render_result_details(self, result: dict) -> None:
        status = result.get("status", "unknown")

        if status == "blocked":
            st.warning("This request was blocked by the security layer.")
        elif status == "error":
            reason = result.get("reason") or result.get("error") or ""
            st.error(f"The request could not be completed. {reason}".strip())

        if result.get("data"):
            self._render_data(result["data"])

        tabs = {}
        if result.get("sql"):
            tabs["SQL"] = ("sql", result["sql"])
        if result.get("prediction"):
            tabs["Prediction"] = ("json", result["prediction"])

        if tabs:
            with st.expander("Technical details"):
                for tab, (kind, payload) in zip(st.tabs(list(tabs)), tabs.values()):
                    with tab:
                        if kind == "sql":
                            code_block(payload, "sql")
                        else:
                            json_block(payload)

        findings = (result.get("security") or {}).get("document_findings") or []
        if findings:
            categories = sorted({f["category"] for f in findings})
            st.warning(
                "Prompt-injection patterns were detected and neutralised "
                f"in retrieved documents: {', '.join(categories)}"
            )

        tools = ", ".join(result.get("tools_used") or ["none"])
        st.markdown(
            '<div class="nova-chips">'
            f'<span class="nova-chip">Tools <b>{escape(tools)}</b></span>'
            f'<span class="nova-chip">Latency <b>{result.get("latency_ms", 0):.0f} ms</b></span>'
            "</div>",
            unsafe_allow_html=True,
        )

"""Ask the Analyst — the controlled RAG + SQL + prediction agent."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.access import get_resources, logger
from app.data.queries import EXAMPLE_QUESTIONS
from app.views.base import Page


class AnalystPage(Page):
    title = "Ask the analyst"

    def render(self) -> None:
        agent, prediction_log = get_resources()
        st.header(self.title)
        st.caption(
            "The model routes your question to SQL, the documentation index, "
            "or the prediction model — and only writes prose over what those "
            "tools returned."
        )

        if "history" not in st.session_state:
            st.session_state.history = []

        question = self._render_question_input()
        if st.button("Ask", type="primary") and question.strip():
            self._ask(agent, prediction_log, question)

        for question_text, result in st.session_state.history[:5]:
            self._render_answer(question_text, result)

    def _render_question_input(self) -> str:
        picked = st.selectbox("Example questions", ["—"] + EXAMPLE_QUESTIONS)
        return st.text_input(
            "Your question",
            value="" if picked == "—" else picked,
            placeholder="e.g. What was total revenue in 2011?",
        )

    def _ask(self, agent, prediction_log, question: str) -> None:
        with st.spinner("Routing, querying, grounding…"):
            result = agent.run(question)
        prediction_log.log_answer(question, result)
        logger.info(
            "answered trace_id=%s status=%s tools=%s",
            result.get("trace_id"),
            result.get("status"),
            result.get("tools_used"),
        )
        st.session_state.history.insert(0, (question, result))

    def _render_answer(self, question_text: str, result: dict) -> None:
        st.divider()
        st.markdown(f"**Q:** {question_text}")

        status = result.get("status")
        if status == "blocked":
            st.error(result["answer"])
        elif status == "error":
            st.warning(result["answer"])
        else:
            st.markdown(result["answer"])

        badges = " · ".join(
            [
                f"tools: `{', '.join(result['tools_used']) or 'none'}`",
                f"status: `{status}`",
                f"{result.get('latency_ms', 0):.0f} ms",
                f"trace `{result.get('trace_id')}`",
                "cached" if result.get("cached") else "fresh",
            ]
        )
        st.caption(badges)

        if result.get("data"):
            st.dataframe(
                pd.DataFrame(result["data"]), use_container_width=True, hide_index=True
            )
        if result.get("sql"):
            with st.expander("SQL executed"):
                st.code(result["sql"], language="sql")
        if result.get("prediction"):
            with st.expander("Model prediction"):
                st.json(result["prediction"])
        if result.get("sources"):
            st.caption("Sources: " + " · ".join(f"`{s}`" for s in result["sources"]))

        findings = (result.get("security") or {}).get("document_findings") or []
        if findings:
            st.warning(
                "Prompt-injection patterns were detected and neutralised in "
                f"the retrieved documents: {sorted({f['category'] for f in findings})}"
            )

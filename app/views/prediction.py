"""Customer Prediction — champion (classical) vs challenger (deep MLP)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data.access import get_resources, sql_frame
from app.data.queries import Q_CUSTOMER_IDS
from app.views.base import Page


class PredictionPage(Page):
    title = "Customer prediction"

    def render(self) -> None:
        agent, prediction_log = get_resources()
        st.header(self.title)

        if agent.prediction_tool is None:
            st.info(
                "No prediction model is loaded. Build the artifacts first:\n\n"
                "```bash\npython scripts/train_models.py\n```"
            )
            return

        tool = agent.prediction_tool
        st.caption(tool.describe())

        customer_id, go = self._render_selector()
        if go and customer_id:
            self._render_result(tool, prediction_log, customer_id)

    def _render_selector(self):
        ids = sql_frame(Q_CUSTOMER_IDS)
        options = ids["customer_id"].astype(str).tolist() if not ids.empty else []

        left, right = st.columns([2, 1])
        with left:
            customer_id = (
                st.selectbox("Customer", options)
                if options
                else st.text_input("Customer id")
            )
        with right:
            st.write("")
            st.write("")
            go = st.button("Predict", type="primary")
        return customer_id, go

    def _render_result(self, tool, prediction_log, customer_id: str) -> None:
        result = tool.predict(customer_id=str(customer_id)).to_dict()
        prediction_log.log_prediction(customer_id, result, source="streamlit")

        if result["status"] != "success":
            st.error(result["reason"])
            return

        probability = result["probability"] or 0.0
        champ, chall = st.columns(2)
        with champ:
            st.subheader("Champion — classical")
            st.metric("Repeat-purchase probability", f"{probability:.1%}")
            st.caption(
                f"decision: **{result['label']}** · model `{result['model_version']}`"
            )
            st.progress(min(max(probability, 0.0), 1.0))

        challenger = result.get("challenger")
        with chall:
            st.subheader("Challenger — deep MLP")
            if not challenger or challenger.get("status") != "success":
                st.info("No challenger model loaded.")
            else:
                cp = challenger["probability"] or 0.0
                st.metric("Repeat-purchase probability", f"{cp:.1%}")
                st.caption(
                    f"decision: **{challenger['label']}** · "
                    f"model `{challenger['model_version']}`"
                )
                st.progress(min(max(cp, 0.0), 1.0))
                if challenger["label"] != result["label"]:
                    st.warning("Champion and challenger disagree on this customer.")

        with st.expander(
            "Features used (computed live from the customer's first order)"
        ):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"feature": k, "value": v}
                        for k, v in result["features_used"].items()
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "This is decision support, not a decision. See the limitations "
            "section of the model registry page."
        )

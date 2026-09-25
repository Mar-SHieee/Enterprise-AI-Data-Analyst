"""Executive dashboard — deterministic KPIs, straight from SQL."""

from __future__ import annotations

import streamlit as st

from app.data.access import scalar, sql_frame
from app.data.queries import (
    Q_AOV,
    Q_CUSTOMERS,
    Q_MONTHLY,
    Q_ORDERS,
    Q_REPEAT_RATE,
    Q_TOP_COUNTRIES,
    Q_TOP_CUSTOMERS,
    Q_TOP_PRODUCTS,
    Q_TOTAL_REVENUE,
)
from app.views.base import Page


class DashboardPage(Page):
    title = "Executive dashboard"

    def render(self) -> None:
        st.header(self.title)
        st.caption(
            "Every figure below is produced by a read-only SQL query and "
            "excludes cancelled orders, following "
            "`data/analytics_guidelines.md`."
        )

        self._render_kpis()
        st.divider()
        self._render_monthly_revenue()
        self._render_top_tables()
        self._render_sql_expander()

    def _render_kpis(self) -> None:
        cols = st.columns(5)
        cols[0].metric("Total revenue", f"£{scalar(Q_TOTAL_REVENUE, 0):,}")
        cols[1].metric("Orders", f"{scalar(Q_ORDERS, 0):,}")
        cols[2].metric("Customers", f"{scalar(Q_CUSTOMERS, 0):,}")
        cols[3].metric("Average order value", f"£{scalar(Q_AOV, 0):,}")
        cols[4].metric("Repeat rate", f"{scalar(Q_REPEAT_RATE, 0)}%")

    def _render_monthly_revenue(self) -> None:
        monthly = sql_frame(Q_MONTHLY)
        if not monthly.empty:
            st.subheader("Monthly revenue")
            st.caption("The first and last months of the window are partial.")
            st.line_chart(monthly.set_index("month")["total_revenue"])

    def _render_top_tables(self) -> None:
        left, right = st.columns(2)
        with left:
            st.subheader("Top products by units sold")
            st.dataframe(
                sql_frame(Q_TOP_PRODUCTS),
                use_container_width=True,
                hide_index=True,
            )
        with right:
            st.subheader("Revenue by country")
            st.dataframe(
                sql_frame(Q_TOP_COUNTRIES),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Top customers by lifetime revenue")
        st.dataframe(
            sql_frame(Q_TOP_CUSTOMERS), use_container_width=True, hide_index=True
        )

    def _render_sql_expander(self) -> None:
        with st.expander("Show the SQL behind these numbers"):
            for label, query in [
                ("Total revenue", Q_TOTAL_REVENUE),
                ("Average order value", Q_AOV),
                ("Repeat rate", Q_REPEAT_RATE),
                ("Monthly revenue", Q_MONTHLY),
            ]:
                st.markdown(f"**{label}**")
                st.code(query, language="sql")

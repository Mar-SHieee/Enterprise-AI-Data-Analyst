"""Enterprise AI Data Analyst — Streamlit application package.

Layout:
    app/
        streamlit_app.py   entry point ("streamlit run app/streamlit_app.py")
        data/               SQL query constants + cached data-access helpers
        views/              one class per page, all implementing Page.render()
        ui/                 shared chrome (sidebar, etc.)
"""

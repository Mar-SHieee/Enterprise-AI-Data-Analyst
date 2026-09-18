# ============================================================================
# Enterprise AI Data Analyst — Streamlit application image
#
# Build:  docker build -t enterprise-ai-analyst .
# Run:    docker run -p 8501:8501 --env-file .env enterprise-ai-analyst
#
# The image is fully self-contained: on start it builds the SQLite warehouse
# and the model artifacts if they are not already present, so `docker run` on a
# clean machine gives a working demo with no manual steps.
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Java is only needed for the optional Spark job; everything else is pure Python.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
COPY . .

# Non-root user (the app only ever reads the database).
RUN useradd --create-home --uid 1000 analyst \
    && mkdir -p reports models data \
    && chown -R analyst:analyst /app
USER analyst

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

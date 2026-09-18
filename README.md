# Enterprise AI Data Analyst

**Project 10 — SQL + Spark + ML/DL + RAG + controlled agent + deployment**

An enterprise-style AI analyst for an online retail company. A manager can view
KPIs, ask business questions in plain English, inspect model predictions and get
grounded explanations — with the numbers coming from deterministic analytics,
never from a language model.

The governing design rule of the whole system:

> **The LLM never produces a number.** Figures come from read-only SQL or from a
> versioned model artifact. The model only writes prose over evidence it was
> handed, and cites where each claim came from.

---

## Quick start

```bash
git clone <repo-url>
cd Enterprise-AI-Data-Analyst
pip install -r requirements.txt

cp .env.example .env          # optional: add an LLM key for generated prose

python scripts/build_database.py   # data/sample.csv -> data/retail.db
python scripts/train_models.py     # classical + deep artifacts, comparison table
streamlit run app/streamlit_app.py # http://localhost:8501
```

With no API key the app still works end to end: routing falls back to
deterministic rules, SQL to vetted templates, and answers display the raw tool
evidence. Set `LLM_PROVIDER` and `LLM_API_KEY` in `.env` for generated prose.

---

## Running locally — step by step

This is the exact sequence used to stand the project up on a clean Windows
machine. macOS/Linux users can skip the drive-letter step.

**1. Get the code and move into the project folder**

```bash
git clone <repo-url>
cd Enterprise-AI-Data-Analyst
```

**2. Create an isolated Python environment**

Keeping the project's dependencies separate from anything else installed on
the machine avoids version conflicts.

```bash
python -m venv venv
```

**3. Activate it**

```bash
venv\Scripts\activate          # Windows (Command Prompt / PowerShell)
source venv/bin/activate       # macOS / Linux
```

A successful activation shows `(venv)` at the start of the prompt line. Every
command below assumes the environment is active.

**4. Install the dependencies**

```bash
pip install -r requirements.txt
```

This pulls in Streamlit, pandas, scikit-learn, PyTorch, PySpark and everything
else the app needs. It can take several minutes the first time — PyTorch and
PySpark are large downloads.

**5. Configure environment variables (optional)**

```bash
cp .env.example .env
```

Leave it as-is to run fully offline in rule-based mode, or add `LLM_PROVIDER`
and `LLM_API_KEY` for generated prose. `.env` is git-ignored and must never be
committed.

**6. Build the SQL warehouse**

```bash
python scripts/build_database.py
```

Converts `data/sample.csv` into the SQLite warehouse (`data/retail.db`) that
every dashboard figure and every SQL-tool answer is read from.

**7. Train the models (optional but recommended)**

```bash
python scripts/train_models.py
```

Trains the classical champion model and the deep-learning challenger (MLP),
and writes the comparison table used on the Model Registry page. Skipping this
step leaves the Customer Prediction and Model Registry pages showing "no
artifacts found."

**8. Run the app**

```bash
streamlit run app/streamlit_app.py
```

Streamlit opens automatically at **http://localhost:8501**. Stop it at any
time with `Ctrl+C` in the terminal.

---

## Running with Docker

Docker packages the app together with every dependency it needs, so it runs
the same way on any machine that has Docker installed — no local Python setup,
no manual `venv`, no "works on my machine."

**Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed and **running** (check for "Engine running" in the bottom-left of
the Docker Desktop window before continuing — `docker compose` fails with a
connection error otherwise).

**1. Move into the project folder** (the one containing `docker-compose.yml`)

```bash
cd Enterprise-AI-Data-Analyst
```

**2. Build the image and start the container**

```bash
docker compose up --build
```

`--build` rebuilds the image from the `Dockerfile` — required the first time,
and any time the code or `requirements.txt` changes. The entrypoint builds the
warehouse and the model artifacts automatically on first start, so a clean
machine gets a working demo from this one command. Expect the first build to
take several minutes (it installs Java for Spark and downloads PyTorch inside
the container).

**3. Open the app**

Once the logs show Streamlit is up, open **http://localhost:8501**.

**4. Run it in the background (optional)**

```bash
docker compose up --build -d
docker compose logs -f      # follow the logs
```

**5. Stop the container**

```bash
docker compose down
```

**Equivalent single-container commands**, if you prefer plain `docker` over
`docker compose`:

```bash
docker build -t enterprise-ai-analyst .
docker run -p 8501:8501 --env-file .env enterprise-ai-analyst
```

---

## The application

| Page                    | What it shows                                                                                                                                      |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Executive dashboard** | Revenue, orders, customers, AOV, repeat rate, monthly trend, top products/countries/customers — every figure from a query you can expand and read. |
| **Ask the analyst**     | The agent: routing, SQL executed, result table, citations, security findings, latency, trace id.                                                   |
| **Customer prediction** | Champion (classical) vs challenger (deep MLP) on the same customer, with the features used and a disagreement warning.                             |
| **Model registry**      | Registered artifacts, versions, metrics, ML vs DL comparison, DL training curves.                                                                  |
| **Monitoring**          | Agent evaluation summary, live audit log, prediction log, probability distribution for drift.                                                      |

---

## Architecture

Full diagrams in **[`docs/architecture.md`](docs/architecture.md)**.

```
                      ┌──────────────── Streamlit app ────────────────┐
                      │  dashboard · analyst · prediction · registry  │
                      └───────────────────┬───────────────────────────┘
                                          │
                              ┌───────────▼───────────┐
                              │   AnalyticsAgent      │
                              │  input screening      │
                              │  router (LLM + rules) │
                              └───┬────────┬───────┬──┘
                                  │        │       │
                   ┌──────────────▼──┐  ┌──▼────┐  ▼──────────────┐
                   │ SQL tool        │  │ RAG   │  │ Prediction   │
                   │ validate → RO   │  │ index │  │ champion +   │
                   │ execute         │  │ cite  │  │ challenger   │
                   └────────┬────────┘  └───┬───┘  └──────┬───────┘
                            │               │             │
                   ┌────────▼───────┐  ┌────▼────┐  ┌─────▼──────┐
                   │ SQLite         │  │ 4 KB    │  │ models/    │
                   │ warehouse      │  │ .md docs│  │ .joblib    │
                   └────────────────┘  └─────────┘  └────────────┘
                            ▲
                   ┌────────┴────────┐
                   │ ETL + Spark job │
                   └─────────────────┘
```

---

## Repository layout

```
├── app/streamlit_app.py        # Streamlit application (5 pages)
├── data/                       # sample data + the 4 knowledge-base documents
│   ├── business_definitions.md
│   ├── kpi_definitions.md
│   ├── data_dictionary.md
│   └── analytics_guidelines.md
├── docs/
│   ├── architecture.md         # diagrams + design decisions
│   └── security_guardrails.md  # threat model + defence in depth
├── notebooks/                  # 01 EDA · 02 SQL/ETL · 03 Spark · 04 ML · 05 DL · 06 RAG/agent
├── rag/                        # knowledge layer + controlled agent
│   ├── ingestion.py            # markdown-header-aware chunking
│   ├── vector_store.py         # MiniLM embeddings, TF-IDF fallback
│   ├── retrieval.py            # retrieve → sanitise → cite
│   ├── security.py             # SQL validator + injection defence
│   ├── sql_tool.py             # read-only execution, row/time caps
│   ├── agent.py                # routing, tools, grounded answer
│   ├── audit.py                # JSONL audit log
│   └── evaluation.py           # the four agent metrics
├── scripts/
│   ├── build_database.py       # csv → SQLite warehouse
│   └── train_models.py         # classical + DL + comparison table
├── spark/spark_pipeline.py     # portable Spark ETL/feature job
├── sql/                        # schema + 12 analytical queries
├── src/
│   ├── data/loader.py          # cleaning + validation
│   ├── features/engineering.py # leakage-safe first-order features
│   ├── models/train.py         # classical pipelines + tuning
│   ├── models/dl.py            # MLP: class weighting, early stopping
│   ├── models/predict.py       # serving: champion + challenger
│   ├── evaluation/metrics.py
│   └── utils/                  # config + logging / prediction log
├── tests/                      # 60 tests, no network required
├── Dockerfile · docker-compose.yml
└── requirements.txt
```

---

## Results

### Agent (19 labelled cases, offline mode)

| Metric                  | Score            |
| ----------------------- | ---------------- |
| Tool-selection accuracy | 100%             |
| Groundedness            | 100%             |
| Hallucination rate      | 0%               |
| Task completion         | 100%             |
| SQL guardrail accuracy  | 100% (12 probes) |

Full breakdown, method and caveats: [`reports/agent_evaluation.md`](reports/agent_evaluation.md).

### Models — classical vs deep

See [`reports/model_comparison.csv`](reports/model_comparison.csv). On the
committed 300-row sample the tuned random forest leads on average precision and
the MLP trades precision for recall. **This is a data-size result, not a model
result** — see Limitations.

---

## Reproducing everything

```bash
python scripts/build_database.py     # warehouse
python spark/spark_pipeline.py       # Spark features (needs Java)
python scripts/train_models.py       # model artifacts + comparison
pytest -q                            # 60 tests
```

Notebooks run top-to-bottom in order (01 → 06) and regenerate every report.

---

## Configuration

All settings come from the environment with repo-relative defaults; see
`.env.example`. Nothing secret is ever committed — `config.summary()` prints
`"set" / "not set"` instead of the key.

| Variable              | Default          | Purpose                                            |
| --------------------- | ---------------- | -------------------------------------------------- |
| `LLM_PROVIDER`        | `rule`           | `openai` \| `gemini` \| `rule` (offline)           |
| `LLM_API_KEY`         | —                | Empty key automatically falls back to offline mode |
| `DB_PATH`             | `data/retail.db` | Warehouse location                                 |
| `KNOWLEDGE_DIR`       | `data`           | Knowledge-base markdown                            |
| `SQL_MAX_ROWS`        | `200`            | Result cap fed to the model                        |
| `SQL_TIMEOUT_SECONDS` | `10`             | Wall-clock query interrupt                         |
| `RETRIEVAL_TOP_K`     | `4`              | Chunks retrieved per question                      |
| `MAX_QUESTION_LENGTH` | `1000`           | Input length cap                                   |

---

## Security

Threat model and the full four-layer guardrail stack:
[`docs/security_guardrails.md`](docs/security_guardrails.md).

Summary: generated SQL must be a single `SELECT`/`WITH`, passes a deny-list, may
not contain comments or touch catalogue tables, and is capped by rows and by
wall-clock time. Underneath that the connection itself is opened `mode=ro`, so a
validator bug still cannot write. Retrieved documents are scanned for
prompt-injection, redacted visibly, and wrapped as untrusted data. Every
attempt — allowed or blocked — lands in the audit log with a trace id.

---

## Limitations

State these in the defence rather than waiting to be asked:

1. **Data size.** The committed `data/sample.csv` is 300 rows / 261 customers /
   ~25 repeat buyers. Every model metric in this repo has a confidence interval
   wide enough to swallow the differences between models. Re-run on the full
   Online Retail II extract before quoting a figure.
2. **The deep model does not beat the forest here**, and on this sample it
   should not be expected to. The DL track's value is the controlled comparison
   and the serving infrastructure, not a win.
3. **Groundedness is numeric, not semantic.** An invented revenue figure is
   caught; a qualitative claim contradicting a source document is not.
4. **Injection detection is pattern-based** — a speed bump against novel
   paraphrases. The real guarantee is the absence of any write path.
5. **No probability calibration.** Predictions rank customers; they are not
   calibrated risks and are labelled as scores in the UI.
6. **Role-based access is documented as a design, not implemented** as
   authentication.
7. **This is a decision-support prototype**, not a production lending, pricing
   or retention system.

---

## Future work

- Re-run the full pipeline on the complete dataset with repeated stratified CV.
- Probability calibration (Platt / isotonic) plus a reliability curve.
- LLM-judge groundedness pass alongside the mechanical check.
- Larger eval set, labelled by someone other than the router's author.
- Swap the NumPy index for FAISS once the knowledge base outgrows a few hundred
  chunks (`Retriever` already hides the store).
- Scheduled drift monitoring on the prediction log rather than an on-demand chart.

---

## Team

| Member                     | Track                                           | Key files                                                    |
| -------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| Member 1 — Mariam Ramdan   | Data engineering, SQL warehouse, EDA            | `notebooks/01`, `notebooks/02`, `sql/`, `data/`, `src/data/` |
| Member 2 — Mosaab Mohamed  | Spark / big-data pipeline                       | `notebooks/03`, `spark/`                                     |
| Member 3 — Mahmoud Walid   | Classical ML                                    | `notebooks/04`, `src/models/`, `models/`                     |
| Member 4 — Mahmoud Elaraby | Deep learning                                   | `notebooks/05`, `src/models/`, `models/`                     |
| Member 5 — Mokhtar Ibrahim | RAG knowledge layer, controlled agent, security | `notebooks/06`, `rag/`                                       |
| Member 6 — Maram Ahmed     | Streamlit app, MLOps, Docker, testing, README   | `app/`, `Dockerfile`, `tests/`, `src/models/predict.py`      |

Dataset: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Please cite the original dataset creators in any published report.

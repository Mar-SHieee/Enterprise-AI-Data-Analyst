# Enterprise AI Data Analyst

**SQL warehouse · Spark pipeline · classical ML + deep learning · RAG agent · Streamlit deployment**

An AI analyst for an online retail business. A manager can open one dashboard,
see live KPIs, ask business questions in plain English, check which customers
are likely to buy again, and see exactly how the assistant reached every
answer it gave.

The rule that shapes every part of the system:

> **The AI never invents a number.** Every figure comes from a read-only SQL
> query or a versioned model artifact. The language model only writes prose
> over evidence it was handed, and states where each claim came from.

---

## Table of contents

- [What it does](#what-it-does)
- [Why the numbers can be trusted](#why-the-numbers-can-be-trusted)
- [Quick start](#quick-start)
- [Running locally, step by step](#running-locally-step-by-step)
- [Running with Docker](#running-with-docker)
- [Deploying a public demo](#deploying-a-public-demo)
- [The application](#the-application)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Results](#results)
- [Reproducing everything](#reproducing-everything)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Team](#team)

---

## What it does

**A live dashboard, not a static report.** Revenue, order count, customer
count, average order value, and repeat-purchase rate, computed from the
current data, alongside a monthly trend and the top products, countries, and
customers. The exact SQL behind the key figures can be expanded and read, so a
number can be checked rather than trusted blindly.

**Ask a question instead of writing a query.** "What was total revenue in
2011?" or "List our high-value customers" gets answered directly. The question
is routed to a SQL query, to the company's own written definitions of terms
like "repeat customer," or to the prediction model, whichever the question
calls for, and the answer states which one it used, with the query or the
source document attached.

**Know who is likely to buy again, before they don't.** For any customer, the
system scores the odds of a second order using their first-order behaviour:
spend, quantity, product mix, and timing. Two models score every customer side
by side, a tuned classical model and a neural network, so a retention decision
gets a second opinion instead of a single black-box number. Disagreement
between the two is flagged, which is exactly the case worth a human look.

**A record of every model in production.** Which model is live, what it was
trained on, when, and how it performs, all in one place, alongside the
comparison between the classical and deep-learning approach so that promoting
a challenger model is a documented decision, not a silent swap.

**A log of what the assistant actually did.** Every question asked, every
query run, every block and why, and how long each took. An attempt to run a
destructive command or manipulate the assistant is logged and refused, not
silently allowed through.

---

## Why the numbers can be trusted

The most common failure of an AI analyst is a confident, wrong number. This
system is built to make that structurally difficult:

- Anything that generates SQL is restricted to a single read-only `SELECT`,
  checked against a deny-list, capped in rows and execution time, and run
  against a database connection opened read-only at the OS level. A bug in
  the validator still cannot write to the database.
- Documents retrieved for context are scanned for hidden instructions before
  the model sees them, and are always framed as reference material, never as
  commands.
- On a labelled set of 19 questions and 12 adversarial SQL probes, the system
  picked the correct tool 100% of the time, every number in its answers
  traced back to a real source, and every unsafe query was correctly blocked.

Full threat model and defence layers: [`docs/security_guardrails.md`](docs/security_guardrails.md).

---

## Quick start

```bash
git clone <repo-url>
cd Enterprise-AI-Data-Analyst
pip install -r requirements.txt

cp .env.example .env               # optional: add an LLM key for generated prose

python scripts/build_database.py   # data/sample.csv -> data/retail.db
python scripts/train_models.py     # classical + deep artifacts, comparison table
streamlit run app/streamlit_app.py # http://localhost:8501
```

With no API key the app still works end to end: routing falls back to
deterministic rules, SQL to vetted templates, and answers display the raw tool
evidence (documentation passages are shown as readable source cards). Set
`LLM_PROVIDER`, `LLM_MODEL` and `LLM_API_KEY` in `.env` for generated prose;
see [Connecting an LLM](#connecting-an-llm).

---

## Running locally, step by step

The exact sequence used to stand the project up on a clean machine.

**1. Get the code and move into the project folder**

```bash
git clone <repo-url>
cd Enterprise-AI-Data-Analyst
```

**2. Create an isolated Python environment**

Keeping the project's dependencies separate from anything else on the machine
avoids version conflicts.

```bash
python -m venv venv
```

**3. Activate it**

```bash
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
```

A successful activation shows `(venv)` at the start of the prompt. Every
command below assumes the environment is active.

**4. Install the dependencies**

```bash
pip install -r requirements.txt
```

Pulls in Streamlit, pandas, scikit-learn, PyTorch, PySpark, and everything
else the app needs. It can take several minutes the first time; PyTorch and
PySpark are large downloads.

The interface relies on recent Streamlit features (`st.container(key=...)`
and `width="stretch"`), so keep the Streamlit version pinned in
`requirements.txt` and upgrade it if the app reports an unexpected argument.

**5. Configure environment variables (optional)**

```bash
cp .env.example .env
```

Leave it as-is to run fully offline in rule-based mode, or connect a language
model as described in [Connecting an LLM](#connecting-an-llm). `.env` is
git-ignored and must never be committed.

**6. Build the SQL warehouse**

```bash
python scripts/build_database.py
```

Converts `data/sample.csv` into the SQLite warehouse (`data/retail.db`) that
every dashboard figure and every SQL-tool answer reads from. This file is not
checked into the repository, since it is fully reproducible from the sample
data; it must be built once after every fresh clone.

**7. Train the models**

```bash
python scripts/train_models.py
```

Trains the classical champion model and the deep-learning challenger (MLP),
and writes the comparison table used on the Model Registry page. This
retrains from scratch and overwrites whatever is already in `models/`, so it
only needs to be run once after a clone, or again deliberately, for example
to pick up a new scikit-learn version. Skipping it leaves the Customer
Prediction and Model Registry pages showing "no artifacts found," although
this repository already ships trained artifacts so the step is not required
to explore the app.

**8. Run the app**

```bash
streamlit run app/streamlit_app.py
```

Streamlit opens at **http://localhost:8501**. Stop it with `Ctrl+C`.

### Connecting an LLM

The agent already calls the language model for three things: routing the
question, generating SQL, and writing the final answer. Connecting one needs
no code change, only environment variables in a `.env` file in the repository
root (next to `app/` and `data/`):

```
LLM_PROVIDER=gemini                 # gemini | openai | rule
LLM_MODEL=<model name from your provider's docs>
LLM_API_KEY=<your key>
LLM_BASE_URL=                       # only for OpenAI-compatible providers
```

- `openai` covers any OpenAI-compatible endpoint (OpenAI, Groq, Together,
  NVIDIA NIM, OpenRouter, a local vLLM server). Set `LLM_BASE_URL` for
  anything other than OpenAI itself.
- `LLM_MODEL` defaults to `gpt-4o-mini`, so set it explicitly for any other
  provider.
- Install the matching SDK: `pip install openai` or `pip install google-genai`
  (and keep it in `requirements.txt`).
- With no key, or if the provider cannot be initialised, the app falls back to
  rule-based mode instead of failing.

Check the connection before starting the app:

```bash
python -c "from rag.llm import get_llm; print(get_llm().name)"
```

It prints `gemini` or `openai` when the model is connected, and `rule` (with a
message explaining why) when it is not. **Restart Streamlit after changing
`.env`:** the agent is created once per server process and cached.

---

## Running with Docker

Docker packages the app with every dependency it needs, so it runs the same
way on any machine with Docker installed: no local Python setup, no manual
`venv`.

**Prerequisite:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed and running (check for "Engine running" in the bottom-left of the
Docker Desktop window; `docker compose` fails with a connection error
otherwise).

**1. Move into the project folder** (the one containing `docker-compose.yml`)

```bash
cd Enterprise-AI-Data-Analyst
```

**2. Build the image and start the container**

```bash
docker compose up --build
```

`--build` rebuilds the image from the `Dockerfile`, required the first time
and whenever the code or `requirements.txt` changes. The entrypoint builds the
warehouse and the model artifacts automatically on first start, so a clean
machine gets a working demo from this one command. The first build takes
several minutes: it installs Java for Spark and downloads PyTorch inside the
container.

**3. Open the app**

Once the logs show Streamlit is up, open **http://localhost:8501**.

**4. Run it in the background (optional)**

```bash
docker compose up --build -d
docker compose logs -f
```

**5. Stop the container**

```bash
docker compose down
```

**Plain Docker**, without Compose:

```bash
docker build -t enterprise-ai-analyst .
docker run -p 8501:8501 --env-file .env enterprise-ai-analyst
```

---

## Deploying a public demo

Two routes are practical:

- **Docker** (already supported above) on any host that runs containers. The
  entrypoint builds the warehouse and model artifacts on first start.
- **Streamlit Community Cloud**: connect the GitHub repository and set the
  main file to `app/streamlit_app.py`.

Checklist before publishing a link:

1. `requirements.txt` is complete, including the LLM SDK you chose. PySpark is
   only used by `spark/spark_pipeline.py`; a hosted demo can use a trimmed
   requirements file without it (confirm nothing under `app/` or `rag/`
   imports it first).
2. `.streamlit/config.toml` sits in the **repository root**, not inside `app/`.
3. `data/retail.db` is not in git. For a hosted demo either commit a built
   copy (it is small for the 300-row sample) or use the Docker route.
4. Put the LLM settings in the host's **Secrets** (Streamlit Cloud exposes
   top-level secrets as environment variables, so no code change is needed):
   ```toml
   LLM_PROVIDER = "gemini"
   LLM_MODEL = "<model name>"
   LLM_API_KEY = "<your key>"
   ```
   Never commit `.env` or a key.

Things to know about a hosted demo:

- **Logs are not durable.** Hosted filesystems are usually ephemeral, so the
  audit and prediction logs reset on restart and the Monitoring page starts
  empty after a redeploy.
- **A public link with a live key spends your quota.** Use a key with a small
  spending limit.

---

## The application

The interface uses a single light pink/mauve theme with a top navigation bar
(there is no sidebar). The **Online** badge at the right of the bar shows
whether the SQL tool is connected; hover it to see the active model, the
database, and the number of available tools.

| Page                     | What it shows                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AI Analyst** (default) | A chat with suggested starter questions. Answers appear as result tables, or as a single-figure card for one-number results. The SQL that ran and any model output sit under _Technical details_. Documentation answers are shown as source cards with the section title, file name, and match score. Security warnings, the tools used, and latency are shown with each answer. The routing reason and trace id are recorded in the audit log. |
| **Dashboard**            | Revenue, orders, customers, AOV, repeat rate, monthly trend, top products/countries/customers, with the SQL behind the key figures available in an expander.                                                                                                                                                                                                                                                                                    |
| **Predictions**          | Champion (classical) vs. challenger (deep MLP) scored on the same customer, with the features used and a disagreement warning. Model details are in a collapsed section.                                                                                                                                                                                                                                                                        |
| **Model Registry**       | Registered artifacts, versions, metrics, the ML vs. DL comparison, and deep-learning training curves.                                                                                                                                                                                                                                                                                                                                           |
| **Monitoring**           | Agent evaluation summary, live audit log (including the routing reason), prediction log, and probability distribution for drift.                                                                                                                                                                                                                                                                                                                |

---

## Architecture

Full diagrams and design decisions: [`docs/architecture.md`](docs/architecture.md).

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
                   │ SQLite         │  │ 4 short │  │ models/    │
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
├── .streamlit/
│   └── config.toml              # Streamlit theme (must sit in the repo root)
├── app/                         # Streamlit application
│   ├── streamlit_app.py         # entry point ("streamlit run app/streamlit_app.py")
│   ├── data/                    # SQL query constants + cached data-access helpers
│   ├── views/                   # one class per page (analyst, dashboard, prediction, registry, monitoring)
│   └── ui/                      # shared look and feel
│       ├── theme.py             # global CSS: palette, cards, chat, code blocks
│       ├── topnav.py            # top navigation bar + Online badge
│       ├── components.py        # page header, cards, SQL/JSON code blocks
│       └── rag_format.py        # turns raw retrieved-document text into readable source cards
├── data/                        # sample data + the knowledge-base documents
│   ├── business_definitions.md
│   ├── kpi_definitions.md
│   ├── data_dictionary.md
│   └── analytics_guidelines.md
├── docs/
│   ├── architecture.md          # diagrams + design decisions
│   └── security_guardrails.md   # threat model + defence in depth
├── notebooks/                   # 01 EDA · 02 SQL/ETL · 03 Spark · 04 ML · 05 DL · 06 RAG/agent
├── rag/                         # knowledge layer + controlled agent
│   ├── ingestion.py              # markdown-header-aware chunking
│   ├── vector_store.py           # MiniLM embeddings, TF-IDF fallback
│   ├── retrieval.py               # retrieve → sanitise → cite
│   ├── security.py                # SQL validator + injection defence
│   ├── sql_tool.py                # read-only execution, row/time caps
│   ├── llm.py                     # provider-agnostic LLM wrapper (openai-compatible, gemini, rule)
│   ├── agent.py                   # routing, tools, grounded answer
│   ├── audit.py                    # JSONL audit log
│   └── evaluation.py               # the four agent metrics
├── scripts/
│   ├── build_database.py         # csv → SQLite warehouse
│   └── train_models.py           # classical + DL + comparison table
├── spark/spark_pipeline.py       # portable Spark ETL/feature job
├── sql/                          # schema + 12 analytical queries
├── src/
│   ├── data/loader.py             # cleaning + validation
│   ├── features/engineering.py    # leakage-safe first-order features
│   ├── models/train.py            # classical pipelines + tuning
│   ├── models/dl.py                # MLP: class weighting, early stopping
│   ├── models/predict.py           # serving: champion + challenger
│   ├── evaluation/metrics.py
│   └── utils/                      # config + logging / prediction log
├── tests/                        # automated tests, no network required
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

Full breakdown, method, and caveats: [`reports/agent_evaluation.md`](reports/agent_evaluation.md).

### Models, classical vs. deep

See [`reports/model_comparison.csv`](reports/model_comparison.csv) once
generated. On the committed 300-row sample, the tuned random forest leads on
average precision and the MLP trades precision for recall. This is a
data-size result, not a model result; see [Limitations](#limitations).

---

## Reproducing everything

```bash
python scripts/build_database.py     # warehouse
python spark/spark_pipeline.py       # Spark features (needs Java)
python scripts/train_models.py       # model artifacts + comparison
pytest -q                            # automated tests
```

Notebooks run top to bottom in order (01 to 06) and regenerate every report.

---

## Configuration

All settings come from the environment with repo-relative defaults; see
`.env.example`. Nothing secret is ever committed; `config.summary()` prints
`"set"` or `"not set"` instead of the key itself.

| Variable              | Default          | Purpose                                                                     |
| --------------------- | ---------------- | --------------------------------------------------------------------------- |
| `LLM_PROVIDER`        | `rule`           | `openai` \| `gemini` \| `rule` (offline)                                    |
| `LLM_MODEL`           | `gpt-4o-mini`    | Model name for the chosen provider; set it explicitly when not using OpenAI |
| `LLM_API_KEY`         | —                | Empty key falls back to offline mode automatically                          |
| `LLM_BASE_URL`        | —                | Endpoint for OpenAI-compatible providers (Groq, OpenRouter, vLLM, …)        |
| `DB_PATH`             | `data/retail.db` | Warehouse location                                                          |
| `KNOWLEDGE_DIR`       | `data`           | Knowledge-base markdown                                                     |
| `SQL_MAX_ROWS`        | `200`            | Result cap fed to the model                                                 |
| `SQL_TIMEOUT_SECONDS` | `10`             | Wall-clock query interrupt                                                  |
| `RETRIEVAL_TOP_K`     | `4`              | Chunks retrieved per question                                               |
| `MAX_QUESTION_LENGTH` | `1000`           | Input length cap                                                            |

---

## Troubleshooting

**The app answers in rule-based mode (the Online badge tooltip shows
`Model: rule`).** The language model is not connected. Check that `.env`
exists in the repository root and contains `LLM_PROVIDER`, `LLM_MODEL` and
`LLM_API_KEY`, that the SDK is installed, and run the check in
[Connecting an LLM](#connecting-an-llm). Then restart Streamlit.

**The page shows default Streamlit colours.** `.streamlit/config.toml` must be
in the folder you run `streamlit run` from (the repository root), not inside
`app/`. Most of the styling comes from `app/ui/theme.py` and applies either
way.

**An error mentions an unexpected `width` or `key` argument.** The installed
Streamlit is too old for the interface. Upgrade it with
`pip install --upgrade streamlit`.

**Prediction or Model Registry pages show "no artifacts found".** Run
`python scripts/train_models.py`.

**Dashboard shows no data or "No database connected".** Run
`python scripts/build_database.py`.

---

## Security

Threat model and the full four-layer guardrail stack:
[`docs/security_guardrails.md`](docs/security_guardrails.md).

In short: generated SQL must be a single `SELECT` or `WITH` statement, passes
a deny-list, may not contain comments or touch catalogue tables, and is capped
by row count and wall-clock time. Underneath that, the connection itself is
opened `mode=ro`, so a validator bug still cannot write. Retrieved documents
are scanned for prompt injection, redacted visibly, and wrapped as untrusted
data. Every attempt, allowed or blocked, lands in the audit log with a trace
id.

---

## Limitations

Stated up front rather than left to be discovered:

1. **Data size.** The committed `data/sample.csv` is 300 rows, about 260
   customers, roughly 25 repeat buyers. Every model metric here has a
   confidence interval wide enough to swallow the differences between models.
   Re-run on the full Online Retail II extract before quoting a figure.
2. **The deep model does not beat the forest here**, and is not expected to
   at this sample size. The DL track's value is the controlled comparison and
   the serving infrastructure, not a performance win.
3. **Groundedness is numeric, not semantic.** An invented revenue figure is
   caught; a qualitative claim that contradicts a source document is not.
4. **Injection detection is pattern-based**, a speed bump against novel
   paraphrases. The real guarantee is the absence of any write path.
5. **No probability calibration.** Predictions rank customers; they are not
   calibrated risks, and are labelled as scores in the UI for that reason.
6. **Role-based access is documented as a design**, not implemented as
   authentication.
7. **This is a decision-support prototype**, not a production lending,
   pricing, or retention system.

---

## Future work

- Re-run the full pipeline on the complete dataset with repeated stratified CV.
- Probability calibration (Platt or isotonic) plus a reliability curve.
- An LLM-judge groundedness pass alongside the mechanical check.
- A larger evaluation set, labelled by someone other than the router's author.
- Swap the NumPy index for FAISS once the knowledge base outgrows a few
  hundred chunks (`Retriever` already hides the store).
- Scheduled drift monitoring on the prediction log rather than an on-demand
  chart.
- Stream the model's answer token by token in the chat.

---

## Team

| Area                                                 | Owner           | Key files                                                    |
| ---------------------------------------------------- | --------------- | ------------------------------------------------------------ |
| Data engineering, SQL warehouse, EDA                 | Mariam Ramdan   | `notebooks/01`, `notebooks/02`, `sql/`, `data/`, `src/data/` |
| Spark / big-data pipeline                            | Mosaab Mohamed  | `notebooks/03`, `spark/`                                     |
| Classical machine learning                           | Mahmoud Walid   | `notebooks/04`, `src/models/`, `models/`                     |
| Deep learning                                        | Mahmoud Elaraby | `notebooks/05`, `src/models/`, `models/`                     |
| RAG knowledge layer, controlled agent, security      | Mokhtar Ibrahim | `notebooks/06`, `rag/`                                       |
| Streamlit app, MLOps, Docker, testing, documentation | Maram Ahmed     | `app/`, `Dockerfile`, `tests/`, `src/models/predict.py`      |

Dataset: [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
Please cite the original dataset creators in any published report.

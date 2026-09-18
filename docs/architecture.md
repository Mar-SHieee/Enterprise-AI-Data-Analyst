# System Architecture

## 1. End-to-end flow

```mermaid
flowchart TB
    subgraph SRC["Source"]
        RAW["UCI Online Retail II<br/>(Excel, 2 sheets)"]
    end

    subgraph ING["Data engineering — Member 1"]
        CLEAN["clean_data()<br/>dedupe · drop null customer<br/>price&gt;0 · qty≠0 · revenue"]
        DB[("SQLite warehouse<br/>customers · products<br/>orders · order_items")]
    end

    subgraph BIG["Big data — Member 2"]
        SPARK["Spark job<br/>spark/spark_pipeline.py"]
        FEAT["customer_features<br/>product_features<br/>country_revenue<br/>(parquet)"]
    end

    subgraph ML["Modelling — Members 3 & 4"]
        ENG["Feature engineering<br/>first-order features only"]
        CLASSIC["Classical: LogReg · RF<br/>tuned via GridSearchCV"]
        DL["Deep: MLP<br/>class-weighted · early stopping"]
        ART["Model registry<br/>.joblib + .json"]
    end

    subgraph AGENT["Advanced AI — Member 5"]
        KB["Knowledge base<br/>4 markdown documents"]
        VEC["Vector store<br/>MiniLM → TF-IDF fallback"]
        ROUTER{"Router<br/>LLM + rule fallback"}
        TSQL["SQL tool<br/>validate → read-only exec"]
        TRAG["Retrieval tool<br/>retrieve → sanitise → cite"]
        TPRED["Prediction tool<br/>champion + challenger"]
        ANS["Grounded answer<br/>prose over evidence only"]
    end

    subgraph APP["Deployment — Member 6"]
        UI["Streamlit<br/>dashboard · analyst · prediction<br/>registry · monitoring"]
        LOGS["Audit log + prediction log<br/>JSONL"]
    end

    RAW --> CLEAN --> DB
    CLEAN --> SPARK --> FEAT
    DB --> ENG --> CLASSIC --> ART
    ENG --> DL --> ART
    KB --> VEC
    ROUTER --> TSQL --> DB
    ROUTER --> TRAG --> VEC
    ROUTER --> TPRED --> ART
    TPRED --> DB
    TSQL --> ANS
    TRAG --> ANS
    TPRED --> ANS
    UI --> ROUTER
    ANS --> UI
    ANS --> LOGS
    DB --> UI
    ART --> UI
```

## 2. Request path for one question

```mermaid
sequenceDiagram
    participant U as Manager
    participant A as Streamlit
    participant AG as AnalyticsAgent
    participant S as Security
    participant T as Tools
    participant L as LLM
    participant AU as Audit log

    U->>A: "Total revenue in 2011, and what does revenue mean?"
    A->>AG: agent.run(question)
    AG->>S: check_user_input()
    S-->>AG: allowed (findings logged)
    AG->>L: route(question)
    L-->>AG: {"tools": ["sql", "rag"]}
    Note over AG: LLM unavailable or malformed → deterministic rules
    AG->>T: retrieve() → sanitise → cite
    AG->>L: generate SQL from schema + context
    AG->>T: validate_sql() → read-only execute
    T-->>AG: rows + latency
    AG->>L: write prose over THIS evidence only
    L-->>AG: answer with [S1] citations
    AG->>AU: trace_id, tools, sql, status, latency
    AG-->>A: {answer, sql, data, sources, prediction, status}
    A-->>U: number + explanation + sources + SQL
```

## 3. Guardrail stack

```mermaid
flowchart LR
    Q["Question"] --> L0["L0 input screening<br/>injection patterns · length cap"]
    L0 --> GEN["SQL generation"]
    GEN --> L1["L1 validator<br/>SELECT/WITH only · single statement<br/>deny-list · no comments · no catalogue"]
    L1 --> L2["L2 read-only connection<br/>file:...?mode=ro"]
    L2 --> L3["L3 resource limits<br/>max_rows · wall-clock interrupt"]
    L3 --> L4["L4 audit log<br/>every attempt, allowed or blocked"]
    L4 --> OUT["Result"]

    DOC["Retrieved document"] --> SAN["sanitize_document()<br/>redact + flag"]
    SAN --> WRAP["&lt;retrieved_documents&gt;<br/>marked UNTRUSTED"]
    WRAP --> GEN
```

No single layer is trusted. L1 is a syntactic filter that a novel paraphrase
could in principle slip past; L2 is the guarantee — the process holds no write
handle to the database at all.

## 4. Component ownership

| Layer | Component | Owner |
|---|---|---|
| Ingestion, cleaning, warehouse, analytical SQL | `src/data/`, `sql/`, notebooks 01–02 | Member 1 |
| Scalable transformation | `spark/spark_pipeline.py`, notebook 03 | Member 2 |
| Classical ML | `src/features/`, `src/models/train.py`, notebook 04 | Member 3 |
| Deep learning | `src/models/dl.py`, notebook 05 | Member 4 |
| RAG, agent, guardrails, evaluation | `rag/`, `docs/security_guardrails.md`, notebook 06 | Member 5 |
| Streamlit app, MLOps, logging, Docker | `app/`, `src/utils/logger.py`, `Dockerfile` | Member 6 |

## 5. Key design decisions

| Decision | Alternative rejected | Why |
|---|---|---|
| LLM never emits a final number | Let the model answer from context | A fabricated revenue figure is the one failure a manager cannot detect. Numbers come from SQL or a model artifact; the LLM only narrates them. |
| Deterministic rule fallback for routing and SQL | LLM-only agent | The demo, the tests and the Docker image must work with no API key. The system degrades to templates rather than failing. |
| NumPy cosine index, TF-IDF fallback | FAISS / Chroma / pgvector | The knowledge base is ~50 chunks. A vector service would add a container and a download for no retrieval benefit; `Retriever` hides the store so it can be swapped later. |
| Champion/challenger served together | Ship only the better model | Both tracks are compared on every live prediction, and disagreement is surfaced in the UI rather than buried in a notebook. |
| Read-only SQLite URI | Application-level checks only | Defence in depth: a validator bug cannot become data loss. |
| Markdown-header-aware chunking | Fixed-size windows | Every chunk is one concept, so citations point at a definition instead of a page. |

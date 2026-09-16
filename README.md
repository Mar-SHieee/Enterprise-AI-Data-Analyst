# Enterprise-AI-Data-Analyst
## Member 1 — Data Engineering & SQL Pipeline (Mariam)

### Setup
```bash
git clone <repo-url>
cd Enterprise-AI-Data-Analyst
pip install -r requirements.txt
```

### Run the pipeline
1. Open `notebooks/01_eda.ipynb` → Restart Kernel → Run All.
   - Downloads the raw UCI Online Retail II dataset into `data/raw/`.
   - Runs data quality checks, EDA, and cleaning (`clean_data()` in `src/data/loader.py`).
2. Open `notebooks/02_sql_etl.ipynb` → Restart Kernel → Run All.
   - Builds `sql/schema.sql` + `sql/analytical_queries.sql`.
   - Loads the cleaned data into a SQLite warehouse (`retail.db`).
   - Runs 12 analytical SQL queries (JOINs, CTEs, window functions).
   - Saves `data/sample.csv` and runs the smoke tests.

### Tests
```bash
pytest tests/test_smoke.py -v
```

### Docs
See `reports/data_dictionary.md` for full column definitions, cleaning rules, and the SQL schema mapping.

## Member 4 — Deep Learning (Mahmoud Elaraby)

### Setup
```bash
git clone <repo-url>
cd Enterprise-AI-Data-Analyst
pip install -r requirements.txt
```

### Run the pipeline
1. Open `notebooks/05_dl_model.ipynb` → Restart Kernel → Run All.
   - Loads `data/sample.csv` (same source as `04_ml_models.ipynb`) and builds
     the customer feature table via `build_repeat_purchase_dataset()` — the
     exact same features, target and train/test split (`test_size=0.25,
     stratify=y, random_state=42`) as Member 3's classical baseline.
   - Trains `ChurnMLP`, an MLP with early stopping and LR scheduling
     (`src/models/train.py`).
   - Loads Member 3's saved `classical_repeat_purchase_rf_v1.joblib` and
     scores it on the same held-out test set for a direct, reproducible
     ML-vs-DL comparison.
   - Re-runs the same pipeline on the full cleaned `Online Retail II`
     dataset (`data/online_retail_II.xlsx`, if present) to show how the
     metrics change with more data.
   - Computes permutation importance and saves the DL artifact.
2. Requires `data/online_retail_II.xlsx` in `data/` to run the full-dataset
   section (see `src/data/loader.download_raw_data()`); otherwise that
   section is skipped and only the `sample.csv` results are produced.

### Docs
See `reports/dl_deep_learning.md` for the model design, the ML-vs-DL
comparison table, explainability results, and known limitations.


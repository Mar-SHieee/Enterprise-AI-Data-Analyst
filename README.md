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

# Data Dictionary — Online Retail II (Cleaned Dataset)

**Owner:** Mariam — Member 1: Data Engineering & SQL Lead
**Source:** UCI Online Retail II (`Year 2009-2010` + `Year 2010-2011` sheets merged)
**Produced by:** `notebooks/01_eda.ipynb` → `clean_data()` in `src/data/loader.py`

This document describes every column in the cleaned dataset (as saved to
`data/sample.csv` and loaded into the SQLite warehouse in
`notebooks/02_sql_etl.ipynb`), plus the cleaning rules applied to get from
the raw UCI file to this cleaned version.

> **Notes:**
> - Negative `Quantity` values are **kept** (they may represent returns); only rows where `Quantity = 0` are removed during cleaning.
> - `IsCancelled` only **flags** cancellation invoices (those starting with `C`) — `clean_data()` does not remove them.
> - `DayOfWeek` and `Hour` are **EDA-derived** columns added in `01_eda.ipynb` for visual analysis; they are not produced by `clean_data()` itself.

## Columns

| Column | Type | Description |
|---|---|---|
| `Invoice` | string | Invoice/transaction number. Values starting with `C` are cancellations. |
| `StockCode` | string | Product/item code. |
| `Description` | string | Product name/description. |
| `Quantity` | integer | Units purchased. Negative values may represent returns; rows with `Quantity = 0` are removed during cleaning. |
| `InvoiceDate` | datetime | Date and time the transaction was recorded. |
| `Price` | float | Unit price, in GBP. |
| `Customer ID` | integer | Unique customer identifier. Rows with no Customer ID are removed during cleaning. |
| `Country` | string | Country where the customer is registered. |
| `Revenue` | float | Derived column: `Quantity * Price`. |
| `IsCancelled` | boolean | Derived column: flags invoices whose `Invoice` number starts with `C`. Cancellation rows are **not** automatically removed by `clean_data()` — they remain in the cleaned dataset with this flag set. |
| `DayOfWeek` | string | *EDA-derived column* (added in `01_eda.ipynb` for the day-of-week/hour visual analysis, not part of `clean_data()`): name of the weekday the invoice was recorded, from `InvoiceDate`. |
| `Hour` | integer | *EDA-derived column* (added in `01_eda.ipynb` for the day-of-week/hour visual analysis, not part of `clean_data()`): hour of day (0–23) the invoice was recorded, from `InvoiceDate`. |

## Cleaning Rules Applied (`clean_data()`)

Applied in this order to the raw merged dataset:

1. **Remove exact duplicate rows.**
2. **Cast `Invoice` to string** so cancellation prefixes (`C...`) can be checked reliably.
3. **Flag cancellations** — add `IsCancelled` = `True` where `Invoice` starts with `"C"`.
4. **Drop rows with a missing `Customer ID`** — these can't be attributed to a customer and are excluded from customer-level analysis.
5. **Cast `Customer ID` to integer.**
6. **Filter to valid transactions** — keep only rows where `Price > 0` and `Quantity != 0`. Negative quantities are kept (they may represent returns); only `Quantity = 0` rows are removed.
7. **Compute `Revenue`** = `Quantity * Price`.
8. **Parse `InvoiceDate`** to a proper datetime type.
9. Index reset after filtering.

## Validation Checks (`validate_data()`)

Run on the cleaned dataframe before it's loaded into the SQL warehouse:

- All required columns are present (`Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`, `Revenue`, `IsCancelled`).
- No null `Customer ID` values remain.
- `Price` is strictly positive for every row.
- `Quantity` is non-zero for every row.
- `InvoiceDate` is a proper datetime column.

## Summary Stats (from `01_eda.ipynb`)

- Raw dataset: **1,067,371** transactions, Dec 2009 – Dec 2011.
- **22.77%** of rows had no Customer ID (excluded).
- **34,335** exact duplicate rows removed.
- **19,494** transactions flagged as cancellations (`Invoice` starting with `C`; not removed, just flagged via `IsCancelled`).
- After cleaning: **797,815** valid transactions across **5,939** customers, **4,646** products, **44,870** orders.
- Referential integrity verified: **0** orphan rows in `order_items` (see `notebooks/02_sql_etl.ipynb`).

## Mapping to the SQL Warehouse (`sql/schema.sql`)

| SQL Table | Column | Source Column(s) |
|---|---|---|
| `customers` | `customer_id`, `country` | `Customer ID`, `Country` |
| `products` | `stock_code`, `description` | `StockCode`, `Description` |
| `orders` | `invoice_no`, `customer_id`, `invoice_date`, `is_cancelled` | `Invoice`, `Customer ID`, `InvoiceDate`, `IsCancelled` |
| `order_items` | `invoice_no`, `stock_code`, `quantity`, `unit_price`, `revenue` | `Invoice`, `StockCode`, `Quantity`, `Price`, `Revenue` |

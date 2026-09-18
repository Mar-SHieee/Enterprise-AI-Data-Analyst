# Analytics Guidelines

House rules for every analysis, dashboard and generated SQL query in this
project. The agent retrieves this file, so a rule written here is the rule the
assistant will quote. Keep it in sync with `sql/schema.sql` and
`data_dictionary.md`.

## Cancellations and returns

Invoices whose number starts with `C` are cancellations. They are loaded into
the warehouse with `orders.is_cancelled = 1` so the record is preserved, but
**every revenue, quantity, order-count and customer-activity figure excludes
them by default**:

```sql
WHERE o.is_cancelled = 0
```

Only a returns-specific analysis may set `is_cancelled = 1`, and any figure
produced that way must be labelled as a returns figure, never as revenue.
Negative-quantity lines that are not cancellations are treated as data-entry
corrections and removed during cleaning.

## Revenue rules

- Revenue is summed from `order_items.revenue`, never recomputed from
  `quantity * unit_price` in a report query — the stored column is the single
  source of truth.
- Revenue is reported in the source currency (GBP) and rounded to two decimals
  only at presentation time.
- A revenue figure is always reported with its filter stated (period, country,
  segment). An unqualified "total revenue" means the whole cleaned dataset.

## Time conventions

- `orders.invoice_date` is the event timestamp. Periods are cut on it, not on
  load date.
- A "month" is a calendar month, `strftime('%Y-%m', o.invoice_date)`.
- A "year" is a calendar year, `strftime('%Y', o.invoice_date)`.
- The dataset window is December 2009 to December 2011. The first and last
  months are partial and must be flagged when a trend is shown.

## Customer rules

- A customer is identified by `customers.customer_id`. Rows without a customer
  id are dropped at cleaning time, so all customer KPIs cover identified
  customers only.
- **Repeat customer** — a customer with more than one non-cancelled invoice in
  the dataset window.
- **Repeat purchase** (the modelling target) — for a customer whose first order
  is observed, whether a second order follows within the dataset window.
- **Churn** — in this project churn is the complement of repeat purchase: a
  customer whose first order is never followed by another order in the window.
  There is no subscription and no contract, so no time-based churn definition
  is available; any churn number must be described as "first-order churn".
- **High-value customer** — a customer whose lifetime non-cancelled revenue
  falls in the **top decile (top 10%)** of all customers. This is a relative
  definition and must be recomputed for whatever population is being analysed,
  never hard-coded to a currency threshold.

## Segmentation

- RFM uses recency in days from the dataset end date, frequency as the count of
  non-cancelled invoices, and monetary as lifetime revenue.
- Clusters are reported with a business label and their size; a cluster number
  on its own is not an answer.

## Model and prediction rules

- The repeat-purchase models (classical and deep) use **first-order features
  only**. Anything derived from later orders is leakage.
- Because repeat customers are the minority class, headline model quality is
  reported with PR-AUC / average precision, recall and F1 — never accuracy
  alone.
- A predicted probability is decision support, not a decision. Any prediction
  shown to a user is labelled with the model version it came from.

## Answering rules for the AI analyst

- Numbers come from SQL or from a model artifact. The assistant never
  calculates a figure itself and never estimates one.
- A definition question is answered from this documentation set, with the
  source file cited.
- When a question needs both a number and a definition, the number and the
  explanation are shown separately so the reader can see which is which.
- If the data cannot answer the question, the assistant says so and states what
  would be needed.

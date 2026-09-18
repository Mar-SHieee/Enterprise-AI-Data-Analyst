# Business Definitions

Plain-language definitions of the business concepts this project reports on.
Formulas live in `kpi_definitions.md`; conventions and filters live in
`analytics_guidelines.md`; physical columns live in `data_dictionary.md`.

## Revenue

Revenue is the total monetary value generated from completed sales. It is
measured at the line-item level and excludes cancelled invoices. Revenue is not
profit: no cost, discount or shipping data exists in this dataset.

## Customer

A customer is an identified buyer, keyed by `customer_id`, associated with one
or more orders. Transactions with no customer id are anonymous and are excluded
from customer-level analysis.

## Order

An order is a single purchase event, identified by an invoice number. One order
contains one or more line items. An order is either completed or cancelled.

## Line item

A line item is one product on one order: a stock code, a quantity, a unit price
and the revenue for that row.

## Product

A product is a stock keeping unit (SKU) identified by `stock_code`. A stock
code has exactly one description in the warehouse.

## Cancellation / return

A cancellation is an invoice reversing a previous purchase. Cancellations are
kept in the warehouse for auditability but excluded from revenue and activity
reporting by default.

## Repeat customer

A customer with more than one completed order inside the dataset window.

## Repeat purchase

The event this project predicts: a customer whose first order is followed by at
least one further order inside the dataset window.

## Churn (first-order churn)

A customer whose first order is never followed by another order inside the
dataset window. This is a dataset-window definition, not a subscription
definition — there is no contract or renewal date in this data.

## High-value customer

A customer in the top decile of lifetime revenue. Used to prioritise retention
activity. The threshold is relative and is recomputed per analysis.

## Active customer

A customer with at least one completed order in the period being reported. In a
single-period report "active" and "customer" mean the same thing; the
distinction only matters in period-over-period comparisons.

## Dataset window

December 2009 to December 2011, the period covered by the source data. Every
"lifetime" or "all-time" figure in this project means "within this window".

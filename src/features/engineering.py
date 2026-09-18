"""Feature engineering for customer-level classical ML models.

The main task is repeat-purchase prediction: using only the customer's
first observed order to predict whether the customer places another order
within the observed dataset window.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    for c in out.columns:
        key = c.strip().lower().replace("_", " ")
        if key == "customer id":
            rename[c] = "Customer ID"
        elif key == "invoice date":
            rename[c] = "InvoiceDate"
        elif key == "stockcode":
            rename[c] = "StockCode"
        elif key == "invoice":
            rename[c] = "Invoice"
        elif key == "quantity":
            rename[c] = "Quantity"
        elif key == "price":
            rename[c] = "Price"
        elif key == "revenue":
            rename[c] = "Revenue"
    out = out.rename(columns=rename)
    required = {"Customer ID", "Invoice", "InvoiceDate", "Quantity", "Price", "Revenue"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out["InvoiceDate"] = pd.to_datetime(out["InvoiceDate"], errors="coerce", utc=True)
    out["Customer ID"] = pd.to_numeric(out["Customer ID"], errors="coerce")
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out["Revenue"] = pd.to_numeric(out["Revenue"], errors="coerce")
    out = out.dropna(subset=["Customer ID", "Invoice", "InvoiceDate"])
    return out


def build_repeat_purchase_dataset(transactions: pd.DataFrame) -> pd.DataFrame:
    """Create one row per customer using first-order behavior only.

    Target:
        repeat_customer = 1 when the customer has >1 observed invoice,
        otherwise 0.

    Features are calculated exclusively from the customer's first invoice,
    avoiding use of later behavior as predictors.
    """
    df = _clean_columns(transactions)
    df = df.sort_values(["Customer ID", "InvoiceDate", "Invoice"])

    first_invoice = (
        df.sort_values(["Customer ID", "InvoiceDate", "Invoice"])
          .groupby("Customer ID", as_index=False)
          .nth(0)[["Customer ID", "Invoice", "InvoiceDate"]]
          .rename(columns={"Invoice": "FirstInvoice", "InvoiceDate": "FirstInvoiceDate"})
    )
    first_ids = first_invoice[["Customer ID", "FirstInvoice"]].rename(
        columns={"FirstInvoice": "Invoice"}
    )
    first_rows = df.merge(first_ids, on=["Customer ID", "Invoice"], how="inner")

    features = first_rows.groupby("Customer ID").agg(
        first_order_revenue=("Revenue", "sum"),
        first_order_quantity=("Quantity", "sum"),
        first_order_avg_price=("Price", "mean"),
        first_order_unique_products=("StockCode", "nunique") if "StockCode" in first_rows else ("Price", "size"),
        first_order_lines=("Invoice", "size"),
    ).reset_index()

    order_counts = df.groupby("Customer ID")["Invoice"].nunique().rename("observed_order_count")
    customers = features.merge(order_counts, on="Customer ID", how="left")
    customers["repeat_customer"] = (customers["observed_order_count"] > 1).astype(int)

    # Additional robust transformations for skewed retail variables.
    for col in ["first_order_revenue", "first_order_quantity", "first_order_avg_price", "first_order_lines"]:
        customers[f"log1p_{col}"] = np.sign(customers[col]) * np.log1p(np.abs(customers[col]))

    return customers.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_prediction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features and customer IDs for inference."""
    ds = build_repeat_purchase_dataset(transactions)
    return ds.drop(columns=["repeat_customer", "observed_order_count"], errors="ignore")

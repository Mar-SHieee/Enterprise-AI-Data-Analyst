
from __future__ import annotations

import sqlite3
import urllib.request
from pathlib import Path
import pandas as pd


RAW_DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def download_raw_data(dest_dir: str = "data/raw") -> Path:
    """Download and unzip the Online Retail II dataset from UCI."""

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    zip_path = dest / "online_retail_ii.zip"

    if not zip_path.exists():
        urllib.request.urlretrieve(RAW_DATA_URL, zip_path)

    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    xlsx_files = list(dest.glob("*.xlsx"))

    if not xlsx_files:
        raise FileNotFoundError(
            f"No .xlsx file found after extracting {zip_path}"
        )

    return xlsx_files[0]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load and merge both sheets of the Online Retail II workbook."""

    df_2009 = pd.read_excel(
        path,
        sheet_name="Year 2009-2010"
    )

    df_2010 = pd.read_excel(
        path,
        sheet_name="Year 2010-2011"
    )

    df = pd.concat(
        [df_2009, df_2010],
        ignore_index=True
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project's cleaning rules to the raw transactions."""

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert Invoice to string
    df["Invoice"] = df["Invoice"].astype(str)

    # Identify cancelled invoices
    df["IsCancelled"] = df["Invoice"].str.startswith("C")

    # Remove transactions without Customer ID
    df = df.dropna(subset=["Customer ID"])

    # Convert Customer ID to integer
    df["Customer ID"] = df["Customer ID"].astype(int)

    # Keep valid prices and non-zero quantities
    valid_price = df["Price"] > 0
    valid_qty = df["Quantity"] != 0

    df = df[valid_price & valid_qty]

    # Calculate revenue
    df["Revenue"] = df["Quantity"] * df["Price"]

    # Convert date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    return df.reset_index(drop=True)


def validate_data(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the cleaned dataframe."""

    required_cols = {
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
        "Revenue",
        "IsCancelled",
    }

    missing = required_cols - set(df.columns)

    assert not missing, f"Missing expected columns: {missing}"

    assert df["Customer ID"].isnull().sum() == 0, \
        "Found null Customer ID after cleaning"

    assert df["Price"].min() > 0, \
        "Found non-positive Price after cleaning"

    assert df["Quantity"].apply(lambda x: x != 0).all(), \
        "Found zero Quantity rows"

    assert df["InvoiceDate"].dtype.kind == "M", \
        "InvoiceDate must be a datetime column"


def build_sqlite_db(
    df: pd.DataFrame,
    schema_path: str,
    db_path: str = "retail.db"
) -> None:
    """Load a cleaned dataframe into a SQLite database."""

    conn = sqlite3.connect(db_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    # Customers
    customers = (
        df[["Customer ID", "Country"]]
        .drop_duplicates(subset=["Customer ID"])
        .rename(
            columns={
                "Customer ID": "customer_id",
                "Country": "country",
            }
        )
    )

    customers.to_sql(
        "customers",
        conn,
        if_exists="append",
        index=False,
    )

    # Products
    products = (
        df[["StockCode", "Description"]]
        .drop_duplicates(subset=["StockCode"])
        .rename(
            columns={
                "StockCode": "stock_code",
                "Description": "description",
            }
        )
    )

    products.to_sql(
        "products",
        conn,
        if_exists="append",
        index=False,
    )

    # Orders
    orders = (
        df[
            [
                "Invoice",
                "Customer ID",
                "InvoiceDate",
                "IsCancelled",
            ]
        ]
        .drop_duplicates(subset=["Invoice"])
        .rename(
            columns={
                "Invoice": "invoice_no",
                "Customer ID": "customer_id",
                "InvoiceDate": "invoice_date",
                "IsCancelled": "is_cancelled",
            }
        )
    )

    orders["is_cancelled"] = orders["is_cancelled"].astype(int)

    orders.to_sql(
        "orders",
        conn,
        if_exists="append",
        index=False,
    )

    # Order items
    items = (
        df[
            [
                "Invoice",
                "StockCode",
                "Quantity",
                "Price",
                "Revenue",
            ]
        ]
        .rename(
            columns={
                "Invoice": "invoice_no",
                "StockCode": "stock_code",
                "Quantity": "quantity",
                "Price": "unit_price",
                "Revenue": "revenue",
            }
        )
    )

    items.to_sql(
        "order_items",
        conn,
        if_exists="append",
        index=False,
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":

    xlsx_path = download_raw_data()

    raw_df = load_raw_data(str(xlsx_path))

    clean_df = clean_data(raw_df)

    validate_data(clean_df)

    build_sqlite_db(
        clean_df,
        schema_path="sql/schema.sql",
        db_path="retail.db",
    )

    print(
        f"Loaded {len(clean_df):,} "
        f"clean transactions into retail.db"
    )

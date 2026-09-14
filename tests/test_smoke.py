import pandas as pd
from src.data.loader import clean_data, validate_data


def test_sample_data_loads():
    df = pd.read_csv("data/sample.csv")
    assert len(df) > 0
    assert "Invoice" in df.columns


def test_clean_data_columns():
    df = pd.read_csv("data/sample.csv")
    cleaned = clean_data(df)
    expected_cols = {
        "Invoice", "StockCode", "Description", "Quantity",
        "InvoiceDate", "Price", "Customer ID", "Country",
        "Revenue", "IsCancelled",
    }
    assert expected_cols.issubset(set(cleaned.columns))


def test_validate_data_passes():
    df = pd.read_csv("data/sample.csv")
    cleaned = clean_data(df)
    validate_data(cleaned)

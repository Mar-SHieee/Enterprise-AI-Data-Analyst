"""
Spark ETL + feature pipeline (scalable transformation layer).

The original Spark work lived only in a notebook that hard-coded
``C:/temp`` and ``C:/hadoop/bin/winutils.exe``, so it could not run on Linux,
in Docker or in CI. This module is the portable version of the same pipeline:

    raw transactions  ->  cleaned transactions
                      ->  customer_features  (parquet)
                      ->  product_features   (parquet)
                      ->  country_revenue    (parquet)

The cleaning rules are deliberately identical to
``src/data/loader.clean_data`` — same drop-duplicates, same "no customer id"
rule, same price/quantity filters, same revenue formula — so the Spark path and
the pandas path produce the same warehouse. If one changes, the other must.

Run:
    python spark/spark_pipeline.py
    python spark/spark_pipeline.py --input data/sample.csv --output spark/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_spark(app_name: str = "EnterpriseAIDataAnalyst"):
    """Create a local SparkSession with no platform-specific configuration."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        # A temp dir under the repo keeps the job self-contained and avoids the
        # Windows-only path that made the original notebook unrunnable.
        .config("spark.sql.warehouse.dir", str(Path("spark/_warehouse").resolve()))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def read_transactions(spark, input_path: str):
    from pyspark.sql import functions as F

    frame = (
        spark.read.option("header", "true").option("inferSchema", "true")
        .csv(input_path)
    )
    # Normalise the two column names that contain a space or differ by case.
    for source, target in [("Customer ID", "CustomerID"), ("Customer Id", "CustomerID")]:
        if source in frame.columns:
            frame = frame.withColumnRenamed(source, target)
    return frame.withColumn("InvoiceDate", F.to_timestamp("InvoiceDate"))


def clean_transactions(frame):
    """Mirror of src.data.loader.clean_data, expressed in Spark."""
    from pyspark.sql import functions as F

    cleaned = (
        frame.dropDuplicates()
        .withColumn("Invoice", F.col("Invoice").cast("string"))
        .withColumn("IsCancelled", F.col("Invoice").startswith("C"))
        .filter(F.col("CustomerID").isNotNull())
        .withColumn("CustomerID", F.col("CustomerID").cast("int"))
        .filter((F.col("Price") > 0) & (F.col("Quantity") != 0))
    )
    return cleaned.withColumn("Revenue", F.round(F.col("Quantity") * F.col("Price"), 4))


def customer_features(cleaned):
    """One row per customer — the aggregate a feature store would serve."""
    from pyspark.sql import functions as F

    completed = cleaned.filter(~F.col("IsCancelled"))
    return (
        completed.groupBy("CustomerID")
        .agg(
            F.countDistinct("Invoice").alias("Total_Orders"),
            F.count("*").alias("Total_Lines"),
            F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
            F.round(F.avg("Revenue"), 2).alias("Avg_Line_Value"),
            F.round(F.max("Revenue"), 2).alias("Max_Purchase"),
            F.sum("Quantity").alias("Total_Units"),
            F.countDistinct("StockCode").alias("Unique_Products"),
            F.min("InvoiceDate").alias("First_Order_Date"),
            F.max("InvoiceDate").alias("Last_Order_Date"),
        )
        .withColumn("Is_Repeat_Customer", (F.col("Total_Orders") > 1).cast("int"))
        .withColumn(
            "Avg_Order_Value",
            F.round(F.col("Total_Revenue") / F.col("Total_Orders"), 2),
        )
    )


def product_features(cleaned):
    from pyspark.sql import functions as F

    completed = cleaned.filter(~F.col("IsCancelled"))
    return (
        completed.groupBy("StockCode")
        .agg(
            F.first("Description").alias("Description"),
            F.sum("Quantity").alias("Total_Units"),
            F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
            F.round(F.avg("Price"), 2).alias("Avg_Price"),
            F.countDistinct("CustomerID").alias("Unique_Buyers"),
        )
        .orderBy(F.col("Total_Revenue").desc())
    )


def country_revenue(cleaned):
    from pyspark.sql import functions as F

    completed = cleaned.filter(~F.col("IsCancelled"))
    return (
        completed.groupBy("Country")
        .agg(
            F.round(F.sum("Revenue"), 2).alias("Total_Revenue"),
            F.countDistinct("Invoice").alias("Orders"),
            F.countDistinct("CustomerID").alias("Customers"),
        )
        .orderBy(F.col("Total_Revenue").desc())
    )


def write_output(frame, output_dir: Path, name: str, fmt: str = "parquet") -> Path:
    target = output_dir / name
    if target.exists():
        shutil.rmtree(target)
    writer = frame.coalesce(1).write.mode("overwrite")
    if fmt == "csv":
        writer.option("header", "true").csv(str(target))
    else:
        writer.parquet(str(target))
    return target


def run(input_path: str = "data/sample.csv", output_dir: str = "spark") -> dict:
    spark = build_spark()
    try:
        raw = read_transactions(spark, input_path)
        cleaned = clean_transactions(raw).cache()

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        stats = {
            "rows_raw": raw.count(),
            "rows_cleaned": cleaned.count(),
            "customers": cleaned.select("CustomerID").distinct().count(),
            "products": cleaned.select("StockCode").distinct().count(),
        }

        write_output(cleaned, out, "final_processed_data", fmt="csv")
        write_output(customer_features(cleaned), out, "customer_features")
        write_output(product_features(cleaned), out, "product_features")
        write_output(country_revenue(cleaned), out, "country_revenue")

        print("Spark pipeline complete:")
        for key, value in stats.items():
            print(f"  {key}: {value:,}")
        print(f"  outputs written under {out.resolve()}")
        return stats
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample.csv")
    parser.add_argument("--output", default="spark")
    args = parser.parse_args()
    run(args.input, args.output)

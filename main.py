"""
CS611 Assignment 1: medallion data pipeline for loan default prediction.

Run from the project folder:
    python main.py

It builds:
    datamart/bronze/<source>/            raw monthly copies (CSV)
    datamart/silver/<source>/            cleaned, typed tables (parquet)
    datamart/gold/label_store/           30dpd_6mob labels (parquet)
    datamart/gold/feature_store/         application-date features (parquet)
"""

import os
import time
from datetime import datetime

import pyspark
from dateutil.relativedelta import relativedelta

from utils.data_processing_bronze_table import SOURCES, process_bronze_table
from utils.data_processing_gold_table import process_feature_store, process_label_store
from utils.data_processing_silver_table import (
    process_silver_attributes,
    process_silver_clickstream,
    process_silver_financials,
    process_silver_loan_daily,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE = "2024-12-01"   # last month with clickstream and loan data for the labelled window

BRONZE_ROOT = "datamart/bronze"
SILVER_ROOT = "datamart/silver"
GOLD_ROOT = "datamart/gold"

SILVER_STEPS = {
    "lms_loan_daily": process_silver_loan_daily,
    "clickstream": process_silver_clickstream,
    "attributes": process_silver_attributes,
    "financials": process_silver_financials,
}


def generate_first_of_month_dates(start_date_str, end_date_str):
    """All first-of-month dates from start to end, inclusive, as 'YYYY-MM-DD' strings."""
    current = datetime.strptime(start_date_str, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    dates = []
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=1)
    return dates


def main():
    started = time.time()
    spark = (
        pyspark.sql.SparkSession.builder.appName("cs611_assignment1")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    dates = generate_first_of_month_dates(START_DATE, END_DATE)
    print(f"Processing {len(dates)} monthly snapshots: {dates[0]} to {dates[-1]}")

    # Bronze: every source, every month
    print("\n=== BRONZE ===")
    for d in dates:
        for source_name in SOURCES:
            process_bronze_table(source_name, d, BRONZE_ROOT, spark)

    # Silver: clean each source, every month
    print("\n=== SILVER ===")
    for d in dates:
        for source_name, step in SILVER_STEPS.items():
            step(d, BRONZE_ROOT, SILVER_ROOT, spark)

    # Gold: label store and feature store, every month
    print("\n=== GOLD ===")
    for d in dates:
        process_label_store(d, SILVER_ROOT, GOLD_ROOT, spark)
        process_feature_store(d, SILVER_ROOT, GOLD_ROOT, spark)

    # Summary
    labels = spark.read.parquet(os.path.join(GOLD_ROOT, "label_store", "*.parquet"))
    features = spark.read.parquet(os.path.join(GOLD_ROOT, "feature_store", "*.parquet"))
    print("\n=== DONE ===")
    print(f"label_store:   {labels.count()} loans, {labels.filter('label = 1').count()} defaults")
    print(f"feature_store: {features.count()} customers, {len(features.columns) - 2} features")
    print(f"time taken: {time.time() - started:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()

"""
Bronze layer: land the raw data, one file per source per month, without changing it.

Why keep bronze raw?
- It is our copy of the source exactly as it arrived, so if a cleaning rule in
  silver turns out to be wrong we can fix the rule and rebuild from bronze
  without asking the source system again.
- Every column is read as text (inferSchema=False) so that dirty values such as
  "52312.68_" or "_______" are kept exactly as they were, not silently turned
  into nulls by Spark's type guessing.
"""

import os

import pyspark.sql.functions as F


# Each source table in data/ and the name we use for it in the datamart
SOURCES = {
    "lms_loan_daily": "data/lms_loan_daily.csv",
    "clickstream": "data/feature_clickstream.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
}


def process_bronze_table(source_name, snapshot_date_str, bronze_root, spark):
    """Copy the rows of one source for one snapshot month into bronze as a CSV."""
    csv_path = SOURCES[source_name]

    # read everything as strings so nothing is lost or reinterpreted
    df = spark.read.csv(csv_path, header=True, inferSchema=False)

    # keep only this month's snapshot (this mimics a monthly batch extract)
    df = df.filter(F.col("snapshot_date") == snapshot_date_str)
    row_count = df.count()

    out_dir = os.path.join(bronze_root, source_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir, f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    )

    # small monthly files, so pandas gives us one clean CSV instead of Spark part files
    df.toPandas().to_csv(out_file, index=False)
    print(f"  bronze {source_name:<15} {snapshot_date_str}  rows={row_count:>6}  -> {out_file}")
    return df

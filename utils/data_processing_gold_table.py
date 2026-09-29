"""
Gold layer: the two tables a model actually uses.

1. label_store   one row per loan, taken at month-on-book 6.
                 label = 1 if the loan is 30 or more days past due at MOB 6.
2. feature_store one row per customer, taken at the APPLICATION month
                 (the month the loan starts), using only data known by then.

How they join (for training):
    feature_store.Customer_ID   = label_store.Customer_ID
    feature_store.snapshot_date = label_store.loan_start_date

Why the application month?
The model is meant to score a customer when they apply. Clickstream data keeps
arriving for months after the loan starts, and those later months may already
reflect repayment trouble. Using them would be temporal leakage: the model
would look great in testing and fail in real use, where that future data does
not exist yet. So every feature is "as of" the application month or earlier.
"""

import os
from datetime import datetime

import pyspark.sql.functions as F
from dateutil.relativedelta import relativedelta
from pyspark.sql.types import FloatType, IntegerType


def _date_tag(snapshot_date_str):
    return snapshot_date_str.replace("-", "_")


def _silver_path(silver_root, source_name, snapshot_date_str):
    return os.path.join(
        silver_root, source_name, f"silver_{source_name}_{_date_tag(snapshot_date_str)}.parquet"
    )


def _write_gold(df, table_name, snapshot_date_str, gold_root):
    out_dir = os.path.join(gold_root, table_name)
    os.makedirs(out_dir, exist_ok=True)
    if df.count() == 0:
        # e.g. the first 6 months have no loan at MOB 6 yet; an empty parquet
        # folder would break anyone reading the whole table, so skip it
        print(f"  gold   {table_name:<15} {snapshot_date_str}  rows=     0  (nothing to write)")
        return df
    out_path = os.path.join(out_dir, f"gold_{table_name}_{_date_tag(snapshot_date_str)}.parquet")
    df.write.mode("overwrite").parquet(out_path)
    print(f"  gold   {table_name:<15} {snapshot_date_str}  rows={df.count():>6}  -> {out_path}")
    return df


# --------------------------------------------------------------------------
# Label store
# --------------------------------------------------------------------------
def process_label_store(snapshot_date_str, silver_root, gold_root, spark, dpd=30, mob=6):
    df = spark.read.parquet(_silver_path(silver_root, "lms_loan_daily", snapshot_date_str))

    # only loans that are exactly 6 months into their life this month
    df = df.filter(F.col("mob") == mob)
    df = df.withColumn("label", F.when(F.col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob"))

    df = df.select("loan_id", "Customer_ID", "loan_start_date", "label", "label_def", "snapshot_date")
    return _write_gold(df, "label_store", snapshot_date_str, gold_root)


# --------------------------------------------------------------------------
# Feature store
# --------------------------------------------------------------------------
OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician",
    "Scientist", "Teacher", "Writer",
]
CLICK_COLS = [f"fe_{i}" for i in range(1, 21)]
CLICK_WINDOW_MONTHS = 3  # application month plus the 2 months before it


def _safe_divide(numerator, denominator):
    return F.when(F.col(denominator) > 0, F.col(numerator) / F.col(denominator)).cast(FloatType())


def process_feature_store(snapshot_date_str, silver_root, gold_root, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # --- customers who applied this month: attributes + financials -----------
    attributes = spark.read.parquet(_silver_path(silver_root, "attributes", snapshot_date_str))
    financials = spark.read.parquet(_silver_path(silver_root, "financials", snapshot_date_str))
    df = attributes.join(financials.drop("snapshot_date"), on="Customer_ID", how="inner")

    # --- engineered financial features -------------------------------------
    df = df.withColumn("debt_to_income", _safe_divide("Outstanding_Debt", "Annual_Income"))
    df = df.withColumn("emi_to_salary", _safe_divide("Total_EMI_per_month", "Monthly_Inhand_Salary"))
    df = df.withColumn(
        "delayed_payments_per_loan",
        F.when(F.col("Num_of_Loan") > 0, F.col("Num_of_Delayed_Payment") / F.col("Num_of_Loan")).cast(FloatType()),
    )

    # --- categories to numbers, with a fixed list so every month has the same columns
    df = df.withColumn(
        "credit_mix_score",
        F.when(F.col("Credit_Mix") == "Bad", 0).when(F.col("Credit_Mix") == "Standard", 1)
        .when(F.col("Credit_Mix") == "Good", 2).cast(IntegerType()),
    )
    df = df.withColumn(
        "pays_min_amount_only",
        F.when(F.col("Payment_of_Min_Amount") == "Yes", 1).when(F.col("Payment_of_Min_Amount") == "No", 0)
        .cast(IntegerType()),
    )
    # "High_spent_Small_value_payments" -> spend level 1, payment size 0
    spend = F.regexp_extract("Payment_Behaviour", r"^(High|Low)_spent", 1)
    size = F.regexp_extract("Payment_Behaviour", r"_spent_(Small|Medium|Large)_", 1)
    df = df.withColumn("spend_level_high", F.when(spend != "", (spend == "High").cast(IntegerType())))
    df = df.withColumn(
        "payment_size_level",
        F.when(size == "Small", 0).when(size == "Medium", 1).when(size == "Large", 2).cast(IntegerType()),
    )
    for occupation in OCCUPATIONS:
        df = df.withColumn(
            f"occ_{occupation.lower()}", (F.col("Occupation") == occupation).cast(IntegerType())
        )
    # the one-hot columns above are NULL when occupation is unknown; make that explicit
    for occupation in OCCUPATIONS:
        c = f"occ_{occupation.lower()}"
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
    df = df.withColumn("occupation_missing", F.col("Occupation").isNull().cast(IntegerType()))

    # --- clickstream, using ONLY months up to and including the application month
    window_months = [
        (snapshot_date - relativedelta(months=k)).strftime("%Y-%m-%d") for k in range(CLICK_WINDOW_MONTHS)
    ]
    paths = [_silver_path(silver_root, "clickstream", d) for d in window_months]
    paths = [p for p in paths if os.path.exists(p)]
    if paths:
        click = spark.read.parquet(*paths)
        click_now = click.filter(F.col("snapshot_date") == F.lit(snapshot_date_str).cast("date")).select(
            "Customer_ID", *[F.col(c).alias(f"click_{c}") for c in CLICK_COLS]
        )
        click_avg = click.groupBy("Customer_ID").agg(
            *[F.avg(c).cast(FloatType()).alias(f"click_{c}_avg{CLICK_WINDOW_MONTHS}m") for c in CLICK_COLS]
        )
        df = df.join(click_now, on="Customer_ID", how="left").join(click_avg, on="Customer_ID", how="left")
    else:
        for c in CLICK_COLS:
            df = df.withColumn(f"click_{c}", F.lit(None).cast(IntegerType()))
            df = df.withColumn(f"click_{c}_avg{CLICK_WINDOW_MONTHS}m", F.lit(None).cast(FloatType()))
    df = df.withColumn("clickstream_missing", F.col("click_fe_1").isNull().cast(IntegerType()))

    # --- tidy: drop the text columns now encoded as numbers ---------------
    df = df.drop("Occupation", "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour")
    key_cols = ["Customer_ID", "snapshot_date"]
    df = df.select(*key_cols, *[c for c in df.columns if c not in key_cols])

    return _write_gold(df, "feature_store", snapshot_date_str, gold_root)

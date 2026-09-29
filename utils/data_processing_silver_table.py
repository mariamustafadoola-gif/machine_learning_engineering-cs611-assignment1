"""
Silver layer: clean, typed, validated data. Still one table per source per month.

The rules here are the "design decisions" of the pipeline. Each one is written
as a small, named step so it can be explained and changed on its own.

General policy
- Wrong or impossible values become NULL. We do NOT fill them in here, because
  any fill value (a median, say) must be learned from the training period only.
  Filling here with statistics from all months would leak future information
  into the past. Imputation is left to the model training step.
- Personal data (Name, SSN) is dropped. It has no predictive use and should not
  travel further down the pipeline.
"""

import os

import pyspark.sql.functions as F
from pyspark.sql.types import DateType, FloatType, IntegerType, StringType


def _date_tag(snapshot_date_str):
    return snapshot_date_str.replace("-", "_")


def _read_bronze(source_name, snapshot_date_str, bronze_root, spark):
    path = os.path.join(
        bronze_root, source_name, f"bronze_{source_name}_{_date_tag(snapshot_date_str)}.csv"
    )
    # read as text again, the casting happens below on purpose
    return spark.read.csv(path, header=True, inferSchema=False)


def _write_silver(df, source_name, snapshot_date_str, silver_root):
    out_dir = os.path.join(silver_root, source_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"silver_{source_name}_{_date_tag(snapshot_date_str)}.parquet"
    )
    df.write.mode("overwrite").parquet(out_path)
    print(f"  silver {source_name:<15} {snapshot_date_str}  rows={df.count():>6}  -> {out_path}")
    return df


def _to_number(col_name, dtype=FloatType()):
    """Strip stray underscores (e.g. '52312.68_') and cast. Anything still not a number becomes NULL."""
    cleaned = F.regexp_replace(F.col(col_name), "_", "")
    return F.when(cleaned.rlike(r"^-?\d+(\.\d+)?$"), cleaned).cast(dtype)


def _null_outside(col_name, low, high):
    """Keep a value only if it lies in a plausible range, else NULL."""
    c = F.col(col_name)
    return F.when((c >= low) & (c <= high), c)


# --------------------------------------------------------------------------
# 1. Loan management system (the source of the label)
# --------------------------------------------------------------------------
def process_silver_loan_daily(snapshot_date_str, bronze_root, silver_root, spark):
    df = _read_bronze("lms_loan_daily", snapshot_date_str, bronze_root, spark)

    column_types = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }
    for col_name, dtype in column_types.items():
        df = df.withColumn(col_name, F.col(col_name).cast(dtype))

    # month on book: how many instalments into the loan we are
    df = df.withColumn("mob", F.col("installment_num").cast(IntegerType()))

    # days past due, same logic as Lab 2
    df = df.withColumn(
        "installments_missed", F.ceil(F.col("overdue_amt") / F.col("due_amt")).cast(IntegerType())
    ).fillna(0, subset=["installments_missed"])
    df = df.withColumn(
        "first_missed_date",
        F.when(
            F.col("installments_missed") > 0,
            F.add_months(F.col("snapshot_date"), -1 * F.col("installments_missed")),
        ).cast(DateType()),
    )
    df = df.withColumn(
        "dpd",
        F.when(
            F.col("overdue_amt") > 0.0, F.datediff(F.col("snapshot_date"), F.col("first_missed_date"))
        ).otherwise(0).cast(IntegerType()),
    )
    return _write_silver(df, "lms_loan_daily", snapshot_date_str, silver_root)


# --------------------------------------------------------------------------
# 2. Clickstream: 20 anonymous numeric features per customer per month
# --------------------------------------------------------------------------
def process_silver_clickstream(snapshot_date_str, bronze_root, silver_root, spark):
    df = _read_bronze("clickstream", snapshot_date_str, bronze_root, spark)
    for i in range(1, 21):
        df = df.withColumn(f"fe_{i}", F.col(f"fe_{i}").cast(IntegerType()))
    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
    # one row per customer per month is expected; guard against duplicates
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    return _write_silver(df, "clickstream", snapshot_date_str, silver_root)


# --------------------------------------------------------------------------
# 3. Customer attributes
# --------------------------------------------------------------------------
AGE_RANGE = (14, 100)  # valid ages in the data run 14 to 56; values like 8678 or -500 are errors


def process_silver_attributes(snapshot_date_str, bronze_root, silver_root, spark):
    df = _read_bronze("attributes", snapshot_date_str, bronze_root, spark)

    # PII: dropped here, never used as a feature
    df = df.drop("Name", "SSN")

    # Age: "45_" -> 45, then impossible ages -> NULL
    df = df.withColumn("Age", _to_number("Age", IntegerType()))
    df = df.withColumn("Age", _null_outside("Age", *AGE_RANGE))

    # Occupation: the placeholder "_______" means unknown
    df = df.withColumn(
        "Occupation", F.when(F.col("Occupation").rlike(r"^_+$"), None).otherwise(F.col("Occupation"))
    )

    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    return _write_silver(df, "attributes", snapshot_date_str, silver_root)


# --------------------------------------------------------------------------
# 4. Customer financials
# --------------------------------------------------------------------------
# Plausible ranges. Values outside them are data errors (e.g. 1,756 bank accounts
# or a 5,789% interest rate) and become NULL. The ranges sit just above the 97th
# percentile of the data, where the genuine values end and the junk begins.
RANGE_RULES = {
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 50),
    "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 50),
    "Num_Credit_Inquiries": (0, 50),
    "Amount_invested_monthly": (0, 9999),  # 10000 is a placeholder value, not real
    "Monthly_Balance": (-1e6, 1e6),  # catches the -3.3e26 placeholder
}

NUMERIC_COLS = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly",
    "Monthly_Balance",
]

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan",
    "Mortgage Loan", "Not Specified", "Payday Loan", "Personal Loan", "Student Loan",
]

VALID_CREDIT_MIX = ["Bad", "Standard", "Good"]
VALID_MIN_PAYMENT = ["Yes", "No"]


def process_silver_financials(snapshot_date_str, bronze_root, silver_root, spark):
    df = _read_bronze("financials", snapshot_date_str, bronze_root, spark)

    # 4a. text to numbers, stripping trailing/leading underscores
    for col_name in NUMERIC_COLS:
        df = df.withColumn(col_name, _to_number(col_name))

    # 4b. range checks
    for col_name, (low, high) in RANGE_RULES.items():
        df = df.withColumn(col_name, _null_outside(col_name, low, high))

    # 4c. cross-field checks
    # yearly income cannot sensibly be more than twice 12 months of take-home pay
    df = df.withColumn(
        "Annual_Income",
        F.when(F.col("Annual_Income") <= 2 * 12 * F.col("Monthly_Inhand_Salary"), F.col("Annual_Income")),
    )
    # monthly loan repayments above monthly take-home pay are data errors
    df = df.withColumn(
        "Total_EMI_per_month",
        F.when(F.col("Total_EMI_per_month") <= F.col("Monthly_Inhand_Salary"), F.col("Total_EMI_per_month")),
    )

    # 4d. categories: placeholders become NULL
    df = df.withColumn(
        "Credit_Mix", F.when(F.col("Credit_Mix").isin(VALID_CREDIT_MIX), F.col("Credit_Mix"))
    )
    df = df.withColumn(
        "Payment_of_Min_Amount",
        F.when(F.col("Payment_of_Min_Amount").isin(VALID_MIN_PAYMENT), F.col("Payment_of_Min_Amount")),
    )  # "NM" (not mentioned) -> NULL
    df = df.withColumn(
        "Payment_Behaviour",
        F.when(
            F.col("Payment_Behaviour").rlike(r"^(High|Low)_spent_(Small|Medium|Large)_value_payments$"),
            F.col("Payment_Behaviour"),
        ),
    )  # "!@9#%8" -> NULL

    # 4e. "10 Years and 9 Months" -> 129 months
    years = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Years?", 1)
    months = F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Months?", 1)
    df = df.withColumn(
        "Credit_History_Months",
        F.when(years != "", years.cast(IntegerType()) * 12 + months.cast(IntegerType())),
    ).drop("Credit_History_Age")

    # 4f. Type_of_Loan is a list inside one text cell. Turn it into one yes/no flag per
    # loan type, plus a count. NULL means the customer has no loans listed.
    loan_list = F.split(F.regexp_replace(F.col("Type_of_Loan"), r",\s*and\s+", ", "), r",\s*")
    df = df.withColumn("loan_list", F.when(F.col("Type_of_Loan").isNotNull(), loan_list))
    for loan_type in LOAN_TYPES:
        flag = "has_" + loan_type.lower().replace(" ", "_").replace("-", "_")
        df = df.withColumn(
            flag, F.coalesce(F.array_contains(F.col("loan_list"), loan_type), F.lit(False)).cast(IntegerType())
        )
    # note: Spark's size() of a NULL list is -1, so handle "no list" explicitly
    df = df.withColumn(
        "Num_Loan_Types_Listed",
        F.when(F.col("loan_list").isNotNull(), F.size("loan_list")).otherwise(0).cast(IntegerType()),
    )
    df = df.drop("loan_list", "Type_of_Loan")

    # Num_of_Loan always equals the number of loans listed in Type_of_Loan when both
    # are valid (checked in EDA), so a corrupted Num_of_Loan (e.g. -100 or 1495)
    # can be recovered from the list instead of being left empty
    df = df.withColumn("Num_of_Loan", F.coalesce(F.col("Num_of_Loan"), F.col("Num_Loan_Types_Listed")))

    # whole-number columns stored as integers
    for col_name in ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan",
                     "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]:
        df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))

    df = df.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))
    df = df.dropDuplicates(["Customer_ID", "snapshot_date"])
    return _write_silver(df, "financials", snapshot_date_str, silver_root)

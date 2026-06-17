from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TRAIN_PATH = RAW_DIR / "train_v2.csv"
MEMBERS_PATH = RAW_DIR / "members_v3.csv"
TRANSACTIONS_PATH = RAW_DIR / "transactions.csv"
OUTPUT_PATH = PROCESSED_DIR / "kkbox_modeling_data_v1.csv"

CUTOFF_DATE = pd.Timestamp("2017-02-28")


def build_member_features(members: pd.DataFrame) -> pd.DataFrame:
    member_features = members.copy()

    member_features["has_member_data"] = 1
    member_features["age_clean"] = member_features["bd"].where(
        member_features["bd"].between(13, 80),
        np.nan,
    )
    member_features["age_missing_or_invalid"] = (
        member_features["age_clean"].isna().astype(int)
    )
    member_features["gender"] = member_features["gender"].fillna("unknown")

    member_features["registration_init_time_parsed"] = pd.to_datetime(
        member_features["registration_init_time"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    member_features["registration_after_cutoff_flag"] = (
        member_features["registration_init_time_parsed"] > CUTOFF_DATE
    ).astype(int)
    member_features["registration_tenure_days"] = (
        CUTOFF_DATE - member_features["registration_init_time_parsed"]
    ).dt.days
    member_features.loc[
        member_features["registration_after_cutoff_flag"] == 1,
        "registration_tenure_days",
    ] = np.nan

    return member_features[
        [
            "msno",
            "has_member_data",
            "age_clean",
            "age_missing_or_invalid",
            "gender",
            "city",
            "registered_via",
            "registration_tenure_days",
            "registration_after_cutoff_flag",
        ]
    ]


def attach_member_features(
    train: pd.DataFrame,
    member_features: pd.DataFrame,
) -> pd.DataFrame:
    data = train.merge(member_features, on="msno", how="left")

    data["has_member_data"] = data["has_member_data"].fillna(0).astype(int)
    data["gender"] = data["gender"].fillna("no_member_data")
    data["city"] = data["city"].fillna(-999).astype(int)
    data["registered_via"] = data["registered_via"].fillna(-999).astype(int)
    data["age_missing_or_invalid"] = (
        data["age_missing_or_invalid"].fillna(1).astype(int)
    )
    data["registration_after_cutoff_flag"] = (
        data["registration_after_cutoff_flag"].fillna(0).astype(int)
    )

    return data


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.copy()

    transactions["transaction_date_parsed"] = pd.to_datetime(
        transactions["transaction_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    transactions["membership_expire_date_parsed"] = pd.to_datetime(
        transactions["membership_expire_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
    transactions["discount_amount"] = (
        transactions["plan_list_price"] - transactions["actual_amount_paid"]
    )
    transactions["is_zero_payment_or_plan"] = (
        (transactions["payment_plan_days"] == 0)
        | (transactions["plan_list_price"] == 0)
        | (transactions["actual_amount_paid"] == 0)
    ).astype(int)

    return transactions


def build_transaction_summary_features(transactions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        transactions.groupby("msno")
        .agg(
            transaction_count=("msno", "size"),
            cancel_count=("is_cancel", "sum"),
            auto_renew_count=("is_auto_renew", "sum"),
            zero_payment_or_plan_count=("is_zero_payment_or_plan", "sum"),
            unique_payment_method_count=("payment_method_id", "nunique"),
            total_actual_amount_paid=("actual_amount_paid", "sum"),
            mean_actual_amount_paid=("actual_amount_paid", "mean"),
            max_actual_amount_paid=("actual_amount_paid", "max"),
            mean_plan_list_price=("plan_list_price", "mean"),
            mean_payment_plan_days=("payment_plan_days", "mean"),
            discount_transaction_count=("discount_amount", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    summary["manual_renew_count"] = (
        summary["transaction_count"] - summary["auto_renew_count"]
    )

    return summary


def build_latest_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions_sorted = transactions.sort_values(
        ["msno", "transaction_date_parsed", "membership_expire_date_parsed"]
    )
    latest_transactions = transactions_sorted.groupby("msno").tail(1).copy()

    latest_features = latest_transactions[
        [
            "msno",
            "payment_method_id",
            "payment_plan_days",
            "plan_list_price",
            "actual_amount_paid",
            "is_auto_renew",
            "is_cancel",
            "transaction_date_parsed",
            "membership_expire_date_parsed",
            "discount_amount",
        ]
    ].copy()

    latest_features = latest_features.rename(
        columns={
            "payment_method_id": "latest_payment_method_id",
            "payment_plan_days": "latest_payment_plan_days",
            "plan_list_price": "latest_plan_list_price",
            "actual_amount_paid": "latest_actual_amount_paid",
            "is_auto_renew": "latest_is_auto_renew",
            "is_cancel": "latest_is_cancel",
            "transaction_date_parsed": "latest_transaction_date",
            "membership_expire_date_parsed": "latest_membership_expire_date",
            "discount_amount": "latest_discount_amount",
        }
    )

    latest_features["latest_transaction_days_before_cutoff"] = (
        CUTOFF_DATE - latest_features["latest_transaction_date"]
    ).dt.days
    latest_features["latest_membership_expire_days_after_cutoff"] = (
        latest_features["latest_membership_expire_date"] - CUTOFF_DATE
    ).dt.days

    return latest_features.drop(
        columns=["latest_transaction_date", "latest_membership_expire_date"]
    )


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = prepare_transactions(transactions)
    summary_features = build_transaction_summary_features(transactions)
    latest_features = build_latest_transaction_features(transactions)

    return summary_features.merge(latest_features, on="msno", how="left")


def attach_transaction_features(
    data: pd.DataFrame,
    transaction_features: pd.DataFrame,
) -> pd.DataFrame:
    data = data.merge(transaction_features, on="msno", how="left")

    count_or_summary_cols = [
        "transaction_count",
        "cancel_count",
        "auto_renew_count",
        "zero_payment_or_plan_count",
        "unique_payment_method_count",
        "total_actual_amount_paid",
        "mean_actual_amount_paid",
        "max_actual_amount_paid",
        "mean_plan_list_price",
        "mean_payment_plan_days",
        "discount_transaction_count",
        "manual_renew_count",
    ]
    latest_numeric_cols = [
        "latest_payment_plan_days",
        "latest_plan_list_price",
        "latest_actual_amount_paid",
        "latest_discount_amount",
        "latest_transaction_days_before_cutoff",
        "latest_membership_expire_days_after_cutoff",
    ]
    latest_category_cols = [
        "latest_payment_method_id",
        "latest_is_auto_renew",
        "latest_is_cancel",
    ]

    data["has_transaction_data"] = data["transaction_count"].notna().astype(int)
    data[count_or_summary_cols] = data[count_or_summary_cols].fillna(0)
    data[latest_numeric_cols] = data[latest_numeric_cols].fillna(-999)
    data[latest_category_cols] = data[latest_category_cols].fillna(-999).astype(int)

    return data


def build_features() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train labels...")
    train = pd.read_csv(TRAIN_PATH)

    print("Loading member data...")
    members = pd.read_csv(MEMBERS_PATH)
    member_features = build_member_features(members)
    data = attach_member_features(train, member_features)

    print("Loading transaction data...")
    transaction_cols = [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ]
    transactions = pd.read_csv(TRANSACTIONS_PATH, usecols=transaction_cols)
    transaction_features = build_transaction_features(transactions)
    data = attach_transaction_features(data, transaction_features)

    return data


def main() -> None:
    data = build_features()
    data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Duplicate users: {data['msno'].duplicated().sum()}")
    print("Target rate:")
    print(data["is_churn"].value_counts(normalize=True).round(4))


if __name__ == "__main__":
    main()

"""Data loading and feature engineering for propensity scoring."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "marketing_campaign.csv"
)


def load_data(path=None):
    """Load the marketing campaign CSV."""
    path = path or DATA_PATH
    df = pd.read_csv(path)
    return df


def engineer_features(df):
    """Create derived features for propensity modeling.

    Returns the dataframe with new columns added.
    """
    df = df.copy()

    # Revenue per tenure month
    df["revenue_per_tenure"] = (
        df["monthly_spend"] / df["tenure_months"].clip(lower=1)
    ).round(4)

    # Usage intensity: normalized composite of data, calls, sms
    data_norm = (df["data_usage_gb"] - df["data_usage_gb"].mean()) / df["data_usage_gb"].std()
    calls_norm = (df["call_minutes"] - df["call_minutes"].mean()) / df["call_minutes"].std()
    sms_norm = (df["sms_count"] - df["sms_count"].mean()) / df["sms_count"].std()
    df["usage_intensity"] = ((data_norm + calls_norm + sms_norm) / 3).round(4)

    # Service count
    df["service_count"] = (
        df["has_streaming"] + df["has_international"] + df["has_device_insurance"]
    )

    # Upsell headroom: how many plan levels can they move up
    plan_level = df["current_plan"].map({"Basic": 0, "Standard": 1, "Premium": 2})
    df["upsell_headroom"] = 2 - plan_level

    # Plan encoded
    df["plan_encoded"] = plan_level

    # Channel encoded
    channel_map = {"Email": 0, "SMS": 1, "App notification": 2, "Direct mail": 3}
    df["channel_encoded"] = df["channel_preference"].map(channel_map)

    # Interaction features
    df["income_x_tenure"] = (df["income"] * df["tenure_months"] / 1e6).round(4)
    df["data_x_streaming"] = df["data_usage_gb"] * df["has_streaming"]

    return df


def get_feature_columns():
    """Return list of feature column names for modeling."""
    return [
        "age", "income", "tenure_months", "monthly_spend",
        "data_usage_gb", "call_minutes", "sms_count",
        "has_streaming", "has_international", "has_device_insurance",
        "previous_upsell_response", "plan_encoded", "channel_encoded",
        "revenue_per_tenure", "usage_intensity", "service_count",
        "upsell_headroom", "income_x_tenure", "data_x_streaming",
    ]


def prepare_splits(df, target="responded", test_size=0.2, seed=42):
    """Split into train/test with stratification on target.

    Returns X_train, X_test, y_train, y_test.
    """
    features = get_feature_columns()
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def prepare_full_pipeline(path=None, test_size=0.2, seed=42):
    """End-to-end: load -> engineer -> split."""
    df = load_data(path)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = prepare_splits(df, test_size=test_size, seed=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "df": df,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": get_feature_columns(),
    }


if __name__ == "__main__":
    result = prepare_full_pipeline()
    print(f"Training set: {result['X_train'].shape}")
    print(f"Test set: {result['X_test'].shape}")
    print(f"Response rate (train): {result['y_train'].mean():.3f}")
    print(f"Response rate (test): {result['y_test'].mean():.3f}")

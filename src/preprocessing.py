"""Preprocessing module: cleaning, encoding, and scaling."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: handle missing values, fix types."""
    df = df.copy()

    # Convert total_charges to numeric (handle blanks)
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    # Fill missing total_charges with monthly_charges * tenure
    mask = df["total_charges"].isna()
    df.loc[mask, "total_charges"] = df.loc[mask, "monthly_charges"] * df.loc[mask, "tenure"]

    # Remove duplicates
    df = df.drop_duplicates(subset=["customer_id"])

    # Ensure correct data types
    df["age"] = df["age"].astype(int)
    df["tenure"] = df["tenure"].astype(int)

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode categorical variables. Returns encoded df and encoder mapping."""
    df = df.copy()
    encoders = {}

    categorical_cols = ["gender", "contract_type", "payment_method", "internet_service"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode target
    df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    numerical_cols = ["age", "tenure", "monthly_charges", "total_charges"]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, scaler


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Full preprocessing pipeline: clean, encode, split, scale."""
    # Clean
    df_clean = clean_data(df)

    # Encode
    df_encoded, encoders = encode_features(df_clean)

    # Split features and target
    feature_cols = ["gender", "age", "tenure", "monthly_charges", "total_charges",
                    "contract_type", "payment_method", "internet_service"]
    X = df_encoded[feature_cols]
    y = df_encoded["churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    from data_loader import load_data_from_db

    df = load_data_from_db()
    data = prepare_data(df)
    print(f"Training set: {data['X_train'].shape}")
    print(f"Test set: {data['X_test'].shape}")
    print(f"Churn rate in train: {data['y_train'].mean():.3f}")
    print(f"Churn rate in test: {data['y_test'].mean():.3f}")

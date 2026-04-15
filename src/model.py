"""Model module: training, prediction, hyperparameter tuning, and persistence."""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def train_logistic_regression(X_train, y_train, tune: bool = False) -> LogisticRegression:
    """Train Logistic Regression model with optional hyperparameter tuning."""
    if tune:
        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "max_iter": [1000],
        }
        grid = GridSearchCV(
            LogisticRegression(random_state=42),
            param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}, Best F1: {grid.best_score_:.4f}")
        return grid.best_estimator_

    model = LogisticRegression(C=1, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, tune: bool = False) -> RandomForestClassifier:
    """Train Random Forest model with optional hyperparameter tuning."""
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}, Best F1: {grid.best_score_:.4f}")
        return grid.best_estimator_

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, tune: bool = False):
    """Train XGBoost model with optional hyperparameter tuning."""
    if not XGBOOST_AVAILABLE:
        print("  XGBoost not installed. Skipping.")
        return None

    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.8, 1.0],
        }
        grid = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
            param_grid, cv=5, scoring="f1", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}, Best F1: {grid.best_score_:.4f}")
        return grid.best_estimator_

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric="logloss", use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, tune: bool = False) -> dict:
    """Train all models and return dictionary."""
    models = {}

    print("Training Logistic Regression...")
    models["Logistic Regression"] = train_logistic_regression(X_train, y_train, tune)

    print("Training Random Forest...")
    models["Random Forest"] = train_random_forest(X_train, y_train, tune)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, tune)
    if xgb_model is not None:
        models["XGBoost"] = xgb_model

    return models


def save_model(model, name: str, scaler=None, encoders=None, feature_cols=None):
    """Save model and preprocessing artifacts using pickle."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    if scaler or encoders or feature_cols:
        artifacts = {"scaler": scaler, "encoders": encoders, "feature_cols": feature_cols}
        artifacts_path = os.path.join(MODELS_DIR, "preprocessing_artifacts.pkl")
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)

    return model_path


def load_model(name: str):
    """Load a saved model."""
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def load_artifacts() -> dict:
    """Load preprocessing artifacts."""
    path = os.path.join(MODELS_DIR, "preprocessing_artifacts.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifacts not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_churn(customer_input: dict, model_name: str = "best_model") -> dict:
    """
    Predict churn for a single customer.

    Args:
        customer_input: dict with keys: gender, age, tenure, monthly_charges,
                        total_charges, contract_type, payment_method, internet_service
        model_name: name of saved model to use

    Returns:
        dict with prediction and probability
    """
    model = load_model(model_name)
    artifacts = load_artifacts()

    scaler = artifacts["scaler"]
    encoders = artifacts["encoders"]
    feature_cols = artifacts["feature_cols"]

    # Build dataframe from input
    df = pd.DataFrame([customer_input])

    # Encode categorical features
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Ensure correct column order
    df = df[feature_cols]

    # Scale numerical features
    numerical_cols = ["age", "tenure", "monthly_charges", "total_charges"]
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability[1]), 4),
        "retention_probability": round(float(probability[0]), 4),
    }


def get_feature_importance(model, feature_cols: list) -> pd.DataFrame:
    """Extract feature importance from a trained model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi

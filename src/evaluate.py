"""Evaluation module: metrics, confusion matrix, ROC curve."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(auc(*roc_curve(y_test, y_prob)[:2]), 4),
    }
    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """Evaluate all models and return comparison DataFrame."""
    results = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
    return pd.DataFrame(results).sort_values("f1_score", ascending=False)


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Plot confusion matrices for all models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"])
        ax.set_title(f"{name}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(models: dict, X_test, y_test):
    """Plot ROC curves for all models."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_importance_df: pd.DataFrame, model_name: str = ""):
    """Plot feature importance bar chart."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=feature_importance_df, x="importance", y="feature", palette="viridis")
    plt.title(f"Feature Importance{' - ' + model_name if model_name else ''}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def print_classification_reports(models: dict, X_test, y_test):
    """Print classification reports for all models."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{'='*50}")
        print(f" {name} - Classification Report")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

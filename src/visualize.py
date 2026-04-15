"""Visualization module: EDA plots for churn analysis."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


def setup_style():
    """Set consistent plot style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.dpi"] = 150


def plot_churn_distribution(df: pd.DataFrame):
    """Plot churn distribution (count and percentage)."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Count plot
    sns.countplot(data=df, x="churn", palette=["#2ecc71", "#e74c3c"], ax=axes[0])
    axes[0].set_title("Churn Distribution (Count)")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Count")

    # Pie chart
    churn_counts = df["churn"].value_counts()
    axes[1].pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
                colors=["#2ecc71", "#e74c3c"], startangle=90)
    axes[1].set_title("Churn Distribution (%)")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "churn_distribution.png"), bbox_inches="tight")
    plt.close()


def plot_tenure_vs_churn(df: pd.DataFrame):
    """Plot tenure distribution by churn status."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    sns.histplot(data=df, x="tenure", hue="churn", kde=True,
                 palette=["#2ecc71", "#e74c3c"], ax=axes[0], bins=30)
    axes[0].set_title("Tenure Distribution by Churn")
    axes[0].set_xlabel("Tenure (months)")

    # Box plot
    sns.boxplot(data=df, x="churn", y="tenure",
                palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_title("Tenure by Churn Status")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tenure_vs_churn.png"), bbox_inches="tight")
    plt.close()


def plot_monthly_charges_vs_churn(df: pd.DataFrame):
    """Plot monthly charges by churn status."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    sns.histplot(data=df, x="monthly_charges", hue="churn", kde=True,
                 palette=["#2ecc71", "#e74c3c"], ax=axes[0], bins=30)
    axes[0].set_title("Monthly Charges by Churn")
    axes[0].set_xlabel("Monthly Charges ($)")

    # Violin plot
    sns.violinplot(data=df, x="churn", y="monthly_charges",
                   palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_title("Monthly Charges Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "monthly_charges_vs_churn.png"), bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap for numerical features."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    # Encode for correlation
    df_encoded = df.copy()
    df_encoded["churn_encoded"] = df_encoded["churn"].map({"Yes": 1, "No": 0})
    numerical = df_encoded[["age", "tenure", "monthly_charges", "total_charges", "churn_encoded"]]

    plt.figure(figsize=(8, 6))
    corr = numerical.corr()
    mask = [[i > j for j in range(len(corr))] for i in range(len(corr))]
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f",
                square=True, mask=mask, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()


def plot_contract_type_analysis(df: pd.DataFrame):
    """Plot churn by contract type."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Churn rate by contract
    churn_by_contract = df.groupby("contract_type")["churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    churn_by_contract.columns = ["contract_type", "churn_rate"]

    sns.barplot(data=churn_by_contract, x="contract_type", y="churn_rate",
                palette="rocket", ax=axes[0])
    axes[0].set_title("Churn Rate by Contract Type")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_xlabel("")

    # Count by contract and churn
    sns.countplot(data=df, x="contract_type", hue="churn",
                  palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_title("Customer Count by Contract & Churn")
    axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "contract_type_analysis.png"), bbox_inches="tight")
    plt.close()


def generate_all_plots(df: pd.DataFrame):
    """Generate all visualization plots."""
    print("Generating visualizations...")
    plot_churn_distribution(df)
    print("  - Churn distribution plot saved")
    plot_tenure_vs_churn(df)
    print("  - Tenure vs churn plot saved")
    plot_monthly_charges_vs_churn(df)
    print("  - Monthly charges vs churn plot saved")
    plot_correlation_heatmap(df)
    print("  - Correlation heatmap saved")
    plot_contract_type_analysis(df)
    print("  - Contract type analysis saved")
    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    from data_loader import load_data_from_db
    df = load_data_from_db()
    generate_all_plots(df)

"""Data loader module: generates synthetic data and manages SQLite database."""

import sqlite3
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "customer_churn_db.sqlite")
SQL_DIR = os.path.join(BASE_DIR, "sql")


def generate_synthetic_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer churn dataset."""
    np.random.seed(seed)

    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_customers + 1)]
    genders = np.random.choice(["Male", "Female"], n_customers)
    ages = np.random.randint(18, 75, n_customers)
    tenures = np.random.randint(0, 72, n_customers)
    contract_types = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_customers,
        p=[0.55, 0.25, 0.20],
    )
    payment_methods = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_customers,
        p=[0.35, 0.20, 0.25, 0.20],
    )
    internet_services = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        n_customers,
        p=[0.35, 0.45, 0.20],
    )

    # Monthly charges depend on internet service
    monthly_charges = np.where(
        internet_services == "Fiber optic",
        np.random.uniform(60, 110, n_customers),
        np.where(
            internet_services == "DSL",
            np.random.uniform(30, 70, n_customers),
            np.random.uniform(18, 35, n_customers),
        ),
    )
    monthly_charges = np.round(monthly_charges, 2)
    total_charges = np.round(monthly_charges * tenures + np.random.uniform(0, 50, n_customers), 2)

    # Churn probability based on realistic factors
    churn_prob = np.full(n_customers, 0.15)
    churn_prob += np.where(contract_types == "Month-to-month", 0.25, 0.0)
    churn_prob += np.where(contract_types == "Two year", -0.10, 0.0)
    churn_prob += np.where(tenures < 12, 0.15, 0.0)
    churn_prob += np.where(tenures > 48, -0.10, 0.0)
    churn_prob += np.where(monthly_charges > 80, 0.10, 0.0)
    churn_prob += np.where(payment_methods == "Electronic check", 0.10, 0.0)
    churn_prob += np.where(internet_services == "Fiber optic", 0.08, 0.0)
    churn_prob = np.clip(churn_prob, 0.05, 0.85)

    churn = np.where(np.random.random(n_customers) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "gender": genders,
        "age": ages,
        "tenure": tenures,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "internet_service": internet_services,
        "churn": churn,
    })
    return df


def init_database(df: pd.DataFrame = None) -> str:
    """Initialize SQLite database with schema and data."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Execute schema
    schema_path = os.path.join(SQL_DIR, "schema.sql")
    with open(schema_path, "r") as f:
        cursor.executescript(f.read())

    # Insert data
    if df is not None:
        cursor.execute("DELETE FROM customers")
        df.to_sql("customers", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    return DB_PATH


def load_data_from_db() -> pd.DataFrame:
    """Load customer data from SQLite database."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}. Run init_database() first.")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()
    return df


def run_sql_query(query: str) -> pd.DataFrame:
    """Execute a SQL query and return results as DataFrame."""
    conn = sqlite3.connect(DB_PATH)
    result = pd.read_sql(query, conn)
    conn.close()
    return result


def run_analysis_queries() -> dict:
    """Run all analysis queries and return results."""
    queries_path = os.path.join(SQL_DIR, "queries.sql")
    with open(queries_path, "r") as f:
        content = f.read()

    # Split by comments to get individual queries
    queries = [q.strip() for q in content.split(";") if q.strip() and not q.strip().startswith("--")]

    results = {}
    query_names = [
        "churn_rate", "avg_charges_by_churn", "tenure_vs_churn",
        "contract_type_churn", "payment_method_churn", "internet_service_churn",
        "gender_churn", "high_risk_customers", "avg_tenure_by_contract", "revenue_impact"
    ]

    conn = sqlite3.connect(DB_PATH)
    for i, query in enumerate(queries):
        if i < len(query_names):
            try:
                results[query_names[i]] = pd.read_sql(query, conn)
            except Exception as e:
                results[query_names[i]] = f"Error: {e}"
    conn.close()
    return results


if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['churn'].value_counts()}")

    print("\nInitializing database...")
    db_path = init_database(df)
    print(f"Database created at: {db_path}")

    print("\nRunning analysis queries...")
    results = run_analysis_queries()
    for name, result in results.items():
        print(f"\n--- {name} ---")
        print(result)

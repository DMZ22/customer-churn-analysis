"""
Main pipeline script: end-to-end customer churn analysis.
Generates data, loads to SQL, trains models, evaluates, and saves artifacts.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import generate_synthetic_data, init_database, load_data_from_db, run_analysis_queries
from src.preprocessing import prepare_data
from src.model import train_all_models, save_model, get_feature_importance, predict_churn
from src.evaluate import (
    evaluate_all_models, plot_confusion_matrices,
    plot_roc_curves, plot_feature_importance, print_classification_reports
)
from src.visualize import generate_all_plots


def main(tune_models: bool = False):
    """Run the full churn analysis pipeline."""
    print("=" * 60)
    print(" CUSTOMER CHURN ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Generate and load data
    print("\n[1/6] Generating synthetic dataset...")
    df = generate_synthetic_data(n_customers=5000)
    print(f"  Dataset: {df.shape[0]} customers, {df.shape[1]} features")
    print(f"  Churn rate: {(df['churn'] == 'Yes').mean():.1%}")

    # Step 2: Initialize database
    print("\n[2/6] Setting up SQLite database...")
    db_path = init_database(df)
    print(f"  Database: {db_path}")

    # Run SQL analysis
    print("\n  Running SQL analysis queries...")
    results = run_analysis_queries()
    if "churn_rate" in results and hasattr(results["churn_rate"], "to_string"):
        print(f"\n  Churn Rate:\n{results['churn_rate'].to_string(index=False)}")
    if "contract_type_churn" in results and hasattr(results["contract_type_churn"], "to_string"):
        print(f"\n  Contract Type Churn:\n{results['contract_type_churn'].to_string(index=False)}")

    # Step 3: Generate EDA visualizations
    print("\n[3/6] Generating visualizations...")
    df_from_db = load_data_from_db()
    generate_all_plots(df_from_db)

    # Step 4: Preprocess data
    print("\n[4/6] Preprocessing data...")
    data = prepare_data(df_from_db)
    print(f"  Training set: {data['X_train'].shape}")
    print(f"  Test set: {data['X_test'].shape}")

    # Step 5: Train models
    print(f"\n[5/6] Training models{'  (with hyperparameter tuning)' if tune_models else ''}...")
    models = train_all_models(data["X_train"], data["y_train"], tune=tune_models)

    # Step 6: Evaluate models
    print("\n[6/6] Evaluating models...")
    metrics_df = evaluate_all_models(models, data["X_test"], data["y_test"])
    print(f"\n  Model Comparison:")
    print(metrics_df.to_string(index=False))

    # Classification reports
    print_classification_reports(models, data["X_test"], data["y_test"])

    # Plot evaluation charts
    plot_confusion_matrices(models, data["X_test"], data["y_test"])
    plot_roc_curves(models, data["X_test"], data["y_test"])

    # Feature importance (use best model)
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]
    fi = get_feature_importance(best_model, data["feature_cols"])
    if not fi.empty:
        plot_feature_importance(fi, best_model_name)
        print(f"\n  Feature Importance ({best_model_name}):")
        print(fi.to_string(index=False))

    # Save best model
    print(f"\n  Saving best model: {best_model_name}")
    save_model(best_model, "best_model",
               scaler=data["scaler"],
               encoders=data["encoders"],
               feature_cols=data["feature_cols"])

    # Save all models
    for name, model in models.items():
        save_model(model, name.lower().replace(" ", "_"))

    # Demo prediction
    print("\n" + "=" * 60)
    print(" DEMO: Predict Churn for a Sample Customer")
    print("=" * 60)
    sample_customer = {
        "gender": "Male",
        "age": 28,
        "tenure": 3,
        "monthly_charges": 89.50,
        "total_charges": 268.50,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
    }
    print(f"\n  Customer: {sample_customer}")
    result = predict_churn(sample_customer)
    print(f"  Prediction: {result['prediction']}")
    print(f"  Churn Probability: {result['churn_probability']:.1%}")
    print(f"  Retention Probability: {result['retention_probability']:.1%}")

    # Top reasons for churn
    print("\n" + "=" * 60)
    print(" TOP REASONS FOR CHURN")
    print("=" * 60)
    if not fi.empty:
        for i, row in fi.head(5).iterrows():
            print(f"  {i+1}. {row['feature']}: importance = {row['importance']:.4f}")

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("  - Launch dashboard:  streamlit run app/streamlit_app.py")
    print("  - Launch API:        uvicorn app.api:app --reload")
    print(f"  - Plots saved to:    plots/")
    print(f"  - Models saved to:   models/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Customer Churn Analysis Pipeline")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning (slower)")
    args = parser.parse_args()
    main(tune_models=args.tune)

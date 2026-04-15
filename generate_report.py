"""Generate a comprehensive project report PDF for the Customer Churn Analysis project."""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
    Table, TableStyle, KeepTogether, ListFlowable, ListItem
)

from src.data_loader import load_data_from_db, run_analysis_queries
from src.preprocessing import prepare_data
from src.model import load_model, get_feature_importance, predict_churn
from src.evaluate import evaluate_all_models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
OUTPUT_PDF = os.path.join(BASE_DIR, "Project_Report.pdf")

LIVE_APP_URL = "https://customer-churn-analysis-czhxwqfchgufgxtkftnixe.streamlit.app/"
GITHUB_URL = "https://github.com/DMZ22/customer-churn-analysis"


# ─── Styles ─────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "CustomTitle", parent=styles["Title"],
    fontSize=26, textColor=colors.HexColor("#1a1a2e"),
    spaceAfter=20, alignment=TA_CENTER, fontName="Helvetica-Bold"
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=14, textColor=colors.HexColor("#e74c3c"),
    alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Oblique"
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=18, textColor=colors.HexColor("#1a1a2e"),
    spaceBefore=18, spaceAfter=10, fontName="Helvetica-Bold",
    borderPadding=6, borderColor=colors.HexColor("#e74c3c"),
    borderWidth=0, leftIndent=0
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=14, textColor=colors.HexColor("#c0392b"),
    spaceBefore=12, spaceAfter=6, fontName="Helvetica-Bold"
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=8
)
code_style = ParagraphStyle(
    "Code", parent=styles["Normal"],
    fontSize=9, fontName="Courier", textColor=colors.HexColor("#2c3e50"),
    backColor=colors.HexColor("#ecf0f1"), borderPadding=6, leading=12,
    leftIndent=10, rightIndent=10, spaceAfter=10
)
bullet_style = ParagraphStyle(
    "Bullet", parent=body_style, leftIndent=18, bulletIndent=6
)


def header_footer(canvas, doc):
    """Custom header/footer for every page."""
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(colors.HexColor("#e74c3c"))
    canvas.setLineWidth(2)
    canvas.line(2 * cm, 28 * cm, 19 * cm, 28 * cm)

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#7f8c8d"))
    canvas.drawString(2 * cm, 28.2 * cm, "Customer Churn Analysis – Project Report")
    canvas.drawRightString(19 * cm, 28.2 * cm, datetime.now().strftime("%Y-%m-%d"))

    # Footer
    canvas.line(2 * cm, 1.5 * cm, 19 * cm, 1.5 * cm)
    canvas.drawCentredString(A4[0] / 2, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def styled_table(data, col_widths=None, header_color="#1a1a2e"):
    """Create a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_color)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def add_plot(story, filename, width=6.5 * inch, caption=None):
    """Add a plot image with optional caption."""
    path = os.path.join(PLOTS_DIR, filename)
    if os.path.exists(path):
        img = Image(path, width=width, height=width * 0.6)
        img.hAlign = "CENTER"
        story.append(img)
        if caption:
            caption_style = ParagraphStyle(
                "Caption", parent=body_style, fontSize=9,
                textColor=colors.HexColor("#7f8c8d"),
                alignment=TA_CENTER, spaceAfter=12, fontName="Helvetica-Oblique"
            )
            story.append(Paragraph(f"<i>Figure: {caption}</i>", caption_style))
        story.append(Spacer(1, 0.15 * inch))


def build_report():
    """Build the complete PDF report."""
    print("Loading data and artifacts...")
    df = load_data_from_db()
    data = prepare_data(df)
    sql_results = run_analysis_queries()

    # Re-evaluate all saved models
    models = {
        "Logistic Regression": load_model("logistic_regression"),
        "Random Forest": load_model("random_forest"),
        "XGBoost": load_model("xgboost"),
    }
    metrics_df = evaluate_all_models(models, data["X_test"], data["y_test"])
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]
    feature_importance = get_feature_importance(best_model, data["feature_cols"])

    print("Building PDF...")
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2.5 * cm, bottomMargin=2 * cm,
        title="Customer Churn Analysis – Project Report",
        author="Data Science Team"
    )

    story = []

    # ═══ COVER PAGE ═══════════════════════════════════════════════════════
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("Customer Churn Analysis", title_style))
    story.append(Paragraph("End-to-End Machine Learning Project", subtitle_style))
    story.append(Spacer(1, 0.3 * inch))

    # Decorative bar
    bar = Table([[""]], colWidths=[15 * cm], rowHeights=[0.1 * cm])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e74c3c"))]))
    story.append(bar)
    story.append(Spacer(1, 0.8 * inch))

    # Info table
    info_data = [
        ["Project Title", "Customer Churn Analysis & Prediction"],
        ["Domain", "Telecom / SaaS Analytics"],
        ["Tech Stack", "Python, SQL, scikit-learn, XGBoost, Streamlit, FastAPI"],
        ["Database", "SQLite (customer_churn_db)"],
        ["Dataset Size", f"{len(df):,} customers"],
        ["Churn Rate", f"{(df['churn'] == 'Yes').mean():.1%}"],
        ["Best Model", best_model_name],
        ["Best ROC-AUC", f"{metrics_df.iloc[0]['roc_auc']:.4f}"],
        ["Live Demo", Paragraph(
            f'<link href="{LIVE_APP_URL}" color="#1e88e5"><u>{LIVE_APP_URL}</u></link>',
            ParagraphStyle("link", fontSize=8.5, textColor=colors.HexColor("#1e88e5"),
                           fontName="Helvetica"))],
        ["GitHub Repo", Paragraph(
            f'<link href="{GITHUB_URL}" color="#1e88e5"><u>{GITHUB_URL}</u></link>',
            ParagraphStyle("link", fontSize=9, textColor=colors.HexColor("#1e88e5"),
                           fontName="Helvetica"))],
        ["Report Date", datetime.now().strftime("%B %d, %Y")],
    ]
    info_table = Table(info_data, colWidths=[5 * cm, 11 * cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (1, 0), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(info_table)
    story.append(PageBreak())

    # ═══ 1. EXECUTIVE SUMMARY ═════════════════════════════════════════════
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "This project delivers a complete, production-ready customer churn analysis "
        "pipeline that combines SQL-based analytics with machine learning to predict "
        "which customers are most likely to leave a subscription service. It covers "
        "the full data science lifecycle: synthetic data generation, database design, "
        "exploratory analysis, feature engineering, model training and evaluation, "
        "interactive dashboards, and REST API deployment.", body_style))
    story.append(Paragraph(
        f"The pipeline was trained on <b>{len(df):,} customer records</b> with an "
        f"overall churn rate of <b>{(df['churn'] == 'Yes').mean():.1%}</b>. Three "
        f"models were compared, with <b>{best_model_name}</b> achieving the best "
        f"F1 score of <b>{metrics_df.iloc[0]['f1_score']:.3f}</b> and ROC-AUC of "
        f"<b>{metrics_df.iloc[0]['roc_auc']:.3f}</b>.", body_style))

    story.append(Paragraph("Key Deliverables", h2_style))
    deliverables = [
        "SQLite database with schema, 5,000 records, and 10 analytical queries",
        "Three trained ML models (Logistic Regression, Random Forest, XGBoost)",
        "Eight exploratory and evaluation visualizations",
        "Interactive Streamlit dashboard with 5 pages",
        "FastAPI REST endpoint with Swagger documentation",
        "Docker + docker-compose deployment configuration",
        "Cloud deployment blueprints (Render, Heroku)",
        "Reusable predict_churn() function and saved preprocessing artifacts",
    ]
    for d in deliverables:
        story.append(Paragraph(f"• {d}", bullet_style))
    story.append(Spacer(1, 0.2 * inch))

    # ═══ 2. PROBLEM STATEMENT ═════════════════════════════════════════════
    story.append(Paragraph("2. Problem Statement", h1_style))
    story.append(Paragraph(
        "Customer churn — the rate at which customers stop doing business with a "
        "company — is one of the most critical metrics for subscription-based "
        "businesses. Acquiring a new customer typically costs 5–7× more than "
        "retaining an existing one, so even small reductions in churn translate "
        "directly into significant revenue gains.", body_style))
    story.append(Paragraph(
        "The goal of this project is to build a system that can (1) analyze "
        "historical customer data to identify the key drivers of churn, and "
        "(2) predict in real time which customers are at highest risk of leaving, "
        "so the business can intervene with targeted retention offers before the "
        "customer actually churns.", body_style))

    # ═══ 3. PROJECT ARCHITECTURE ══════════════════════════════════════════
    story.append(Paragraph("3. Project Architecture", h1_style))
    story.append(Paragraph(
        "The project follows a modular architecture that separates data, SQL, "
        "source code, and application layers for clean development and deployment.",
        body_style))
    arch = """customer_churn_project/
├── data/              → SQLite database
├── sql/
│   ├── schema.sql     → Table definitions & indexes
│   └── queries.sql    → 10 analytical queries
├── src/
│   ├── data_loader.py → Synthetic generator + DB operations
│   ├── preprocessing.py → Cleaning, encoding, scaling
│   ├── model.py       → Training, tuning, prediction
│   ├── evaluate.py    → Metrics and evaluation plots
│   └── visualize.py   → EDA visualizations
├── app/
│   ├── streamlit_app.py → Interactive dashboard
│   └── api.py         → FastAPI endpoint
├── models/            → Pickled models + artifacts
├── plots/             → Generated visualizations
├── deploy/            → Render, Heroku configs
├── Dockerfile
├── docker-compose.yml
├── main.py            → End-to-end pipeline
└── requirements.txt"""
    story.append(Paragraph(arch.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style))

    # ═══ 4. DATASET ═══════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("4. Dataset Description", h1_style))
    story.append(Paragraph(
        "A realistic synthetic telecom dataset was generated with probabilistic "
        "churn labels that depend on contract type, tenure, monthly charges, "
        "payment method, and internet service — mirroring real-world patterns.",
        body_style))

    schema_table = [
        ["Column", "Type", "Description"],
        ["customer_id", "TEXT", "Unique customer identifier"],
        ["gender", "TEXT", "Male / Female"],
        ["age", "INTEGER", "Customer age (18–75)"],
        ["tenure", "INTEGER", "Months as a customer (0–72)"],
        ["monthly_charges", "REAL", "Current monthly bill ($)"],
        ["total_charges", "REAL", "Lifetime charges ($)"],
        ["contract_type", "TEXT", "Month-to-month / One year / Two year"],
        ["payment_method", "TEXT", "Electronic check / Mailed check / Bank / Credit"],
        ["internet_service", "TEXT", "DSL / Fiber optic / No"],
        ["churn", "TEXT", "Target variable: Yes / No"],
    ]
    story.append(styled_table(schema_table, col_widths=[3.8 * cm, 2.5 * cm, 9.5 * cm]))
    story.append(Spacer(1, 0.2 * inch))

    # Dataset stats
    story.append(Paragraph("Dataset Statistics", h2_style))
    stats = [
        ["Metric", "Value"],
        ["Total Records", f"{len(df):,}"],
        ["Features", f"{df.shape[1] - 2}"],
        ["Churn Rate", f"{(df['churn'] == 'Yes').mean():.2%}"],
        ["Avg Monthly Charges", f"${df['monthly_charges'].mean():.2f}"],
        ["Avg Tenure", f"{df['tenure'].mean():.1f} months"],
        ["Avg Age", f"{df['age'].mean():.1f} years"],
    ]
    story.append(styled_table(stats, col_widths=[8 * cm, 7 * cm]))

    # ═══ 5. SQL ANALYSIS ══════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("5. SQL-Based Exploratory Analysis", h1_style))
    story.append(Paragraph(
        "Ten analytical SQL queries were written to explore churn patterns "
        "directly in the database. Key findings from these queries are shown below.",
        body_style))

    story.append(Paragraph("5.1 Contract Type Churn Rate", h2_style))
    if "contract_type_churn" in sql_results and hasattr(sql_results["contract_type_churn"], "values"):
        ct = sql_results["contract_type_churn"]
        rows = [["Contract Type", "Total", "Churned", "Churn Rate %"]]
        for _, row in ct.iterrows():
            rows.append([str(row["contract_type"]), str(row["total_customers"]),
                         str(row["churned"]), f"{row['churn_rate']}%"])
        story.append(styled_table(rows, col_widths=[5 * cm, 3 * cm, 3 * cm, 4 * cm]))

    story.append(Paragraph("5.2 Revenue Impact of Churn", h2_style))
    if "revenue_impact" in sql_results and hasattr(sql_results["revenue_impact"], "values"):
        rev = sql_results["revenue_impact"]
        rows = [["Churn", "Customers", "Monthly Revenue", "Avg Lifetime Value"]]
        for _, row in rev.iterrows():
            rows.append([str(row["churn"]), str(row["customers"]),
                         f"${row['total_monthly_revenue']:,.2f}",
                         f"${row['avg_lifetime_value']:,.2f}"])
        story.append(styled_table(rows, col_widths=[3 * cm, 3 * cm, 5 * cm, 5 * cm]))
    story.append(Spacer(1, 0.1 * inch))

    # ═══ 6. VISUALIZATIONS ════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("6. Exploratory Data Visualizations", h1_style))

    story.append(Paragraph("6.1 Churn Distribution", h2_style))
    add_plot(story, "churn_distribution.png", caption="Overall churn class balance")

    story.append(Paragraph("6.2 Tenure vs Churn", h2_style))
    add_plot(story, "tenure_vs_churn.png", caption="Customers with low tenure churn more often")

    story.append(PageBreak())
    story.append(Paragraph("6.3 Monthly Charges vs Churn", h2_style))
    add_plot(story, "monthly_charges_vs_churn.png", caption="Higher monthly charges correlate with churn")

    story.append(Paragraph("6.4 Correlation Heatmap", h2_style))
    add_plot(story, "correlation_heatmap.png", width=5.5 * inch, caption="Numerical feature correlations")

    story.append(PageBreak())
    story.append(Paragraph("6.5 Contract Type Analysis", h2_style))
    add_plot(story, "contract_type_analysis.png", caption="Month-to-month contracts drive most churn")

    # ═══ 7. MACHINE LEARNING ══════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("7. Machine Learning Pipeline", h1_style))

    story.append(Paragraph("7.1 Preprocessing", h2_style))
    prep_steps = [
        "<b>Cleaning:</b> Handle missing values, remove duplicates, enforce data types.",
        "<b>Encoding:</b> LabelEncoder for categorical features (gender, contract, payment, internet).",
        "<b>Scaling:</b> StandardScaler on numerical features (age, tenure, charges).",
        "<b>Split:</b> Stratified 80/20 train/test split to preserve churn ratio.",
    ]
    for s in prep_steps:
        story.append(Paragraph(f"• {s}", bullet_style))

    story.append(Paragraph("7.2 Models Trained", h2_style))
    model_desc = [
        ["Model", "Description", "Use Case"],
        ["Logistic Regression", "Linear probabilistic classifier", "Interpretable baseline"],
        ["Random Forest", "Bagged tree ensemble", "Non-linear patterns"],
        ["XGBoost", "Gradient-boosted trees", "High accuracy"],
    ]
    story.append(styled_table(model_desc, col_widths=[4.5 * cm, 6.5 * cm, 5 * cm]))

    story.append(Paragraph("7.3 Model Performance Comparison", h2_style))
    perf_rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
    for _, row in metrics_df.iterrows():
        perf_rows.append([row["model"], f"{row['accuracy']:.3f}",
                          f"{row['precision']:.3f}", f"{row['recall']:.3f}",
                          f"{row['f1_score']:.3f}", f"{row['roc_auc']:.3f}"])
    story.append(styled_table(perf_rows, col_widths=[4 * cm, 2.2 * cm, 2.4 * cm,
                                                      2 * cm, 2 * cm, 2.4 * cm]))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("7.4 Confusion Matrices", h2_style))
    add_plot(story, "confusion_matrices.png", caption="True vs predicted labels for all three models")

    story.append(PageBreak())
    story.append(Paragraph("7.5 ROC Curves", h2_style))
    add_plot(story, "roc_curves.png", width=5.5 * inch,
             caption="ROC-AUC comparison across all models")

    story.append(Paragraph("7.6 Feature Importance", h2_style))
    add_plot(story, "feature_importance.png", caption=f"Top churn drivers ({best_model_name})")

    # Top reasons table
    if not feature_importance.empty:
        fi_rows = [["Rank", "Feature", "Importance"]]
        for i, row in feature_importance.head(5).iterrows():
            fi_rows.append([str(i + 1), row["feature"], f"{row['importance']:.4f}"])
        story.append(styled_table(fi_rows, col_widths=[2 * cm, 7 * cm, 4 * cm]))

    # ═══ 8. KEY INSIGHTS ══════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("8. Key Business Insights", h1_style))
    insights = [
        "<b>Contract type is the strongest churn predictor.</b> Month-to-month "
        "customers churn at dramatically higher rates than one-year or two-year "
        "contracts. Incentivizing longer contracts is the highest-ROI retention lever.",
        "<b>Low tenure = high churn risk.</b> New customers (< 12 months) are "
        "far more likely to churn. Early-lifecycle engagement programs are critical.",
        "<b>High monthly charges increase churn.</b> Customers paying above $80/month "
        "show elevated churn, especially on fiber-optic plans. Consider loyalty discounts.",
        "<b>Payment method matters.</b> Electronic check users churn more than "
        "those on automatic bank transfer or credit card — auto-pay reduces friction.",
        "<b>Internet service type is a factor.</b> Fiber optic customers churn "
        "more often than DSL or no-internet customers, suggesting quality or "
        "pricing issues with the premium tier.",
    ]
    for ins in insights:
        story.append(Paragraph(f"• {ins}", bullet_style))
        story.append(Spacer(1, 0.05 * inch))

    # ═══ 9. DEPLOYMENT ════════════════════════════════════════════════════
    story.append(Paragraph("9. Deployment", h1_style))
    story.append(Paragraph(
        "The project ships with multiple deployment options covering local, "
        "containerized, and cloud environments. <b>A live version is already "
        "deployed on Streamlit Community Cloud</b> and can be accessed directly "
        "in any browser — no setup required.", body_style))

    # Live deployment callout
    live_rows = [
        ["🚀 LIVE DEMO", Paragraph(
            f'<link href="{LIVE_APP_URL}" color="#ffffff"><u>{LIVE_APP_URL}</u></link>',
            ParagraphStyle("livelink", fontSize=9, textColor=colors.white,
                           fontName="Helvetica-Bold"))],
        ["📦 SOURCE CODE", Paragraph(
            f'<link href="{GITHUB_URL}" color="#ffffff"><u>{GITHUB_URL}</u></link>',
            ParagraphStyle("ghlink", fontSize=9.5, textColor=colors.white,
                           fontName="Helvetica-Bold"))],
    ]
    live_table = Table(live_rows, colWidths=[4 * cm, 12 * cm])
    live_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#e74c3c")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#c0392b")),
        ("BACKGROUND", (0, 1), (0, 1), colors.HexColor("#1a1a2e")),
        ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (0, -1), 11),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("LEFTPADDING", (1, 0), (1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(live_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("9.1 Local Development", h2_style))
    story.append(Paragraph(
        "pip install -r requirements.txt<br/>"
        "python main.py<br/>"
        "streamlit run app/streamlit_app.py<br/>"
        "uvicorn app.api:app --reload", code_style))

    story.append(Paragraph("9.2 Docker", h2_style))
    story.append(Paragraph(
        "docker-compose up --build<br/>"
        "# Dashboard → http://localhost:8501<br/>"
        "# API       → http://localhost:8000/docs", code_style))

    story.append(Paragraph("9.3 Cloud Deployment", h2_style))
    story.append(Paragraph(
        "<b>Render:</b> one-click deploy via deploy/render.yaml blueprint<br/>"
        "<b>Heroku:</b> deploy/Procfile for web + release steps<br/>"
        "<b>Streamlit Cloud:</b> point to app/streamlit_app.py in a GitHub repo",
        body_style))

    # ═══ 10. API REFERENCE ════════════════════════════════════════════════
    story.append(Paragraph("10. REST API Reference", h1_style))
    api_rows = [
        ["Method", "Endpoint", "Description"],
        ["GET", "/", "Root — API info"],
        ["GET", "/health", "Health check"],
        ["GET", "/docs", "Swagger documentation UI"],
        ["POST", "/predict", "Predict churn for a single customer"],
        ["GET", "/analytics/churn-rate", "Overall churn statistics"],
    ]
    story.append(styled_table(api_rows, col_widths=[2.5 * cm, 5 * cm, 8 * cm]))

    story.append(Paragraph("Example Request", h2_style))
    story.append(Paragraph(
        'POST /predict<br/>'
        'Content-Type: application/json<br/><br/>'
        '{<br/>'
        '&nbsp;&nbsp;"gender": "Female",<br/>'
        '&nbsp;&nbsp;"age": 25,<br/>'
        '&nbsp;&nbsp;"tenure": 2,<br/>'
        '&nbsp;&nbsp;"monthly_charges": 95.0,<br/>'
        '&nbsp;&nbsp;"total_charges": 190.0,<br/>'
        '&nbsp;&nbsp;"contract_type": "Month-to-month",<br/>'
        '&nbsp;&nbsp;"payment_method": "Electronic check",<br/>'
        '&nbsp;&nbsp;"internet_service": "Fiber optic"<br/>'
        '}', code_style))

    story.append(Paragraph("Example Response", h2_style))
    story.append(Paragraph(
        '{<br/>'
        '&nbsp;&nbsp;"prediction": "Yes",<br/>'
        '&nbsp;&nbsp;"churn_probability": 0.7009,<br/>'
        '&nbsp;&nbsp;"retention_probability": 0.2991<br/>'
        '}', code_style))

    # ═══ 11. TECHNOLOGY STACK ═════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("11. Technology Stack", h1_style))
    tech_rows = [
        ["Layer", "Technology", "Purpose"],
        ["Language", "Python 3.11", "Primary runtime"],
        ["Database", "SQLite", "Customer data storage"],
        ["Data", "pandas, numpy", "Data manipulation"],
        ["ML", "scikit-learn", "Preprocessing, LR, RF"],
        ["ML", "XGBoost", "Gradient boosting model"],
        ["Viz", "matplotlib, seaborn", "EDA and evaluation plots"],
        ["Dashboard", "Streamlit", "Interactive web UI"],
        ["API", "FastAPI + Pydantic", "REST endpoint + validation"],
        ["Server", "Uvicorn", "ASGI server"],
        ["Container", "Docker + Compose", "Reproducible deployment"],
        ["Report", "ReportLab", "PDF generation"],
    ]
    story.append(styled_table(tech_rows, col_widths=[3.5 * cm, 5 * cm, 7 * cm]))

    # ═══ 12. FUTURE WORK ══════════════════════════════════════════════════
    story.append(Paragraph("12. Future Improvements", h1_style))
    future = [
        "Replace synthetic data with a real telecom dataset (e.g., IBM Telco Churn)",
        "Add SHAP for per-prediction explainability in the dashboard",
        "Implement model monitoring and data drift detection",
        "Add A/B testing framework for retention campaigns",
        "Migrate from SQLite to PostgreSQL for production scale",
        "Add CI/CD pipeline (GitHub Actions) for automated retraining",
        "Implement authentication for the API (JWT / API keys)",
        "Add unit tests and integration tests with pytest",
    ]
    for f in future:
        story.append(Paragraph(f"• {f}", bullet_style))

    # ═══ 13. CONCLUSION ═══════════════════════════════════════════════════
    story.append(Paragraph("13. Conclusion", h1_style))
    story.append(Paragraph(
        "This project demonstrates a complete end-to-end machine learning workflow "
        "for customer churn prediction — from raw data generation through SQL "
        "analytics, feature engineering, model training and evaluation, all the "
        "way to production-grade deployment via Streamlit, FastAPI, and Docker. "
        "The modular architecture makes it straightforward to swap in real data, "
        "add new models, or extend the dashboard with additional analytics. "
        "The trained models, together with the SQL insights, form a foundation "
        "that a business team can use today to identify at-risk customers and "
        "drive targeted retention campaigns.", body_style))

    story.append(Spacer(1, 0.5 * inch))

    # Decorative closing bar
    closing_bar = Table([[""]], colWidths=[15 * cm], rowHeights=[0.1 * cm])
    closing_bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#e74c3c"))]))
    story.append(closing_bar)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("— End of Report —",
                           ParagraphStyle("End", parent=body_style,
                                          alignment=TA_CENTER,
                                          textColor=colors.HexColor("#7f8c8d"),
                                          fontName="Helvetica-Oblique")))

    # Build
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"\n[OK] Report generated: {OUTPUT_PDF}")
    print(f"  Size: {os.path.getsize(OUTPUT_PDF) / 1024:.1f} KB")
    return OUTPUT_PDF


if __name__ == "__main__":
    build_report()

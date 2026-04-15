# Customer Churn Analysis Project

End-to-end machine learning pipeline for predicting customer churn using Python, SQL, and ML models.

## Features

- Synthetic dataset generation (5,000 customers)
- SQLite database with analytical queries
- Three ML models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with GridSearchCV
- Interactive Streamlit dashboard
- REST API with FastAPI
- Feature importance analysis and churn insights

## Project Structure

```
customer_churn_project/
├── data/                    # SQLite database
├── sql/
│   ├── schema.sql           # Table schema
│   └── queries.sql          # Analysis queries
├── src/
│   ├── data_loader.py       # Data generation & SQL operations
│   ├── preprocessing.py     # Cleaning, encoding, scaling
│   ├── model.py             # Training, prediction, saving
│   ├── evaluate.py          # Metrics, confusion matrix, ROC
│   └── visualize.py         # EDA plots
├── app/
│   ├── streamlit_app.py     # Dashboard
│   └── api.py               # FastAPI endpoint
├── models/                  # Saved model files
├── plots/                   # Generated visualizations
├── main.py                  # Full pipeline script
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This generates data, creates the database, trains models, evaluates them, and saves everything.

With hyperparameter tuning (slower but better results):

```bash
python main.py --tune
```

### Run SQL Queries Directly

```bash
python src/data_loader.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Features:
- Dataset overview with key metrics
- Interactive data explorer with filters
- Live churn prediction form
- Model performance charts
- CSV upload for batch predictions

### Launch FastAPI

```bash
uvicorn app.api:app --reload
```

API docs at `http://localhost:8000/docs`

Endpoints:
- `POST /predict` - Predict churn for a customer
- `GET /analytics/churn-rate` - Get churn statistics
- `GET /health` - Health check

### Predict Churn Programmatically

```python
from src.model import predict_churn

result = predict_churn({
    "gender": "Male",
    "age": 28,
    "tenure": 3,
    "monthly_charges": 89.50,
    "total_charges": 268.50,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
})
print(result)
# {'prediction': 'Yes', 'churn_probability': 0.82, 'retention_probability': 0.18}
```

## Models

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble tree-based model |
| XGBoost | Gradient boosted trees |

Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Generated Outputs

- `plots/churn_distribution.png` - Churn split
- `plots/tenure_vs_churn.png` - Tenure analysis
- `plots/monthly_charges_vs_churn.png` - Charges analysis
- `plots/correlation_heatmap.png` - Feature correlations
- `plots/confusion_matrices.png` - Model confusion matrices
- `plots/roc_curves.png` - ROC-AUC comparison
- `plots/feature_importance.png` - Top churn drivers
- `Project_Report.pdf` - Comprehensive PDF project report

## Project Report

A full PDF report covering problem statement, architecture, dataset, SQL analysis,
visualizations, ML pipeline, results, insights, deployment, and API reference is
generated from the trained artifacts:

```bash
python generate_report.py
# → Project_Report.pdf
```

## Deployment

### Option 1: Docker Compose (recommended)

```bash
docker-compose up --build
# Dashboard → http://localhost:8501
# API       → http://localhost:8000/docs
```

This builds one image and runs two containers (Streamlit + FastAPI) sharing
`data/`, `models/`, and `plots/` volumes. The `main.py` pipeline runs at build
time so the containers ship with a pre-trained model.

### Option 2: Single Docker Container

```bash
docker build -t customer-churn .

# Run the dashboard
docker run -p 8501:8501 customer-churn

# Run the API instead
docker run -p 8000:8000 customer-churn uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Option 3: Render.com (free tier)

Push this repo to GitHub and point Render at `deploy/render.yaml`. The blueprint
provisions both services (API + dashboard) automatically. No config needed.

### Option 4: Heroku

```bash
heroku create churn-api
heroku buildpacks:set heroku/python
cp deploy/Procfile .
cp deploy/runtime.txt .
git push heroku main
```

### Option 5: Streamlit Community Cloud

Fork the repo → share.streamlit.io → point to `app/streamlit_app.py`.
Works out of the box with `requirements.txt` and `.streamlit/config.toml`.

### Option 6: Direct Python (dev only)

```bash
python main.py                        # train + evaluate
streamlit run app/streamlit_app.py    # dashboard
uvicorn app.api:app --reload          # API
```

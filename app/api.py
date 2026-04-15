"""FastAPI endpoint for churn prediction."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.model import predict_churn
from src.data_loader import run_analysis_queries

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using trained ML models",
    version="1.0.0",
)


class CustomerInput(BaseModel):
    gender: str = Field(..., pattern="^(Male|Female)$")
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0, le=100)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract_type: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    payment_method: str = Field(..., pattern="^(Electronic check|Mailed check|Bank transfer|Credit card)$")
    internet_service: str = Field(..., pattern="^(DSL|Fiber optic|No)$")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 35,
                "tenure": 12,
                "monthly_charges": 75.50,
                "total_charges": 906.00,
                "contract_type": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "Fiber optic",
            }
        }


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    retention_probability: float


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    """Predict churn for a single customer."""
    try:
        result = predict_churn(customer.model_dump())
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/churn-rate")
def churn_rate():
    """Get overall churn rate statistics."""
    try:
        results = run_analysis_queries()
        return {"churn_rate": results.get("churn_rate", {}).to_dict("records") if hasattr(results.get("churn_rate"), "to_dict") else str(results.get("churn_rate"))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

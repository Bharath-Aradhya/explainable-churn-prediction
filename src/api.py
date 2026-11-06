"""Serve the trained churn model.  Run:  uvicorn src.api:app --reload"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

bundle = joblib.load("models/best_model.joblib")
app = FastAPI(title="Telco churn prediction")


class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float | None = None
    SeniorCitizen: int = 0
    gender: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str


@app.post("/predict")
def predict(c: Customer):
    proba = float(bundle["pipeline"].predict_proba(pd.DataFrame([c.model_dump()]))[0, 1])
    return {"churn_probability": round(proba, 4),
            "contact_for_retention": proba >= bundle["threshold"]}

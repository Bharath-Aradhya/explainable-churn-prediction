"""Load and clean the IBM Telco Customer Churn dataset."""
from pathlib import Path

import pandas as pd

TARGET = "Churn"
ID_COL = "customerID"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return features X and binary target y (1 = churned).

    Cleaning steps (documented because they are typical interview questions):
    * TotalCharges is stored as text and is blank for brand-new customers
      (tenure == 0). It is converted to numbers; blanks become NaN and are
      imputed later INSIDE the model pipeline, so no test-set information
      leaks into training.
    * customerID is dropped: it is a unique identifier, and models can
      memorise it without learning anything general.
    """
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns=[ID_COL], errors="ignore")

    y = (df.pop(TARGET).astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    return X, y

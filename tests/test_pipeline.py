"""Smoke tests on a small synthetic dataset with the same columns as Telco."""
import json

import numpy as np
import pandas as pd
import pytest

from src.data import CATEGORICAL_COLS, load_data
from src.train import main

CHOICES = {
    "gender": ["Male", "Female"], "Partner": ["Yes", "No"], "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"], "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)",
                      "Credit card (automatic)"],
}


@pytest.fixture
def fake_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({c: rng.choice(v, n) for c, v in CHOICES.items()})
    df["customerID"] = [f"id{i}" for i in range(n)]
    df["tenure"] = rng.integers(0, 72, n)
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["MonthlyCharges"] = rng.uniform(20, 120, n).round(2)
    total = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)
    total[df["tenure"] == 0] = " "  # blanks, like the real dataset
    df["TotalCharges"] = total
    risk = (df["Contract"] == "Month-to-month") * 1.5 - df["tenure"] / 30
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-risk)), "Yes", "No")
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    return path


def test_load_data_cleans_columns(fake_csv):
    X, y = load_data(fake_csv)
    assert "customerID" not in X.columns
    assert X["TotalCharges"].dtype.kind == "f"
    assert set(y.unique()) <= {0, 1}
    assert set(CATEGORICAL_COLS) <= set(X.columns)


def test_training_runs_and_beats_chance(fake_csv, tmp_path):
    res = main(str(fake_csv), out_dir=str(tmp_path / "r"), model_dir=str(tmp_path / "m"))
    assert res["test"]["roc_auc"] > 0.6
    assert (tmp_path / "m" / "best_model.joblib").exists()
    json.loads((tmp_path / "r" / "metrics.json").read_text())

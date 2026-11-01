from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]

def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create row-level churn features; never uses the target or customer ID."""
    df = frame.copy()
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "tenure" in df.columns:
        tenure = pd.to_numeric(df["tenure"], errors="coerce")
        df["tenure_years"] = tenure / 12.0
        df["tenure_group"] = pd.cut(
            tenure, bins=[-1, 12, 24, 48, 60, 1000],
            labels=["0-1y", "1-2y", "2-4y", "4-5y", "5y+"]
        ).astype("object")
    if {"TotalCharges", "tenure"}.issubset(df.columns):
        total = pd.to_numeric(df["TotalCharges"], errors="coerce")
        tenure = pd.to_numeric(df["tenure"], errors="coerce").replace(0, np.nan)
        df["avg_charge_per_month"] = total / tenure
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available = [c for c in service_cols if c in df.columns]
    if available:
        df["number_of_services"] = sum(
            df[c].astype(str).isin(["Yes", "Yes"]).astype(int) for c in available
        )
    if "Contract" in df.columns:
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    if "PaymentMethod" in df.columns:
        df["uses_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    if "InternetService" in df.columns:
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
    ])
    return ColumnTransformer([
        ("numeric", num_pipe, numeric),
        ("categorical", cat_pipe, categorical)
    ], remainder="drop")

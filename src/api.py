"""
FastAPI /predict endpoint for the Adult Census Income best model.

Loads:
  • models/best_model.pkl       — best classifier
  • models/scaler.pkl           — fitted StandardScaler
  • models/ohe_encoder.pkl      — fitted OneHotEncoder
  • models/preprocessing_meta.pkl — imputation values, IQR bounds, top countries

Run:
  uvicorn src.api:app --host 0.0.0.0 --port 8080
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Load artifacts once at startup
model   = joblib.load("models/best_model.pkl")
scaler  = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/ohe_encoder.pkl")
meta    = joblib.load("models/preprocessing_meta.pkl")
# meta contains: fill_values, iqr_bounds, top_countries, label_encoders

# Pydantic input schema
class IncomeInput(BaseModel):
    age:              int   = Field(..., example=39)
    workclass:        str   = Field(..., example="State-gov")
    fnlwgt:           int   = Field(..., example=77516)
    education:        str   = Field(..., example="Bachelors")
    education_num:    int   = Field(..., example=13)
    marital_status:   str   = Field(..., example="Never-married")
    occupation:       str   = Field(..., example="Adm-clerical")
    relationship:     str   = Field(..., example="Not-in-family")
    race:             str   = Field(..., example="White")
    sex:              str   = Field(..., example="Male")
    capital_gain:     int   = Field(..., example=2174)
    capital_loss:     int   = Field(..., example=0)
    hours_per_week:   int   = Field(..., example=40)
    native_country:   str   = Field(..., example="United-States")


class IncomeOutput(BaseModel):
    prediction:  int
    label:       str
    probability: float



# Preprocessing for a single input row
def preprocess_input(data: IncomeInput):
    """Replicate the training preprocessing pipeline for one sample."""
    df = pd.DataFrame([data.model_dump()])

    # Impute missing values (replace '?' with training modes)
    for col, val in meta["fill_values"].items():
        df[col] = df[col].replace("?", val)

    # Outlier clipping (IQR bounds from training set)
    for col, (lower, upper) in meta["iqr_bounds"].items():
        df[col] = df[col].clip(lower=lower, upper=upper)

    # Group rare native_country values
    top_countries = meta["top_countries"]
    df["native_country"] = df["native_country"].where(
        df["native_country"].isin(top_countries), other="Other"
    )

    # Label-encode binary columns (sex)
    for col, le in meta["label_encoders"].items():
        df[col] = le.transform(df[col])

    # One-Hot encode nominal columns
    ohe_cols = [c for c in encoder.feature_names_in_]
    encoded_array = encoder.transform(df[ohe_cols])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(ohe_cols),
        index=df.index,
    )
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Feature engineering
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]
    df["hours_education_interaction"] = df["hours_per_week"] * df["education_num"]

    # ensure column order matches training
    df = df[scaler.feature_names_in_]
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    return df


# App
app = FastAPI(
    title="Adult Census Income Predictor",
    description="Predict whether income exceeds $50K/year",
    version="1.0.0",
)

@app.post("/predict", response_model=IncomeOutput)
def predict(data: IncomeInput):
    try:
        X = preprocess_input(data)
        prob = float(model.predict_proba(X)[:, 1][0])
        pred = int(prob >= 0.5)
        label = ">50K" if pred == 1 else "<=50K"
        return IncomeOutput(prediction=pred, label=label, probability=round(prob, 4))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
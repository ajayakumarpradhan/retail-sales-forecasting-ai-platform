# ────────────────────────────────────────────────
# api.py  —  Rossmann Sales Prediction REST API
# Run: uvicorn api:app --reload --port 8000
# ────────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib

app = FastAPI(
    title="Rossmann Sales Prediction API",
    description="Predicts daily sales for Rossmann stores using XGBoost",
    version="1.0.0"
)

# Load artifacts on startup
model    = joblib.load("model_artifacts/xgb_rossmann_model.pkl")
FEATURES = joblib.load("model_artifacts/feature_list.pkl")


class PredictionRequest(BaseModel):
    store_id             : int   = Field(..., ge=1, le=1115, examples=[1])
    date                 : str   = Field(..., examples=["2025-06-15"])
    promo                : int   = Field(..., ge=0, le=1, examples=[1])
    state_holiday        : int   = Field(0, ge=0, le=3, examples=[0])
    school_holiday       : int   = Field(0, ge=0, le=1, examples=[0])
    store_type           : int   = Field(..., ge=0, le=3, examples=[0])
    assortment           : int   = Field(..., ge=0, le=2, examples=[0])
    competition_distance : float = Field(..., gt=0, examples=[1270.0])
    promo2               : int   = Field(0, ge=0, le=1, examples=[0])
    promo_interval       : str   = Field("None", examples=["Jan,Apr,Jul,Oct"])


class PredictionResponse(BaseModel):
    store_id        : int
    date            : str
    predicted_sales : float
    promo_active    : bool


@app.get("/")
def root():
    return {"message": "Rossmann Sales Prediction API", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost", "features": len(FEATURES)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        dt = pd.to_datetime(req.date)
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid date format. Use YYYY-MM-DD")

    month_abbr_map = {
        "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
        "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
    }
    is_promo2_active = 0
    if req.promo2 == 1 and req.promo_interval != "None":
        try:
            promo_months = [month_abbr_map[m.strip()] for m in req.promo_interval.split(",")]
            is_promo2_active = 1 if dt.month in promo_months else 0
        except KeyError:
            is_promo2_active = 0

    row = {
        "Store"                 : req.store_id,
        "DayOfWeek"             : dt.dayofweek + 1,
        "Year"                  : dt.year,
        "Month"                 : dt.month,
        "Day"                   : dt.day,
        "WeekOfYear"            : int(dt.isocalendar().week),
        "Quarter"               : dt.quarter,
        "IsWeekend"             : int(dt.dayofweek >= 5),
        "IsMonthStart"          : int(dt.day <= 5),
        "IsMonthEnd"            : int(dt.day >= 25),
        "Promo"                 : req.promo,
        "StateHoliday"          : req.state_holiday,
        "SchoolHoliday"         : req.school_holiday,
        "StoreType"             : req.store_type,
        "Assortment"            : req.assortment,
        "CompetitionDistance"   : req.competition_distance,
        "CompetitionOpenMonths" : 0,
        "Promo2"                : req.promo2,
        "IsPromo2Active"        : is_promo2_active
    }

    X_input = pd.DataFrame([row])[FEATURES]
    prediction = float(max(0, model.predict(X_input)[0]))

    return PredictionResponse(
        store_id        = req.store_id,
        date            = req.date,
        predicted_sales = round(prediction, 2),
        promo_active    = bool(req.promo)
    )


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]):
    return [predict(req) for req in requests]

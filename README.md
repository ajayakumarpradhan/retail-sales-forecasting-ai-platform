# Rossmann Retail Sales Forecasting AI Platform

An end-to-end machine learning system for predicting daily sales across 1,115 Rossmann drug stores.

## Features

- **3 ML Models**: Linear Regression (baseline), Random Forest, XGBoost (final, R2 ~ 0.89)
- **19 Engineered Features**: temporal, promotional, competition, store attributes
- **FastAPI REST API**: `/predict` and `/predict/batch` endpoints
- **Streamlit Dashboard**: KPIs, forecast charts, What-If analysis
- **AI Chatbot**: Context-aware assistant for querying forecast data
- **What-If Analysis**: Promo impact simulation, competition distance sensitivity

## Quick Start

```bash
# Install dependencies
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib fastapi uvicorn streamlit

# Train the model (generates model_artifacts/)
python Rossmann_Retail_Sales_Prediction.py

# Start FastAPI
python -m uvicorn api:app --port 8000

# Start Streamlit Dashboard
python -m streamlit run app.py --server.port 8501
```

## Project Structure

```
├── Rossmann_Retail_Sales_Prediction.py   # Full ML pipeline (EDA + training)
├── api.py                                 # FastAPI REST API
├── app.py                                 # Streamlit Dashboard + AI Chatbot
├── model_artifacts/
│   ├── xgb_rossmann_model.pkl            # Trained XGBoost model
│   ├── feature_list.pkl                  # Feature names
│   └── comp_distance_median.pkl          # Imputation value
├── Rossmann Stores Data.csv              # Main dataset (not in repo)
├── store.csv                             # Store metadata (not in repo)
└── .gitignore
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_id":1,"date":"2025-06-15","promo":1,"state_holiday":0,"school_holiday":0,"store_type":0,"assortment":0,"competition_distance":1270.0}'
```

## Dataset

Download from [Kaggle - Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) and place CSV files in the project root.

## Tech Stack

| Component | Technology |
|---|---|
| ML Framework | XGBoost, Scikit-learn |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Language | Python 3.10+ |

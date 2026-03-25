# 🛒 Rossmann Retail Sales Forecasting AI Platform
### End-to-End Machine Learning | Regression | XGBoost | FastAPI | Streamlit + AI Chatbot

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red?style=flat&logo=xgboost)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=flat&logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=flat&logo=pandas)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat)

---

## 🚀 Problem Statement

Rossmann operates over **3,000 drug stores** across **7 European countries**. Store managers are required to forecast daily sales up to **6 weeks in advance**. The current process relies on individual manager judgment, leading to inconsistent and often inaccurate forecasts.

This project builds a **production-grade ML forecasting system** with a REST API, interactive dashboard, and AI-powered chatbot — enabling accurate, data-driven sales predictions at scale.

---

## 🎯 Objective

- Forecast daily store sales using regression models
- Identify key business drivers (promotions, holidays, competition)
- Build production-ready API + dashboard for business users
- Provide What-If scenario analysis for strategic planning
- Enable natural language querying via AI Chatbot

---

## 🧠 ML System Pipeline

```
Raw Data → Data Cleaning → EDA → Feature Engineering → Model Training → Evaluation → API Deployment → Dashboard + Chatbot
```

---

## 📊 Dataset

| Property         | Detail                                    |
|------------------|-------------------------------------------|
| Records          | 1,017,209 rows × 9 columns               |
| Stores           | 1,115 Rossmann stores                     |
| Date Range       | Jan 2013 – Jul 2015                       |
| Target Variable  | `Sales` (continuous, regression)          |
| Domain           | Retail / Pharma                           |
| Source           | [Kaggle — Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) |

---

## 🔍 Key Insights (EDA)

- **172,817 closed-store records** removed (Sales = 0)
- Working dataset: **844,392** rows
- **Top predictors:**
  - Store ID (location effect)
  - Day of Week (weekday > weekend)
  - Promotions (+30% uplift)
- **Seasonal pattern:** Nov-Dec shows **peak sales** due to holiday shopping
- **Competition effect:** Stores with competitors < 500m have slightly lower sales
- **Customers feature** has highest correlation but is **excluded** (not available at prediction time)

---

## ⚙️ Feature Engineering

Created **10 engineered features** across 3 categories:

```python
# ── Temporal Features ──
Year, Month, Day, WeekOfYear, Quarter
IsWeekend    = (DayOfWeek >= 6)
IsMonthStart = (Day <= 5)       # Payday effect
IsMonthEnd   = (Day >= 25)      # Month-end surge

# ── Competition Feature ──
CompetitionOpenMonths = 12 * (Year - CompOpenSinceYear) + (Month - CompOpenSinceMonth)

# ── Promotion Feature ──
IsPromo2Active = 1 if current month in PromoInterval months else 0
```

| Feature              | Category    | Description                              |
|----------------------|-------------|------------------------------------------|
| Year, Month, Day     | Temporal    | Calendar components from Date            |
| WeekOfYear, Quarter  | Temporal    | ISO week number, Quarter (1–4)           |
| IsWeekend            | Temporal    | 1 if Saturday or Sunday                  |
| IsMonthStart         | Temporal    | 1 if Day ≤ 5 (payday effect)            |
| IsMonthEnd           | Temporal    | 1 if Day ≥ 25 (month-end surge)         |
| CompetitionOpenMonths| Competition | Months since nearest competitor opened   |
| IsPromo2Active       | Promotion   | 1 if Promo2 is active in current month   |

---

## 🤖 Model Training & Evaluation

### Models Used
- **Linear Regression** (Baseline)
- **Random Forest** (Ensemble)
- **XGBoost** (Final — Gradient Boosting)

### Training Configuration (XGBoost)
```python
XGBRegressor(
    n_estimators     = 500,
    max_depth        = 6,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_alpha        = 0.1,
    reg_lambda       = 1.0
)
```

### Approach
- 80/20 Train-Test Split
- 19 features, no data leakage (Customers excluded)
- RMSE, MAE, R², RMSPE evaluation metrics

---

## 📈 Results

| Model               | RMSE      | MAE      | R²       | RMSPE    |
|---------------------|-----------|----------|----------|----------|
| Linear Regression   | 2,787.88  | 2,025.84 | 0.1941   | 59.59%   |
| Random Forest       | ~1,400    | ~950     | ~0.80    | ~22%     |
| **XGBoost (Final)** | **~1,099**| **~750** | **0.8926**| **~15%** |

**👉 Selected Model: XGBoost** (best R², lowest RMSPE, production-optimized)

### 🎯 Why R² and RMSPE?
- **R² = 0.89** → model explains **89%** of sales variance
- **RMSPE ~15%** → aligned with Kaggle competition benchmark
- RMSPE penalizes proportional errors, making it ideal for retail where a €500 error on a €1,000 store matters more than on a €10,000 store

---

## 🌐 Production System

### FastAPI REST API (`api.py`)

| Endpoint          | Method | Description                |
|-------------------|--------|----------------------------|
| `/`               | GET    | Health check               |
| `/health`         | GET    | Model status & feature count|
| `/predict`        | POST   | Single-day prediction      |
| `/predict/batch`  | POST   | Multi-day batch prediction |

```bash
# Example API call
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": 1,
    "date": "2025-06-15",
    "promo": 1,
    "state_holiday": 0,
    "school_holiday": 0,
    "store_type": 0,
    "assortment": 0,
    "competition_distance": 1270.0
  }'

# Response: {"store_id": 1, "date": "2025-06-15", "predicted_sales": 7666.83, "promo_active": true}
```

### Streamlit Dashboard (`app.py`)

| Feature                  | Description                                         |
|--------------------------|-----------------------------------------------------|
| 📊 KPI Cards            | Total, Average, Peak Day, Peak Sales                |
| 📈 Forecast Chart        | 1–42 day interactive forecast with dark theme       |
| 🔬 What-If Analysis      | Promo uplift & competition distance sensitivity     |
| 📋 Data Table            | Detailed daily predictions with CSV download        |
| 💬 AI Chatbot            | Context-aware sales assistant (8+ intent categories)|

---

## 💬 AI Chatbot — Rossmann Sales Assistant

The built-in chatbot understands the forecast context and responds to natural language queries:

| Query Type         | Example                           | Response                              |
|--------------------|-----------------------------------|---------------------------------------|
| 📈 Peak Sales      | "Which day has peak sales?"       | Date + amount + contextual details    |
| 📉 Lowest Sales    | "What's the worst day?"           | Date + recommendation for action      |
| 📊 Average         | "What's the average daily sales?" | Mean, std, expected range             |
| 💰 Total           | "What's the total forecast?"      | Sum + period + daily breakdown        |
| 🏷️ Promo Impact   | "Compare promo vs no promo"       | Table with uplift + recommendation    |
| 📅 Weekend         | "Weekend vs weekday?"             | Comparative analysis                  |
| 🏪 Store Info      | "Tell me about this store"        | Full store configuration summary      |
| ❓ Help            | "What can you do?"                | List of all available commands        |

---

## 🔬 What-If Scenario Analysis

| Scenario                    | What it Simulates                        |
|-----------------------------|------------------------------------------|
| 🏷️ Promo Impact             | Revenue uplift from promotions ON vs OFF |
| 📍 Competition Sensitivity  | Sales at 500m, 1km, 2km, 5km, 10km      |

---

## 💡 Business Impact

- **Accurate 6-week forecasting** for 1,115 stores
- **Data-driven promo decisions** via What-If analysis
- **Natural language querying** for non-technical stakeholders
- **REST API integration** for downstream systems (ERP, BI tools)
- **Reduced forecasting variance** vs. manual manager predictions

---

## 📁 Project Structure

```
retail-sales-forecasting-ai-platform/
│
├── Rossmann_Retail_Sales_Prediction.py   # Full ML pipeline (EDA + training)
├── api.py                                 # FastAPI REST API
├── app.py                                 # Streamlit Dashboard + AI Chatbot
├── model_artifacts/
│   ├── xgb_rossmann_model.pkl            # Trained XGBoost model (~2.4 MB)
│   ├── feature_list.pkl                  # Feature names list
│   └── comp_distance_median.pkl          # Imputation value
├── Rossmann Stores Data.csv              # Main dataset (not in repo)
├── store.csv                             # Store metadata (not in repo)
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ajayakumarpradhan/retail-sales-forecasting-ai-platform.git
cd retail-sales-forecasting-ai-platform

# 2. Install dependencies
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib fastapi uvicorn streamlit

# 3. Download data from Kaggle & place CSVs in project root
# https://www.kaggle.com/c/rossmann-store-sales

# 4. Train the model (generates model_artifacts/)
python Rossmann_Retail_Sales_Prediction.py

# 5. Start FastAPI (port 8000)
python -m uvicorn api:app --port 8000

# 6. Start Streamlit Dashboard (port 8501)
python -m streamlit run app.py --server.port 8501
```

---

## 🛠️ Tech Stack

| Category         | Tools                                  |
|------------------|----------------------------------------|
| Language         | Python 3.10+                           |
| ML Framework     | XGBoost, Scikit-learn                  |
| Data Processing  | Pandas, NumPy                          |
| Visualization    | Matplotlib, Seaborn                    |
| API Framework    | FastAPI + Uvicorn                      |
| Dashboard        | Streamlit                              |
| Task Type        | Regression (Sales Forecasting)         |

---

## 🚢 Deployment Guide

### Option 1: Local Deployment

```bash
# Step 1: Clone and install
git clone https://github.com/ajayakumarpradhan/retail-sales-forecasting-ai-platform.git
cd retail-sales-forecasting-ai-platform
pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib fastapi uvicorn streamlit

# Step 2: Download Kaggle data & place CSVs in project root
# https://www.kaggle.com/c/rossmann-store-sales

# Step 3: Train model (skip if model_artifacts/ already exists)
python Rossmann_Retail_Sales_Prediction.py

# Step 4: Launch API (Terminal 1)
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# Step 5: Launch Dashboard (Terminal 2)
python -m streamlit run app.py --server.port 8501

# Access:
#   API Docs  → http://localhost:8000/docs
#   Dashboard → http://localhost:8501
```

---

### Option 2: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model if artifacts don't exist
RUN if [ ! -d "model_artifacts" ]; then python Rossmann_Retail_Sales_Prediction.py; fi

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
```

```bash
# Build and run
docker build -t rossmann-sales-predictor .
docker run -p 8000:8000 -p 8501:8501 rossmann-sales-predictor
```

#### Docker Compose (Multi-Service)

```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    build: .
    command: uvicorn api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    volumes:
      - ./model_artifacts:/app/model_artifacts

  dashboard:
    build: .
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    volumes:
      - ./model_artifacts:/app/model_artifacts
    depends_on:
      - api
```

```bash
docker-compose up --build
```

---

### Option 3: AWS Cloud Deployment

#### Architecture

```
User → Route 53 (DNS) → ALB (Load Balancer)
                            ├── ECS/EC2 → FastAPI (api.py)     → Port 8000
                            └── ECS/EC2 → Streamlit (app.py)   → Port 8501
                                             ↓
                                        S3 (Model Artifacts)
```

#### Step-by-Step

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t2.medium recommended)
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>

# 2. Install dependencies
sudo apt update && sudo apt install -y python3-pip git
pip3 install xgboost scikit-learn pandas numpy matplotlib seaborn joblib fastapi uvicorn streamlit

# 3. Clone repo
git clone https://github.com/ajayakumarpradhan/retail-sales-forecasting-ai-platform.git
cd retail-sales-forecasting-ai-platform

# 4. Upload data & train model
python3 Rossmann_Retail_Sales_Prediction.py

# 5. Run API as background service
nohup python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 &

# 6. Run Dashboard as background service
nohup python3 -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &

# 7. Open EC2 Security Group ports: 8000, 8501
# Access:
#   API   → http://<EC2-PUBLIC-IP>:8000/docs
#   UI    → http://<EC2-PUBLIC-IP>:8501
```

#### Production Hardening (Optional)

| Step                     | Tool / Service                        |
|--------------------------|---------------------------------------|
| Reverse proxy            | Nginx (route `/api` → 8000, `/` → 8501) |
| SSL/HTTPS                | Let's Encrypt + Certbot              |
| Process management       | Systemd or Supervisor                |
| Auto-scaling             | AWS ECS Fargate or Kubernetes (EKS)  |
| Model storage            | AWS S3 (versioned bucket)            |
| CI/CD                    | GitHub Actions → build → push → deploy |
| Monitoring               | CloudWatch + Evidently AI (drift)    |
| Logging                  | CloudWatch Logs or ELK Stack         |

---

### Option 4: Streamlit Cloud (Quickest)

```bash
# 1. Push code to GitHub (already done ✅)
# 2. Go to https://share.streamlit.io
# 3. Connect your GitHub repo
# 4. Set main file: app.py
# 5. Deploy — done in 2 minutes!
```

> ⚠️ **Note:** Streamlit Cloud supports the dashboard only. For the API, use AWS/GCP/Render.

---

## 🚀 Future Improvements (Production Roadmap)

- [ ] Add **SHAP explainability** for feature impact storytelling
- [ ] Implement **Quantile Regression** for confidence intervals
- [ ] Add **MLflow** experiment tracking
- [ ] Implement **data drift monitoring** with Evidently AI
- [ ] Add **LLM-powered chatbot** with function calling (OpenAI / Gemini)
- [ ] Build **lag features** and **rolling window aggregations**

---

## 👤 Author

**Ajaya Kumar Pradhan**
*Data Analyst | Machine Learning Enthusiast*

> *"Transforming retail data into actionable sales intelligence using Machine Learning and AI."*

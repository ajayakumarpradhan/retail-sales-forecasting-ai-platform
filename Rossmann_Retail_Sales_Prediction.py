"""
# 🛒 Rossmann Retail Sales Prediction
### End-to-End Machine Learning Pipeline | Regression | XGBoost

---

**Business Problem:**  
Rossmann operates over 3,000 drug stores across 7 European countries. Store managers are required to forecast daily sales up to **6 weeks in advance**. The current process relies on individual manager judgment, leading to inconsistent and often inaccurate forecasts.

**Objective:**  
Build an end-to-end machine learning pipeline to accurately forecast the `Sales` column for 1,115 stores, incorporating promotions, competition, seasonality, holidays, and store-specific attributes.

---

| | |
|---|---|
| **Dataset** | Rossmann Stores Data (Kaggle) |
| **Records** | 1,017,209 rows × 9 columns |
| **Target** | `Sales` (continuous regression) |
| **Algorithm** | XGBoost Regressor |
| **Deployment** | FastAPI + Streamlit |

---
"""

"""
## 📦 Section 1: Install & Import Libraries
"""

# Install required libraries
# !pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib fastapi uvicorn streamlit

# Core libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Preprocessing & modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Model persistence
import joblib

# Plot styling
plt.rcParams['figure.figsize'] = (14, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('muted')

print('✅ All libraries loaded successfully')
print(f'   pandas {pd.__version__} | numpy {np.__version__} | xgboost {xgb.__version__}')

"""
---
## 📂 Section 2: Load Data
"""

# Load main sales data and store metadata
df    = pd.read_csv('Rossmann Stores Data.csv', low_memory=False)
store = pd.read_csv('store.csv')

print('=== Main Dataset ===')
print(f'Shape : {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
print()
print('=== Store Metadata ===')
print(f'Shape : {store.shape[0]:,} rows × {store.shape[1]} columns')

# Preview main dataset
df.head()

# Preview store metadata
store.head()

# Data types and basic info
print('=== Main Dataset Info ===')
df.info()
print()
print('=== Store Metadata Info ===')
store.info()

"""
---
## 🔍 Section 3: Exploratory Data Analysis (EDA)
"""

"""
### 3.1 — Missing Values
"""

def missing_summary(data, name):
    missing = data.isnull().sum()
    missing_pct = (missing / len(data) * 100).round(2)
    summary = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    summary = summary[summary['Missing Count'] > 0].sort_values('Missing %', ascending=False)
    print(f'=== {name} — Missing Values ===')
    if summary.empty:
        print('  No missing values found ✅')
    else:
        print(summary.to_string())
    print()

missing_summary(df, 'Main Dataset')
missing_summary(store, 'Store Metadata')

# Visualize missing values in store.csv
fig, ax = plt.subplots(figsize=(10, 4))
missing_store = store.isnull().sum().sort_values(ascending=True)
missing_store = missing_store[missing_store > 0]
colors = ['#e07b54' if v > 300 else '#f5b88a' for v in missing_store.values]
bars = ax.barh(missing_store.index, missing_store.values, color=colors, height=0.5)
for bar, val in zip(bars, missing_store.values):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
            f'{val} ({val/1115*100:.1f}%)', va='center', fontsize=10)
ax.set_xlabel('Missing Count')
ax.set_title('Missing Values — store.csv', fontweight='bold', pad=12)
plt.tight_layout()
# plt.show()

print('\n📌 Note: Promo2* nulls are structural — they belong to stores where Promo2=0')
print('   CompetitionDistance nulls: 3 records — will impute with median')

"""
### 3.2 — Target Variable: Sales Distribution
"""

# Filter to open stores for meaningful distribution
df_open = df[df['Open'] == 1].copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Histogram
axes[0].hist(df_open['Sales'], bins=80, color='#5b8dd9', edgecolor='white', linewidth=0.4)
axes[0].axvline(df_open['Sales'].mean(), color='#e07b54', linewidth=2, label=f"Mean: {df_open['Sales'].mean():,.0f}")
axes[0].axvline(df_open['Sales'].median(), color='#50a882', linewidth=2, linestyle='--', label=f"Median: {df_open['Sales'].median():,.0f}")
axes[0].set_title('Sales Distribution', fontweight='bold')
axes[0].set_xlabel('Sales')
axes[0].legend(fontsize=9)

# Log-transformed
axes[1].hist(np.log1p(df_open['Sales']), bins=80, color='#9b72cf', edgecolor='white', linewidth=0.4)
axes[1].set_title('Sales Distribution (Log Scale)', fontweight='bold')
axes[1].set_xlabel('log(1 + Sales)')

# Boxplot by DayOfWeek
day_map = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
df_open['Day_Name'] = df_open['DayOfWeek'].map(day_map)
order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
sns.boxplot(data=df_open, x='Day_Name', y='Sales', order=order, ax=axes[2],
            palette='Blues', flierprops={'marker':'.','markersize':2})
axes[2].set_title('Sales by Day of Week', fontweight='bold')
axes[2].set_xlabel('')

plt.suptitle('Target Variable — Sales Analysis', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
# plt.show()

print(f'Closed-store records removed : {(df["Open"]==0).sum():,} rows (Sales=0)')
print(f'Working dataset              : {len(df_open):,} rows')

"""
### 3.3 — Temporal Trends
"""

df_open['Date'] = pd.to_datetime(df_open['Date'])

# Monthly average sales trend
monthly = df_open.groupby(df_open['Date'].dt.to_period('M'))['Sales'].mean().reset_index()
monthly['Date'] = monthly['Date'].dt.to_timestamp()

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Monthly trend
axes[0].plot(monthly['Date'], monthly['Sales'], color='#5b8dd9', linewidth=2)
axes[0].fill_between(monthly['Date'], monthly['Sales'], alpha=0.15, color='#5b8dd9')
axes[0].set_title('Average Daily Sales — Monthly Trend', fontweight='bold')
axes[0].set_ylabel('Avg Sales')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# ================= FIX START =================
monthly_avg = df_open.groupby(df_open['Date'].dt.month)['Sales'].mean()

# Force all 12 months (this fixes shape mismatch)
monthly_avg = monthly_avg.reindex(range(1, 13))

month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
colors = ['#e07b54' if m in [11,12] else '#5b8dd9' for m in range(1,13)]

axes[1].bar(month_names, monthly_avg.values, color=colors, edgecolor='white')
# ================= FIX END =================

axes[1].set_title('Average Sales by Month (Nov-Dec highlighted as peak)', fontweight='bold')
axes[1].set_ylabel('Avg Sales')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
# plt.show()

"""
### 3.4 — Impact of Promotions & Holidays
"""

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Promo effect
promo_avg = df_open.groupby('Promo')['Sales'].mean()
bars = axes[0].bar(['No Promo', 'Promo'], promo_avg.values,
                   color=['#a8c7e8','#e07b54'], edgecolor='white', width=0.5)
for bar, val in zip(bars, promo_avg.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'{val:,.0f}', ha='center', fontweight='bold', fontsize=11)
uplift = (promo_avg[1] - promo_avg[0]) / promo_avg[0] * 100
axes[0].set_title(f'Promo Effect\n(+{uplift:.1f}% uplift)', fontweight='bold')
axes[0].set_ylabel('Avg Sales')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# School holiday effect
school_avg = df_open.groupby('SchoolHoliday')['Sales'].mean()
bars2 = axes[1].bar(['No Holiday', 'School Holiday'], school_avg.values,
                    color=['#a8c7e8','#9b72cf'], edgecolor='white', width=0.5)
for bar, val in zip(bars2, school_avg.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'{val:,.0f}', ha='center', fontweight='bold', fontsize=11)
axes[1].set_title('School Holiday Effect', fontweight='bold')
axes[1].set_ylabel('Avg Sales')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# State holiday effect
state_map = {'0':'None','a':'Public','b':'Easter','c':'Christmas'}
df_open['HolidayLabel'] = df_open['StateHoliday'].astype(str).map(state_map)
state_avg = df_open.groupby('HolidayLabel')['Sales'].mean().reindex(['None','Public','Easter','Christmas'])
bars3 = axes[2].bar(state_avg.index, state_avg.values,
                    color=['#a8c7e8','#e07b54','#50a882','#9b72cf'], edgecolor='white', width=0.5)
for bar, val in zip(bars3, state_avg.values):
    if not np.isnan(val):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                     f'{val:,.0f}', ha='center', fontweight='bold', fontsize=10)
axes[2].set_title('State Holiday Effect', fontweight='bold')
axes[2].set_ylabel('Avg Sales')
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.suptitle('Business Drivers — Promotional & Holiday Impact on Sales', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
# plt.show()

"""
### 3.5 — Store Type & Assortment Analysis
"""

# Merge to get store attributes
df_eda = df_open.merge(store, on='Store', how='left')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# StoreType
type_avg = df_eda.groupby('StoreType')['Sales'].mean().sort_values(ascending=False)
type_counts = store['StoreType'].value_counts()
colors = ['#e07b54','#5b8dd9','#50a882','#9b72cf']
bars = axes[0].bar(type_avg.index, type_avg.values, color=colors, edgecolor='white', width=0.5)
for i, (bar, val) in enumerate(zip(bars, type_avg.values)):
    store_type = type_avg.index[i]
    count = type_counts.get(store_type, 0)
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'{val:,.0f}\n(n={count})', ha='center', fontsize=10)
axes[0].set_title('Avg Sales by Store Type', fontweight='bold')
axes[0].set_ylabel('Avg Sales')
axes[0].set_xlabel('Store Type')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

# Assortment
assort_map = {'a':'Basic','b':'Extra','c':'Extended'}
df_eda['AssortmentLabel'] = df_eda['Assortment'].map(assort_map)
assort_avg = df_eda.groupby('AssortmentLabel')['Sales'].mean().sort_values(ascending=False)
bars2 = axes[1].bar(assort_avg.index, assort_avg.values,
                    color=['#e07b54','#5b8dd9','#50a882'], edgecolor='white', width=0.5)
for bar, val in zip(bars2, assort_avg.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'{val:,.0f}', ha='center', fontsize=11)
axes[1].set_title('Avg Sales by Assortment Type', fontweight='bold')
axes[1].set_ylabel('Avg Sales')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.tight_layout()
# plt.show()

"""
### 3.6 — Competition Distance vs Sales
"""

df_eda['CompetitionDistance'] = df_eda['CompetitionDistance'].fillna(df_eda['CompetitionDistance'].median())

# Bin competition distance
bins = [0, 500, 1000, 2000, 5000, 10000, 80000]
labels = ['<500m','500m-1km','1-2km','2-5km','5-10km','>10km']
df_eda['CompBin'] = pd.cut(df_eda['CompetitionDistance'], bins=bins, labels=labels)
comp_avg = df_eda.groupby('CompBin', observed=True)['Sales'].mean()

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(comp_avg.index, comp_avg.values, color='#5b8dd9', edgecolor='white', width=0.6)
for bar, val in zip(bars, comp_avg.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{val:,.0f}', ha='center', fontsize=10)
ax.set_title('Avg Sales by Competition Distance', fontweight='bold')
ax.set_xlabel('Distance to Nearest Competitor')
ax.set_ylabel('Avg Sales')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
# plt.show()

print('\n📌 Insight: Stores closer to competition tend to have slightly lower sales on average.')
print('   Isolated stores (>10km) may serve as the primary option, boosting footfall.')

"""
### 3.7 — Correlation Heatmap
"""

numeric_cols = ['Sales','Customers','DayOfWeek','Promo','SchoolHoliday',
                'Promo2','CompetitionDistance']
corr = df_eda[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix — Numeric Features', fontweight='bold', pad=12)
plt.tight_layout()
# plt.show()

print('📌 Key finding: Customers has the highest correlation with Sales (expected).')
print('   NOTE: Customers will NOT be used as a feature — it is not available at prediction time.')

"""
---
## ⚙️ Section 4: Data Preprocessing & Feature Engineering
"""

"""
### 4.1 — Merge Datasets
"""

# Start fresh from raw files
df    = pd.read_csv('Rossmann Stores Data.csv', low_memory=False)
store = pd.read_csv('store.csv')

# Step 1: Parse date
df['Date'] = pd.to_datetime(df['Date'])

# Step 2: Remove closed-store records (Sales = 0, no prediction needed)
print(f'Before filtering closed stores : {len(df):,} rows')
df = df[df['Open'] == 1].copy()
print(f'After filtering closed stores  : {len(df):,} rows')
print(f'Removed                        : {1017209 - len(df):,} rows')

# Step 3: Merge with store metadata
df = df.merge(store, on='Store', how='left')
print(f'After merge with store.csv     : {df.shape}')

"""
### 4.2 — Handle Missing Values
"""

# CompetitionDistance — 3 nulls — impute with median
comp_median = df['CompetitionDistance'].median()
df['CompetitionDistance'] = df['CompetitionDistance'].fillna(comp_median)
print(f'CompetitionDistance median fill: {comp_median:,.0f} metres')

# Promo2 related — structural nulls (Promo2 = 0 stores), fill with 0
df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
df['CompetitionOpenSinceYear']  = df['CompetitionOpenSinceYear'].fillna(0)
df['Promo2SinceWeek']           = df['Promo2SinceWeek'].fillna(0)
df['Promo2SinceYear']           = df['Promo2SinceYear'].fillna(0)
df['PromoInterval']             = df['PromoInterval'].fillna('None')

print(f'Missing values remaining: {df.isnull().sum().sum()}')

"""
### 4.3 — Feature Engineering
"""

# ── Temporal Features ────────────────────────────────────────────────
df['Year']       = df['Date'].dt.year
df['Month']      = df['Date'].dt.month
df['Day']        = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsWeekend']  = (df['DayOfWeek'] >= 6).astype(int)

# Month position signals
df['IsMonthStart'] = (df['Day'] <= 5).astype(int)
df['IsMonthEnd']   = (df['Day'] >= 25).astype(int)

# Quarter
df['Quarter'] = df['Date'].dt.quarter


# ── Competition Age Feature ───────────────────────────────────────────
df['CompetitionOpenMonths'] = (
    12 * (df['Year'] - df['CompetitionOpenSinceYear']) +
    (df['Month'] - df['CompetitionOpenSinceMonth'])
).clip(lower=0)


# ── Promo2 Active Feature (FIXED) ─────────────────────────────────────
month_abbr_map = {
    'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
    'Jul':7, 'Aug':8, 'Sep':9, 'Sept':9, 'Oct':10, 'Nov':11, 'Dec':12
}

def is_promo2_active(row):
    if row['Promo2'] == 0:
        return 0

    if pd.isna(row['PromoInterval']) or row['PromoInterval'] == 'None':
        return 0

    try:
        promo_months = [month_abbr_map[m.strip()] for m in row['PromoInterval'].split(',')]
        return 1 if row['Month'] in promo_months else 0
    except KeyError:
        return 0  # fallback safety


df['IsPromo2Active'] = df.apply(is_promo2_active, axis=1)


# ── Encode Categoricals ───────────────────────────────────────────────
df['StateHoliday'] = df['StateHoliday'].map({'0':0, 'a':1, 'b':2, 'c':3}).fillna(0).astype(int)
df['StoreType']    = df['StoreType'].map({'a':0, 'b':1, 'c':2, 'd':3})
df['Assortment']   = df['Assortment'].map({'a':0, 'b':1, 'c':2})


print('✅ Feature engineering complete')
print(f'Total features created successfully')
print(f'Dataset shape: {df.shape}')

# Summary of all engineered features
feature_summary = pd.DataFrame({
    'Feature': [
        'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter',
        'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
        'CompetitionOpenMonths', 'IsPromo2Active'
    ],
    'Category': [
        'Temporal', 'Temporal', 'Temporal', 'Temporal', 'Temporal',
        'Temporal', 'Temporal', 'Temporal',
        'Competition', 'Promotion'
    ],
    'Description': [
        'Calendar year extracted from Date',
        'Calendar month (1–12)',
        'Day of month',
        'ISO week number',
        'Quarter (1–4)',
        '1 if DayOfWeek ≥ 6',
        '1 if Day ≤ 5 (payday effect)',
        '1 if Day ≥ 25 (month-end surge)',
        'Months since nearest competitor opened',
        '1 if Promo2 is active in current month'
    ]
})
print(feature_summary.to_string(index=False))

"""
---
## 🤖 Section 5: Model Building
"""

"""
### 5.1 — Define Features & Target
"""

FEATURES = [
    # Store identity
    'Store',
    # Time-based
    'DayOfWeek', 'Year', 'Month', 'Day', 'WeekOfYear', 'Quarter',
    'IsWeekend', 'IsMonthStart', 'IsMonthEnd',
    # Business events
    'Promo', 'StateHoliday', 'SchoolHoliday',
    # Store attributes
    'StoreType', 'Assortment',
    # Competition
    'CompetitionDistance', 'CompetitionOpenMonths',
    # Promo2
    'Promo2', 'IsPromo2Active'
]

TARGET = 'Sales'

X = df[FEATURES]
y = df[TARGET]

print(f'Features : {len(FEATURES)}')
print(f'X shape  : {X.shape}')
print(f'y shape  : {y.shape}')
print(f'\nFeature list:')
for i, f in enumerate(FEATURES, 1):
    print(f'  {i:2d}. {f}')

"""
### 5.2 — Train-Test Split
"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f'Training set : {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X)*100:.0f}%)')
print(f'Test set     : {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X)*100:.0f}%)')
print()
print(f'Train Sales  — mean: {y_train.mean():,.0f} | std: {y_train.std():,.0f}')
print(f'Test  Sales  — mean: {y_test.mean():,.0f} | std: {y_test.std():,.0f}')

"""
METRIC FUNCTION
"""

def evaluate(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    mask = y_true > 0
    rmspe = np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

    print('-'*40)
    print(f'  {label}')
    print('-'*40)
    print(f'  RMSE  : {rmse:>10,.2f}')
    print(f'  MAE   : {mae:>10,.2f}')
    print(f'  R²    : {r2:>10.4f}')
    print(f'  RMSPE : {rmspe:>10.4f}  ({rmspe*100:.2f}%)')

    return rmse, mae, r2, rmspe

"""
 BASELINE MODEL (LINEAR REGRESSION)
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import numpy as np
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

evaluate(y_test, y_pred_lr, "Linear Regression (Baseline)")


"""
RANDOM FOREST MODEL
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import joblib
import os
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

evaluate(y_test, y_pred_rf, "Random Forest")


"""
### 5.3 — Train XGBoost Model
"""

model = xgb.XGBRegressor(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.1,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    eval_metric       = 'rmse'
)

print('🚀 Training XGBoost model...')
model.fit(
    X_train, y_train,
    eval_set        = [(X_train, y_train), (X_test, y_test)],
    verbose         = 100
)
print('\n✅ Training complete!')

"""
---
## 📊 Section 6: Model Evaluation
"""

"""
### 6.1 — Performance Metrics
"""

y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_metrics = evaluate(y_train, y_pred_train, 'TRAIN SET')
test_metrics  = evaluate(y_test,  y_pred_test,  'TEST SET')


"""
### 6.2 — Training History (Learning Curves)
"""

results = model.evals_result()

train_rmse = results['validation_0']['rmse']
test_rmse  = results['validation_1']['rmse']

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(train_rmse, label='Train RMSE', linewidth=2)
ax.plot(test_rmse, label='Test RMSE', linewidth=2)

best_iter = np.argmin(test_rmse)

ax.axvline(best_iter, linestyle='--', linewidth=1,
           label=f'Best iteration: {best_iter}')

ax.set_xlabel('Boosting Rounds')
ax.set_ylabel('RMSE')
ax.set_title('XGBoost Learning Curves', fontweight='bold')
ax.legend()

plt.tight_layout()
# plt.show()


"""
### 6.3 — Actual vs Predicted
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sample_idx = np.random.choice(len(y_test), size=5000, replace=False)

y_sample = np.array(y_test)[sample_idx]
y_pred_sample = y_pred_test[sample_idx]

axes[0].scatter(y_sample, y_pred_sample, alpha=0.3, s=8)

max_val = max(y_sample.max(), y_pred_sample.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5)

axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].set_title(f'Actual vs Predicted\nR² = {test_metrics[2]:.4f}', fontweight='bold')


# Residuals
residuals = np.array(y_test) - y_pred_test

axes[1].hist(residuals, bins=100)
axes[1].axvline(0, linewidth=1.5, linestyle='--')

axes[1].set_title(f'Residuals Distribution\nMAE = {test_metrics[1]:,.2f}', fontweight='bold')

plt.tight_layout()
# plt.show()

"""
### 6.4 — Feature Importance
"""

FEATURES = X_train.columns

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=True)

plt.figure(figsize=(10, 7))
importance.plot(kind='barh')
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance Score')
# plt.show()


print('\n📌 Top Features:')
print(importance.sort_values(ascending=False).head(5))

"""
---
## 💾 Section 7: Save Model & Artifacts
"""

import os
os.makedirs('model_artifacts', exist_ok=True)

joblib.dump(model, 'model_artifacts/xgb_rossmann_model.pkl')
joblib.dump(FEATURES, 'model_artifacts/feature_list.pkl')

# Optional safe fallback if variable not defined
if 'comp_median' in globals():
    joblib.dump(comp_median, 'model_artifacts/comp_distance_median.pkl')

print('\n✅ Artifacts saved:')
for f in os.listdir('model_artifacts'):
    size = os.path.getsize(f'model_artifacts/{f}')
    print(f'   {f:<45} {size/1024:>6.1f} KB')

"""
---
## 🔮 Section 8: Prediction Function
"""

def predict_sales(store_id, date_str, promo, state_holiday,
                  school_holiday, store_type, assortment,
                  competition_distance, promo2=0, promo_interval='None'):

    import pandas as pd

    dt = pd.to_datetime(date_str)

    # ── Robust month mapping (FIXED) ──
    month_abbr_map = {
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
        'Jul':7,'Aug':8,'Sep':9,'Sept':9,'Oct':10,'Nov':11,'Dec':12
    }

    # ── Promo2 active check (FIXED) ──
    is_promo2_active = 0
    if promo2 == 1 and promo_interval not in ['None', None]:
        try:
            promo_months = [
                month_abbr_map[m.strip()]
                for m in promo_interval.split(',')
            ]
            is_promo2_active = 1 if dt.month in promo_months else 0
        except KeyError:
            is_promo2_active = 0  # fallback safety

    # ── Handle missing competition distance (FIXED) ──
    if competition_distance is None:
        competition_distance = comp_median if 'comp_median' in globals() else 0

    # ── Build feature row ──
    row = {
        'Store'                 : store_id,
        'DayOfWeek'             : dt.dayofweek + 1,
        'Year'                  : dt.year,
        'Month'                 : dt.month,
        'Day'                   : dt.day,
        'WeekOfYear'            : int(dt.isocalendar().week),
        'Quarter'               : dt.quarter,
        'IsWeekend'             : int(dt.dayofweek >= 5),
        'IsMonthStart'          : int(dt.day <= 5),
        'IsMonthEnd'            : int(dt.day >= 25),
        'Promo'                 : promo,
        'StateHoliday'          : state_holiday,
        'SchoolHoliday'         : school_holiday,
        'StoreType'             : store_type,
        'Assortment'            : assortment,
        'CompetitionDistance'   : competition_distance,
        'CompetitionOpenMonths' : 0,
        'Promo2'                : promo2,
        'IsPromo2Active'        : is_promo2_active
    }

    # Ensure correct feature order
    X_input = pd.DataFrame([row])[FEATURES]

    # Prediction
    prediction = model.predict(X_input)[0]

    return max(0, round(float(prediction), 2))

"""
---
## 🌐 Section 9: FastAPI Deployment Code

> Save this cell's content as `api.py` and run: `uvicorn api:app --reload`
"""

fastapi_code = '''
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
    store_id             : int   = Field(..., ge=1, le=1115, example=1)
    date                 : str   = Field(..., example="2025-06-15")
    promo                : int   = Field(..., ge=0, le=1, example=1)
    state_holiday        : int   = Field(0, ge=0, le=3, example=0)
    school_holiday       : int   = Field(0, ge=0, le=1, example=0)
    store_type           : int   = Field(..., ge=0, le=3, example=0)
    assortment           : int   = Field(..., ge=0, le=2, example=0)
    competition_distance : float = Field(..., gt=0, example=1270.0)
    promo2               : int   = Field(0, ge=0, le=1, example=0)
    promo_interval       : str   = Field("None", example="Jan,Apr,Jul,Oct")


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
        promo_months = [month_abbr_map[m] for m in req.promo_interval.split(",")]
        is_promo2_active = 1 if dt.month in promo_months else 0

    row = {
        "Store"                 : req.store_id,
        "DayOfWeek"             : dt.dayofweek + 1,
        "Year"                  : dt.year,
        "Month"                 : dt.month,
        "Day"                   : dt.day,
        "WeekOfYear"            : dt.isocalendar().week,
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
'''

with open('api.py', 'w') as f:
    f.write(fastapi_code)

print('✅ api.py saved')
print('   Run with: uvicorn api:app --reload --port 8000')
print('   Swagger UI: http://localhost:8000/docs')

"""
---
## 🖥️ Section 10: Streamlit Dashboard Code

> Save this cell's content as `app.py` and run: `streamlit run app.py`
"""

streamlit_code = '''
# ────────────────────────────────────────────────
# app.py  —  Rossmann Sales Prediction Dashboard
# Run: streamlit run app.py
# ────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ─── Page config ────────────────────────────────
st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="🛒",
    layout="wide"
)

# ─── Load model ─────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model_artifacts/xgb_rossmann_model.pkl")
    features = joblib.load("model_artifacts/feature_list.pkl")
    return model, features

model, FEATURES = load_model()

# ─── Header ──────────────────────────────────────
st.title("🛒 Rossmann Store Sales Forecast")
st.markdown("Predict daily sales for any store up to **6 weeks in advance** using XGBoost.")
st.divider()

# ─── Sidebar inputs ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Store Configuration")

    store_id = st.number_input("Store ID", min_value=1, max_value=1115, value=1)

    store_type_map  = {"Type A (602 stores)":0, "Type B (17 stores)":1,
                       "Type C (148 stores)":2, "Type D (348 stores)":3}
    assortment_map  = {"Basic":0, "Extra":1, "Extended":2}

    store_type  = store_type_map[st.selectbox("Store Type", list(store_type_map.keys()))]
    assortment  = assortment_map[st.selectbox("Assortment", list(assortment_map.keys()))]
    comp_dist   = st.number_input("Competition Distance (m)", min_value=20, max_value=75860, value=1270)

    st.header("📅 Forecast Settings")
    start_date    = st.date_input("Start Date", value=date.today())
    forecast_days = st.slider("Forecast Horizon (days)", 1, 42, 14)
    promo         = st.checkbox("Running Promotion (Promo)", value=True)
    promo2        = st.checkbox("Enrolled in Promo2", value=False)
    school_holiday = st.checkbox("School Holiday", value=False)

# ─── Prediction logic ────────────────────────────
month_abbr_map = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}

rows = []
for i in range(forecast_days):
    dt = pd.to_datetime(start_date) + timedelta(days=i)
    is_promo2 = 0
    if promo2:
        promo_months = [month_abbr_map[m] for m in "Jan,Apr,Jul,Oct".split(",")]
        is_promo2 = 1 if dt.month in promo_months else 0
    rows.append({
        "Store": store_id, "DayOfWeek": dt.dayofweek + 1,
        "Year": dt.year, "Month": dt.month, "Day": dt.day,
        "WeekOfYear": dt.isocalendar().week, "Quarter": dt.quarter,
        "IsWeekend": int(dt.dayofweek >= 5),
        "IsMonthStart": int(dt.day <= 5), "IsMonthEnd": int(dt.day >= 25),
        "Promo": int(promo), "StateHoliday": 0,
        "SchoolHoliday": int(school_holiday),
        "StoreType": store_type, "Assortment": assortment,
        "CompetitionDistance": comp_dist, "CompetitionOpenMonths": 0,
        "Promo2": int(promo2), "IsPromo2Active": is_promo2,
        "Date": dt.strftime("%Y-%m-%d")
    })

forecast_df = pd.DataFrame(rows)
X_forecast  = forecast_df[FEATURES]
forecast_df["Predicted_Sales"] = np.maximum(0, model.predict(X_forecast))

# ─── KPI cards ───────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Forecast",    f"{forecast_df['Predicted_Sales'].sum():,.0f}")
c2.metric("Daily Average",     f"{forecast_df['Predicted_Sales'].mean():,.0f}")
c3.metric("Peak Day",          forecast_df.loc[forecast_df['Predicted_Sales'].idxmax(), 'Date'])
c4.metric("Peak Sales",        f"{forecast_df['Predicted_Sales'].max():,.0f}")

st.divider()

# ─── Line chart ──────────────────────────────────
st.subheader(f"📈 {forecast_days}-Day Sales Forecast — Store {store_id}")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(forecast_df["Date"], forecast_df["Predicted_Sales"],
        marker="o", linewidth=2, markersize=5, color="#5b8dd9")
ax.fill_between(forecast_df["Date"], forecast_df["Predicted_Sales"], alpha=0.12, color="#5b8dd9")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Sales")
plt.xticks(rotation=45, fontsize=8)
ax.spines[["top","right"]].set_visible(False)
st.pyplot(fig)

# ─── Data table ──────────────────────────────────
st.subheader("📋 Forecast Details")
display_df = forecast_df[["Date","DayOfWeek","Promo","IsWeekend","Predicted_Sales"]].copy()
display_df.columns = ["Date","Day of Week","Promo","Is Weekend","Predicted Sales"]
display_df["Predicted Sales"] = display_df["Predicted Sales"].map("{:,.0f}".format)
st.dataframe(display_df, use_container_width=True)

# ─── Download ────────────────────────────────────
csv = forecast_df[["Date","Predicted_Sales"]].to_csv(index=False)
st.download_button("⬇️ Download Forecast CSV", data=csv,
                   file_name=f"store_{store_id}_forecast.csv", mime="text/csv")
'''

with open('app.py', 'w') as f:
    f.write(streamlit_code)

print('✅ app.py saved')
print('   Run with: streamlit run app.py')

"""
---
## 📋 Section 11: Model Summary
"""

print('='*60)
print('  ROSSMANN RETAIL SALES PREDICTION — MODEL SUMMARY')
print('='*60)
print()
print('  DATASET')
print(f'  ├─ Total records (raw)       : 1,017,209')
print(f'  ├─ Records after filtering   :   844,392  (Open stores only)')
print(f'  ├─ Stores                    : 1,115')
print(f'  └─ Date range                : Jan 2013 – Jul 2015')
print()
print('  FEATURES')
print(f'  ├─ Total features used       : {len(FEATURES)}')
print(f'  ├─ Temporal features         : 9')
print(f'  ├─ Business event features   : 3  (Promo, Holiday, SchoolHoliday)')
print(f'  ├─ Store attribute features  : 2  (StoreType, Assortment)')
print(f'  └─ Competition / Promo2      : 4')
print()
print('  MODEL — XGBoost Regressor')
print(f'  ├─ n_estimators              : 500')
print(f'  ├─ max_depth                 : 6')
print(f'  ├─ learning_rate             : 0.1')
print(f'  └─ subsample / colsample     : 0.8 / 0.8')
print()
print('  PERFORMANCE (Test Set — 20%)')
print(f'  ├─ RMSE   :  1,017.55')
print(f'  ├─ MAE    :    714.27')
print(f'  ├─ R²     :      0.8926  (model explains 89.26% of variance)')
print(f'  └─ RMSPE  : ~14–16%     (Kaggle competition benchmark)')
print()
print('  DEPLOYMENT')
print(f'  ├─ REST API   : FastAPI (api.py) → uvicorn api:app --reload')
print(f'  └─ Dashboard  : Streamlit (app.py) → streamlit run app.py')
print()
print('='*60)


# ────────────────────────────────────────────────────────────
# app.py  —  Rossmann Sales Prediction Dashboard + AI Chatbot
# Run: streamlit run app.py
# ────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import date, timedelta

# ─── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="🛒",
    layout="wide"
)

# ─── Custom CSS for premium look ─────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 16px; border: 1px solid #2a2a4a; }
    .stMetric label { color: #8892b0 !important; font-size: 0.85rem !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #e6f1ff !important;
                font-size: 1.8rem !important; font-weight: 700 !important; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a192f 0%, #112240 100%); }
    .chat-msg { background: #112240; border-radius: 10px; padding: 12px 16px;
                margin: 6px 0; border-left: 3px solid #64ffda; }
</style>
""", unsafe_allow_html=True)


# ─── Load model ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model_artifacts/xgb_rossmann_model.pkl")
    features = joblib.load("model_artifacts/feature_list.pkl")
    return model, features

model, FEATURES = load_model()


# ─── Header ──────────────────────────────────────────────────
st.title("🛒 Rossmann Store Sales Forecast")
st.markdown("Predict daily sales for any store up to **6 weeks in advance** using XGBoost.")
st.divider()


# ─── Sidebar inputs ─────────────────────────────────────────
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


# ─── Prediction logic ───────────────────────────────────────
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
        "WeekOfYear": int(dt.isocalendar().week), "Quarter": dt.quarter,
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


# ─── KPI cards ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Forecast",    f"EUR {forecast_df['Predicted_Sales'].sum():,.0f}")
c2.metric("Daily Average",     f"EUR {forecast_df['Predicted_Sales'].mean():,.0f}")
c3.metric("Peak Day",          forecast_df.loc[forecast_df['Predicted_Sales'].idxmax(), 'Date'])
c4.metric("Peak Sales",        f"EUR {forecast_df['Predicted_Sales'].max():,.0f}")

st.divider()


# ─── Line chart ──────────────────────────────────────────────
st.subheader(f"📈 {forecast_days}-Day Sales Forecast — Store {store_id}")
fig, ax = plt.subplots(figsize=(12, 4))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

ax.plot(forecast_df["Date"], forecast_df["Predicted_Sales"],
        marker="o", linewidth=2, markersize=5, color="#64ffda")
ax.fill_between(range(len(forecast_df)), forecast_df["Predicted_Sales"],
                alpha=0.12, color="#64ffda")
ax.set_xlabel("Date", color='#8892b0')
ax.set_ylabel("Predicted Sales (EUR)", color='#8892b0')
ax.tick_params(colors='#8892b0')
plt.xticks(rotation=45, fontsize=8)
ax.spines[["top","right"]].set_visible(False)
ax.spines[["bottom","left"]].set_color('#2a2a4a')
st.pyplot(fig)


# ─── What-If Analysis ───────────────────────────────────────
st.divider()
st.subheader("🔬 What-If Scenario Analysis")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Scenario A: Promo Impact**")
    # Predict with promo OFF
    rows_no_promo = []
    for i in range(forecast_days):
        dt = pd.to_datetime(start_date) + timedelta(days=i)
        rows_no_promo.append({
            "Store": store_id, "DayOfWeek": dt.dayofweek + 1,
            "Year": dt.year, "Month": dt.month, "Day": dt.day,
            "WeekOfYear": int(dt.isocalendar().week), "Quarter": dt.quarter,
            "IsWeekend": int(dt.dayofweek >= 5),
            "IsMonthStart": int(dt.day <= 5), "IsMonthEnd": int(dt.day >= 25),
            "Promo": 0, "StateHoliday": 0,
            "SchoolHoliday": int(school_holiday),
            "StoreType": store_type, "Assortment": assortment,
            "CompetitionDistance": comp_dist, "CompetitionOpenMonths": 0,
            "Promo2": int(promo2), "IsPromo2Active": 0,
            "Date": dt.strftime("%Y-%m-%d")
        })
    df_no_promo = pd.DataFrame(rows_no_promo)
    df_no_promo["Predicted_Sales"] = np.maximum(0, model.predict(df_no_promo[FEATURES]))

    # Predict with promo ON
    rows_with_promo = []
    for i in range(forecast_days):
        dt = pd.to_datetime(start_date) + timedelta(days=i)
        rows_with_promo.append({
            "Store": store_id, "DayOfWeek": dt.dayofweek + 1,
            "Year": dt.year, "Month": dt.month, "Day": dt.day,
            "WeekOfYear": int(dt.isocalendar().week), "Quarter": dt.quarter,
            "IsWeekend": int(dt.dayofweek >= 5),
            "IsMonthStart": int(dt.day <= 5), "IsMonthEnd": int(dt.day >= 25),
            "Promo": 1, "StateHoliday": 0,
            "SchoolHoliday": int(school_holiday),
            "StoreType": store_type, "Assortment": assortment,
            "CompetitionDistance": comp_dist, "CompetitionOpenMonths": 0,
            "Promo2": int(promo2), "IsPromo2Active": 0,
            "Date": dt.strftime("%Y-%m-%d")
        })
    df_with_promo = pd.DataFrame(rows_with_promo)
    df_with_promo["Predicted_Sales"] = np.maximum(0, model.predict(df_with_promo[FEATURES]))

    delta = df_with_promo["Predicted_Sales"].sum() - df_no_promo["Predicted_Sales"].sum()
    pct   = (delta / df_no_promo["Predicted_Sales"].sum()) * 100 if df_no_promo["Predicted_Sales"].sum() > 0 else 0

    st.metric("Promo Uplift (Total)", f"EUR {delta:,.0f}", f"{pct:+.1f}%")

with col_b:
    st.markdown("**Scenario B: Competition Distance Sensitivity**")
    distances = [500, 1000, 2000, 5000, 10000]
    results = {}
    for d in distances:
        temp_rows = []
        for i in range(forecast_days):
            dt = pd.to_datetime(start_date) + timedelta(days=i)
            temp_rows.append({
                "Store": store_id, "DayOfWeek": dt.dayofweek + 1,
                "Year": dt.year, "Month": dt.month, "Day": dt.day,
                "WeekOfYear": int(dt.isocalendar().week), "Quarter": dt.quarter,
                "IsWeekend": int(dt.dayofweek >= 5),
                "IsMonthStart": int(dt.day <= 5), "IsMonthEnd": int(dt.day >= 25),
                "Promo": int(promo), "StateHoliday": 0,
                "SchoolHoliday": int(school_holiday),
                "StoreType": store_type, "Assortment": assortment,
                "CompetitionDistance": d, "CompetitionOpenMonths": 0,
                "Promo2": int(promo2), "IsPromo2Active": 0,
                "Date": dt.strftime("%Y-%m-%d")
            })
        temp_df = pd.DataFrame(temp_rows)
        results[f"{d}m"] = np.maximum(0, model.predict(temp_df[FEATURES])).mean()

    comp_df = pd.DataFrame({"Distance": list(results.keys()), "Avg Daily Sales": list(results.values())})
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ─── Data table ──────────────────────────────────────────────
st.divider()
st.subheader("📋 Forecast Details")
display_df = forecast_df[["Date","DayOfWeek","Promo","IsWeekend","Predicted_Sales"]].copy()
display_df.columns = ["Date","Day of Week","Promo","Is Weekend","Predicted Sales (EUR)"]
display_df["Predicted Sales (EUR)"] = display_df["Predicted Sales (EUR)"].map("{:,.0f}".format)
st.dataframe(display_df, use_container_width=True, hide_index=True)


# ─── Download ────────────────────────────────────────────────
csv = forecast_df[["Date","Predicted_Sales"]].to_csv(index=False)
st.download_button("⬇️ Download Forecast CSV", data=csv,
                   file_name=f"store_{store_id}_forecast.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════
#  SECTION: AI CHATBOT — Rossmann Sales Assistant
# ═══════════════════════════════════════════════════════════════
st.divider()
st.subheader("💬 Rossmann AI Sales Assistant")
st.markdown(
    "_Ask me about the forecast above! Try: 'Which day has peak sales?', "
    "'What is the total forecast?', 'Compare promo vs no promo'._"
)

# ─── Session state for chat history ─────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hello! I'm your Rossmann Sales Assistant. "
                    "Ask me anything about the current forecast, promo impact, or store performance."}
    ]

# ─── Display chat history ───────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Chat input handler ─────────────────────────────────────
if user_input := st.chat_input("E.g., Which day has the highest sales?"):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ─── Intent-based response engine ────────────────────────
    q = user_input.lower()
    response = ""

    try:
        # --- Peak / highest sales ---
        if any(w in q for w in ["highest", "peak", "max", "best", "top"]):
            peak_idx = forecast_df['Predicted_Sales'].idxmax()
            peak_row = forecast_df.loc[peak_idx]
            response = (
                f"📈 **Peak Sales Day:** {peak_row['Date']}\n\n"
                f"Predicted sales: **EUR {peak_row['Predicted_Sales']:,.0f}**\n\n"
                f"This is a **{'weekday' if peak_row['IsWeekend'] == 0 else 'weekend'}** "
                f"{'with' if peak_row['Promo'] else 'without'} active promotion."
            )

        # --- Lowest / minimum sales ---
        elif any(w in q for w in ["lowest", "min", "worst", "bottom"]):
            min_idx = forecast_df['Predicted_Sales'].idxmin()
            min_row = forecast_df.loc[min_idx]
            response = (
                f"📉 **Lowest Sales Day:** {min_row['Date']}\n\n"
                f"Predicted sales: **EUR {min_row['Predicted_Sales']:,.0f}**\n\n"
                f"💡 *Recommendation:* Consider running a promotion or flash sale on this day."
            )

        # --- Average / mean ---
        elif any(w in q for w in ["average", "mean", "daily"]):
            avg = forecast_df['Predicted_Sales'].mean()
            std = forecast_df['Predicted_Sales'].std()
            response = (
                f"📊 **Average Daily Sales:** EUR {avg:,.0f}\n\n"
                f"Standard deviation: EUR {std:,.0f}\n\n"
                f"This means daily sales are expected to fluctuate between "
                f"**EUR {max(0, avg - std):,.0f}** and **EUR {avg + std:,.0f}** on most days."
            )

        # --- Total / sum ---
        elif any(w in q for w in ["total", "sum", "overall", "cumulative"]):
            total = forecast_df['Predicted_Sales'].sum()
            days  = len(forecast_df)
            response = (
                f"💰 **Total Forecasted Revenue:** EUR {total:,.0f}\n\n"
                f"Over **{days} days** for Store {store_id}.\n\n"
                f"Daily average: EUR {total / days:,.0f}"
            )

        # --- Promo impact ---
        elif any(w in q for w in ["promo", "promotion", "uplift", "discount"]):
            with_p  = df_with_promo["Predicted_Sales"].sum()
            no_p    = df_no_promo["Predicted_Sales"].sum()
            delta_p = with_p - no_p
            pct_p   = (delta_p / no_p) * 100 if no_p > 0 else 0
            response = (
                f"🏷️ **Promotion Impact Analysis:**\n\n"
                f"| Scenario | Total Sales |\n|---|---|\n"
                f"| Without Promo | EUR {no_p:,.0f} |\n"
                f"| With Promo | EUR {with_p:,.0f} |\n"
                f"| **Uplift** | **EUR {delta_p:,.0f} ({pct_p:+.1f}%)** |\n\n"
                f"💡 *Recommendation:* {'Promotions show strong ROI. Keep them running.' if pct_p > 10 else 'Promotion impact is moderate. Consider targeted promos instead of blanket ones.'}"
            )

        # --- Weekend ---
        elif any(w in q for w in ["weekend", "saturday", "sunday"]):
            weekday_avg = forecast_df[forecast_df['IsWeekend'] == 0]['Predicted_Sales'].mean()
            weekend_avg = forecast_df[forecast_df['IsWeekend'] == 1]['Predicted_Sales'].mean()
            if pd.isna(weekend_avg):
                response = "No weekend days found in the current forecast horizon."
            else:
                response = (
                    f"📅 **Weekday vs Weekend Sales:**\n\n"
                    f"| Period | Avg Daily Sales |\n|---|---|\n"
                    f"| Weekdays | EUR {weekday_avg:,.0f} |\n"
                    f"| Weekends | EUR {weekend_avg:,.0f} |\n\n"
                    f"{'Weekends show higher traffic.' if weekend_avg > weekday_avg else 'Weekdays drive more revenue — consider weekend promotions.'}"
                )

        # --- Store info ---
        elif any(w in q for w in ["store", "info", "configuration", "setup"]):
            type_names = {0: "Type A", 1: "Type B", 2: "Type C", 3: "Type D"}
            assort_names = {0: "Basic", 1: "Extra", 2: "Extended"}
            response = (
                f"🏪 **Store {store_id} Configuration:**\n\n"
                f"- Store Type: {type_names.get(store_type, 'Unknown')}\n"
                f"- Assortment: {assort_names.get(assortment, 'Unknown')}\n"
                f"- Competition Distance: {comp_dist:,}m\n"
                f"- Promo Active: {'Yes' if promo else 'No'}\n"
                f"- Promo2 Enrolled: {'Yes' if promo2 else 'No'}"
            )

        # --- Help ---
        elif any(w in q for w in ["help", "what can", "how to", "commands"]):
            response = (
                "🤖 **I can help with:**\n\n"
                "- **'peak sales'** — Find the highest predicted day\n"
                "- **'lowest sales'** — Find the weakest day\n"
                "- **'average sales'** — Get daily averages & range\n"
                "- **'total forecast'** — Total revenue projection\n"
                "- **'promo impact'** — Compare promo vs no-promo\n"
                "- **'weekend analysis'** — Weekday vs weekend comparison\n"
                "- **'store info'** — Current store configuration\n"
            )

        # --- Fallback ---
        else:
            response = (
                "I couldn't match that to a known query. Try:\n\n"
                "- *'What is the peak sales day?'*\n"
                "- *'Show me the promo impact'*\n"
                "- *'What is the total forecast?'*\n"
                "- Type **'help'** for all available commands."
            )

    except Exception as e:
        response = f"⚠️ Error processing your query: {str(e)}"

    # Append and display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

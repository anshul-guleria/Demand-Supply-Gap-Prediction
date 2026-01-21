import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="LSTM Demand Forecasting Dashboard",
    layout="wide"
)

BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models", "LSTM")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "LSTM")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load Artifacts
@st.cache_resource
def load_lstm_artifacts():
    model = load_model(os.path.join(MODELS_DIR, "model.keras"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "demand_scaler.pkl"))
    prod_enc = joblib.load(os.path.join(MODELS_DIR, "product_encoder.pkl"))
    cat_enc = joblib.load(os.path.join(MODELS_DIR, "category_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
        config = json.load(f)

    return model, scaler, prod_enc, cat_enc, config

@st.cache_data
def load_metrics():
    with open(os.path.join(BASE_DIR, "outputs","lstm_performance_metrics.json")) as f:
        return json.load(f)["metrics"]


# Load Data
model, scaler, prod_enc, cat_enc, config = load_lstm_artifacts()
metrics = load_metrics()

products = ['E001', 'E002','E003','E004','E005', 'F001', 'F002','F003','F004','F005',
               'S001', 'S002','S003','S004','S005']
categories = ['Electronics', 'Fashion','Sweets & Grocery']

# Dashboard Header
st.title("📈 LSTM Demand Forecasting & Inventory Risk Dashboard")

# Metrics Section
st.subheader("📊 Model Performance (Test Set)")

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", round(metrics["mae"], 2))
m2.metric("RMSE", round(metrics["rmse"], 2))
m3.metric("Stockout Accuracy", round(metrics["stockout_accuracy"], 2))
m4.metric("F1 Score", round(metrics["f1-score"], 2))

# Visualizations Section
st.subheader("📉 Evaluation Visualizations")

viz_dir = os.path.join(OUTPUTS_DIR)
col1, col2 = st.columns(2)

with col1:
    st.image(os.path.join(viz_dir, "demand_vs_supply.png"), caption="Demand vs Supply")
    st.image(os.path.join(viz_dir, "demand_supply_gap.png"), caption="Demand & Supply Gap Over Time")
with col2:
    st.image(os.path.join(viz_dir, "stockout_prediction.png"), caption="Stockout Prediction Over Time")


st.subheader("📉 Evaluation Visualizations for last 300 test points")

col1, col2 = st.columns(2)

with col1:
    st.image(os.path.join(viz_dir, "demand_vs_supply_300.png"), caption="Demand vs Supply")
    st.image(os.path.join(viz_dir, "demand_supply_gap_300.png"), caption="Demand & Supply Gap Over Time")
with col2:
    st.image(os.path.join(viz_dir, "stockout_prediction_300.png"), caption="Stockout Prediction Over Time")


# Forecasting Section
st.subheader("🔮 Demand & Gap Forecast")

c1, c2, c3, c4 = st.columns(4)

with c1:
    forecast_date = st.date_input(
        "Forecast Date",
        value=datetime(2025, 10, 15)
    )

    is_festival_manual = st.selectbox(
        "Is the selected date a festival?",
        ["No", "Yes"],
        help="Select 'Yes' if the forecast date is a festival"
    )
    festival_flag = 1 if is_festival_manual == "Yes" else 0

with c2:
    category = st.selectbox("Category", categories)

with c3:
    product = st.selectbox("Product", [p for p in products if p[0]==category[0]])

with c4:
    expected_supply = st.number_input(
        "Expected Supply",
        min_value=0.0,
        value=100.0
    )

# Run Forecast

df=pd.read_csv(os.path.join(DATA_DIR, "demand_supply_data.csv"))

df["Date"] = pd.to_datetime(df["Date"])

#Renaming columns for easier access
df = df.rename(columns={
    "Product_Name": "product_name",
    "Demand": "demand",
    "Date": "date",
    "Available_Supply": "supply",
    "Seasonal_Indicator": "festival_flag"
})

if st.button("🚀 Run Forecast"):
    hist = df[
        (df["product_name"] == product) &
        (df["Category"] == category)
    ].sort_values("date")

    window = config["window_size"]
    features = config["features"]

    if len(hist) < window:
        # print((hist.head()))
        st.error("Not enough historical data for the selected product/category.")
        st.stop()

    # Feature reconstruction
    hist["demand_scaled"] = scaler.transform(hist[["demand"]])

    for lag in [1, 7, 14]:
        hist[f"lag_{lag}"] = hist["demand_scaled"].shift(lag)

    hist["rolling_7"] = hist["demand_scaled"].rolling(7).mean()
    hist["month"] = hist["date"].dt.month
    hist["quarter"] = hist["date"].dt.quarter
    hist["festival_flag"] = festival_flag

    hist = hist.dropna()

    X_time = hist[features].values[-window:]
    X_time = X_time.reshape(1, window, len(features))

    X_prod = prod_enc.transform([product]).reshape(1, 1)
    X_cat = cat_enc.transform([category]).reshape(1, 1)

    pred_scaled = model.predict([X_time, X_prod, X_cat], verbose=0)
    pred_demand = scaler.inverse_transform(pred_scaled)[0][0]

    gap = pred_demand - expected_supply

    risk_type = "Understock" if gap > 0 else "Overstock"
    risk_level = (
        "High" if abs(gap) > 50 else
        "Medium" if abs(gap) > 15 else
        "Low"
    )

    # Output Section
    st.success("Forecast completed successfully")

    o1, o2, o3 = st.columns(3)
    o1.metric("Predicted Demand", round(pred_demand, 2))
    o2.metric("Gap (Demand − Supply)", round(gap, 2))
    o3.metric("Risk Type", risk_type)

    st.markdown(f"### 🚦 Risk Level: **{risk_level}**")


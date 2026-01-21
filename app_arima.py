import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(
    page_title="ARIMA / SARIMA Forecast Dashboard",
    layout="wide"
)


BASE_DIR = os.getcwd()
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

METRIC_FILES = {
    "ARIMA": "./outputs/arima_performance_logs.json",
    "SARIMA": "./outputs/sarima_performance_logs.json"
}

@st.cache_data
def load_metrics(model_type):
    with open(METRIC_FILES[model_type], "r") as f:
        return json.load(f)


def get_model_names(metrics_data):
    return ([m["name"] for m in metrics_data])


def display_metrics(metrics):
    # col1, col2, col3, col4, col5 = st.columns(5)
    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", round(metrics["MAE"], 3))
    col2.metric("RMSE", round(metrics["RMSE"], 3))
    col3.metric("MAPE (%)", round(metrics["MAPE"], 2))
    # col4.metric("AIC", round(metrics["AIC"], 2))
    # col5.metric("Stockout Accuracy", metrics.get("stockout_accuracy", "N/A"))

def load_visualizations(model_type, level, model_name):
    vis_dir = os.path.join(
        OUTPUTS_DIR,
        model_type,
        level
    )
    if not os.path.exists(vis_dir):
        return []
    
    print(vis_dir)

    return [
        os.path.join(vis_dir, f'{model_name}_{model_type}_Forecast.png') 
    ]

def load_model(model_type, level, model_name):
    model_path = os.path.join(
        MODELS_DIR,
        model_type,
        level,
        f"{model_name}_{str.lower(model_type)}_model.pkl"
    )

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    return joblib.load(model_path)

#SideBar
st.sidebar.title("Model Selection")

model_type = st.sidebar.selectbox(
    "Choose Model",
    ["ARIMA", "SARIMA"]
)

level = st.sidebar.selectbox(
    "Forecast Level",
    ["Product", "Category"]
)

metrics_data = load_metrics(model_type)
model_names = ['E001', 'E002','E003','E004','E005', 'F001', 'F002','F003','F004','F005',
               'S001', 'S002','S003','S004','S005']
category_names = ['Electronics', 'Fashion','Sweets & Grocery']

if level=="Product":
    model_name = st.sidebar.selectbox(
    "Select Model ID",
    model_names
    )
    selected_entry = next(
    m for m in metrics_data if m["name"] == model_name
    )
else:
    model_name = st.sidebar.selectbox(
        "Select Model Category",
        category_names
    )
    selected_entry = next(
    m for m in metrics_data if m["category"] == model_name
    )

#Main Dashboard
st.title(f"{model_type} Forecasting Dashboard")


st.subheader("📊 Model Performance Metrics")
display_metrics(selected_entry["metrics"])

#Metrics
with st.expander("Show Full Metrics Table"):
    df = pd.json_normalize(metrics_data)
    st.dataframe(df, use_container_width=True)


#Visualizations
st.subheader("📈 Model Visualizations")

visualizations = load_visualizations(model_type, level, model_name)

if not visualizations:
    st.warning("No visualizations found for this model.")
else:
    for img_path in visualizations:
        image = Image.open(img_path)
        st.image(
            image,
            caption=os.path.basename(img_path),
            use_container_width=True
        )


# Sub-functions for Forecasting
@st.cache_data
def load_festival_calendar():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "festival_data.csv"))

    festival_dates = set()

    for _, row in df.iterrows():
        for year in df.columns[1:]:
            date_str = f"{row[year]} {year}"
            festival_date = pd.to_datetime(
                date_str,
                format="%b %d %Y",
                errors="coerce"
            )
            if pd.notna(festival_date):
                festival_dates.add(festival_date.date())

    return festival_dates

def build_future_dates(start_date, end_date, freq="D"):
    return pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq
    )

def build_festival_exog(future_dates, festival_dates):
    return np.array([
        1 if d.date() in festival_dates else 0
        for d in future_dates
    ]).reshape(-1, 1)


#Forecasting
st.subheader("🔮 Predict Future Values (Date-based)")

enable_forecast = st.checkbox("Enable Date-based Forecast")

if enable_forecast:
    start_date = pd.to_datetime("2025-10-10")

    end_date = st.date_input(
        "Select forecast end date",
        value=pd.to_datetime("2025-10-20")
    )

    if end_date <= start_date.date():
        st.error("End date must be after 10-Oct-2025")

    expected_supply = st.number_input(
        "Expected Available Supply per Day",
        min_value=0.0,
        value=0.0,
        help="Used to calculate expected stockout (Forecast > Supply)"
    )

    

    if st.button("Predict"):
        model = load_model(model_type, level, model_name)

        if model is None:
            st.stop()

        future_dates = build_future_dates(
            start_date=start_date,
            end_date=end_date,
            freq="D"
        )

        steps = len(future_dates)

        # ARIMA & SARIMA
        if model.model.exog is not None:
            festival_dates = load_festival_calendar()
            future_exog = build_festival_exog(
                future_dates,
                festival_dates
            )

            forecast = model.forecast(
                steps=steps,
                exog=future_exog
            )
        else:
            forecast = model.forecast(steps=steps)

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        })

        forecast_df["Expected Supply"] = expected_supply
        forecast_df["Expected Stockout"] = forecast_df["Forecast"] > forecast_df["Expected Supply"]

        forecast_df["Expected Stockout"] = forecast_df["Expected Stockout"].map(
    {True: "Yes", False: "No"})

        st.success("Forecast generated successfully")

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(forecast_df, use_container_width=True)

        with col2:
        # Plot
            fig, ax = plt.subplots()
            ax.plot(
                forecast_df["Date"],
                forecast_df["Forecast"],
                marker="o"
            )
            ax.axhline(
                y=expected_supply,
                linestyle="--",
                label="Expected Supply"
            )

            # Highlight stockout points
            stockout_days = forecast_df[forecast_df["Expected Stockout"] == "Yes"]

            ax.scatter(
                stockout_days["Date"],
                stockout_days["Forecast"],
                label="Expected Stockout",
                zorder=5,
                color="red",
            )

            ax.set_title("Forecast with Expected Stockout")
            ax.set_xlabel("Date")
            ax.set_ylabel("Units")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

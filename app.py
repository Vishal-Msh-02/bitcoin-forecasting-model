import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
import os

# App configuration
st.set_page_config(page_title="BTC Price Forecast", layout="wide")
st.title("📈 Bitcoin Price Forecasting App using Prophet")

# 1. Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("bitcoin_history.csv")

    # Ensure 'Date' column exists and convert it to datetime
    if 'Date' in df.columns:
        df['ds'] = pd.to_datetime(df['Date'])
    else:
        st.error("CSV file must contain a 'Date' column.")
        st.stop()

    # Use 'Close' or 'Adj Close' as target value
    if 'Close' in df.columns:
        df['y'] = df['Close']
    elif 'Adj Close' in df.columns:
        df['y'] = df['Adj Close']
    else:
        st.error("CSV must contain a 'Close' or 'Adj Close' column for target.")
        st.stop()

    return df[['ds', 'y']]

# Load data
df = load_data()

# Show raw data
with st.expander("🔍 View Raw Data"):
    st.dataframe(df.tail())

# 2. Split train/test
train_df = df[:-90]
test_df = df[-90:]

# 3. Fit the Prophet model
model = Prophet()
model.fit(train_df)

# 4. Forecasting next 90 days
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# 5. Forecast plot
st.subheader("📅 Forecast for Next 90 Days")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1, use_container_width=True)

# 6. Components plot
st.subheader("📊 Forecast Components")
fig2 = plot_components_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

# 7. Evaluation
forecast_test = forecast[['ds', 'yhat']].tail(90).reset_index(drop=True)
test_df_eval = test_df.reset_index(drop=True)

mae = mean_absolute_error(test_df_eval['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test_df_eval['y'], forecast_test['yhat']))

st.subheader("📏 Evaluation Metrics (Last 90 Days)")
st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.2f}")

# 8. Forecast CSV download
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button("📥 Download Forecast CSV", data=csv, file_name="btc_forecast.csv", mime="text/csv")

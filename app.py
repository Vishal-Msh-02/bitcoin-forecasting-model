import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go
import os

st.set_page_config(page_title="BTC Price Forecast", layout="wide")
st.title("📈 Bitcoin Price Forecasting App using Prophet")

@st.cache_data
def load_data():
    # Debug: Print current directory contents
    print("Files in current directory:", os.listdir())
    
    # Load CSV file (uploaded to Streamlit Cloud or local)
    df = pd.read_csv("bitcoin_history.csv")
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Rename required columns for Prophet
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})  # Prophet needs 'ds' and 'y'
    
    # Ensure y is numeric
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Drop missing values
    df = df.dropna(subset=['ds', 'y'])
    
    return df

# Load and prepare data
df = load_data()

# Show raw data
with st.expander("🔍 View Raw Data"):
    st.dataframe(df.tail())

# Split data for evaluation
train_df = df[:-90]
test_df = df[-90:]

# Fit the Prophet model
model = Prophet()
model.fit(train_df)

# Make future predictions
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot the forecast
st.subheader("📅 Forecast for Next 90 Days")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1, use_container_width=True)

# Plot forecast components
st.subheader("📊 Forecast Components")
fig2 = plot_components_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

# Evaluate model performance on last 90 days
forecast_test = forecast[['ds', 'yhat']].tail(90).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))

st.subheader("📏 Evaluation Metrics (Last 90 Days)")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")

# Option to download forecast
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button("📥 Download Forecast CSV", data=csv, file_name="btc_forecast.csv", mime="text/csv")

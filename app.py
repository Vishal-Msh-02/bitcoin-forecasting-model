import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import plotly.graph_objs as go

st.set_page_config(page_title="BTC Price Forecast", layout="wide")

st.title("📈 Bitcoin Price Forecasting App using Prophet")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("btc.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close'].astype(str).str.replace(',', '').astype(float)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df

df = load_data()

# Show raw data
with st.expander("🔍 View Raw Data"):
    st.dataframe(df.tail())

# Split data for evaluation
train_df = df[:-90]
test_df = df[-90:]

# Fit model
model = Prophet()
model.fit(train_df)

# Make forecast
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Show forecast
st.subheader("📅 Forecast for Next 90 Days")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1, use_container_width=True)

# Components
st.subheader("📊 Forecast Components")
fig2 = plot_components_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

# Evaluation
forecast_test = forecast[['ds', 'yhat']].tail(90).reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

mae = mean_absolute_error(test_df['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test_df['y'], forecast_test['yhat']))

st.subheader("📏 Evaluation Metrics (Last 90 Days)")
st.write(f"**Mean Absolute Error (MAE):** ${mae:,.2f}")
st.write(f"**Root Mean Squared Error (RMSE):** ${rmse:,.2f}")

# Optional download
csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
st.download_button("📥 Download Forecast CSV", data=csv, file_name="btc_forecast.csv", mime="text/csv")

import streamlit as st
import yfinance as yf
import pandas as pd
from model import train_model, predict_trend
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY ML Dashboard", layout="wide")

st.title("📈 NIFTY 50 – ML Trend Prediction Dashboard")

# Sidebar
period = st.sidebar.selectbox(
    "Select Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

# Load data
@st.cache_data
def load_data(period):
    data = yf.download("^NSEI", period=period, auto_adjust=True)
    return data

data = load_data(period)

# Train model
model, X = train_model(data)

# Prediction
prediction = predict_trend(model, X)

# Display prediction
st.subheader("📊 Tomorrow's Trend Prediction")
if prediction == 1:
    st.success("📈 Market likely to go UP")
else:
    st.error("📉 Market likely to go DOWN")

# Price Chart
st.subheader("📉 NIFTY Closing Price")
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(data.index, data["Close"])
ax.set_xlabel("Date")
ax.set_ylabel("Index Value")
st.pyplot(fig)

# Data Preview
st.subheader("📄 Data Preview")
st.dataframe(data.tail())

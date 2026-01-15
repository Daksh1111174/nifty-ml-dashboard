import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from model import train_model
from signals import generate_signal, risk_management

st.set_page_config(page_title="PRO Trading Signals", layout="wide")

st.title("📉 PRO Buy / Sell Signal Dashboard (ML + Indicators)")

# Sidebar
period = st.sidebar.selectbox(
    "Select Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

confidence_threshold = st.sidebar.slider(
    "ML Confidence Threshold",
    min_value=0.55,
    max_value=0.75,
    value=0.60,
    step=0.01
)

@st.cache_data
def load_data(period):
    return yf.download("^NSEI", period=period, auto_adjust=True)

df = load_data(period)

# Train ML model
model, df, features = train_model(df)

# Generate Signal
signal, prob = generate_signal(
    model=model,
    df=df,
    features=features,
    prob_threshold=confidence_threshold
)

st.subheader("📊 Trading Signal")

if signal == "BUY":
    st.success(f"🟢 BUY | Confidence: {prob:.2f}")
elif signal == "SELL":
    st.error(f"🔴 SELL | Confidence: {1 - prob:.2f}")
else:
    st.warning(f"🟡 HOLD | ML Confidence: {prob:.2f}")

# Risk Management
entry, sl, target = risk_management(df)

st.markdown("### 🛡️ Risk Management")
st.write(f"**Entry Price:** {entry:.2f}")
st.write(f"**Stop Loss:** {sl:.2f}")
st.write(f"**Target:** {target:.2f}")

# Price Chart
st.subheader("📈 NIFTY Price Chart")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["Close"])
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# Data Preview
st.subheader("📄 Latest Data")
st.dataframe(df.tail())

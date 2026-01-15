import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from model import train_model
from signals import generate_signal, risk_management

st.set_page_config(page_title="PRO Trading Signals", layout="wide")

st.title("📉 PRO Buy / Sell Signal Dashboard")

# Sidebar
period = st.sidebar.selectbox(
    "Select Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=2
)

confidence_threshold = st.sidebar.slider(
    "ML Confidence Threshold",
    0.55, 0.75, 0.60, 0.01
)

@st.cache_data
def load_data(period):
    return yf.download("^NSEI", period=period, auto_adjust=True)

df = load_data(period)

# Train model
model, df, features = train_model(df)

# ---------- HANDLE LOW DATA SAFELY ----------
if model is None:
    st.warning("⚠️ Not enough clean data for ML training. Defaulting to HOLD.")
    signal = "HOLD"
    prob = 0.5
else:
    signal, prob = generate_signal(
        model=model,
        df=df,
        features=features,
        prob_threshold=confidence_threshold
    )

# ---------- DISPLAY SIGNAL ----------
st.subheader("📊 Trading Signal")

if signal == "BUY":
    st.success(f"🟢 BUY | Confidence: {prob:.2f}")
elif signal == "SELL":
    st.error(f"🔴 SELL | Confidence: {1 - prob:.2f}")
else:
    st.warning("🟡 HOLD | No high-confidence setup")

# ---------- RISK MANAGEMENT ----------
entry, sl, target = risk_management(df)

st.markdown("### 🛡️ Risk Management")
st.write(f"Entry Price: {entry:.2f}")
st.write(f"Stop Loss: {sl:.2f}")
st.write(f"Target: {target:.2f}")

# ---------- PRICE CHART ----------
st.subheader("📈 NIFTY Price Chart")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index, df["Close"])
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# ---------- DATA ----------
st.subheader("📄 Latest Data")
st.dataframe(df.tail())

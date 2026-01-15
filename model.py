import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def add_indicators(df):
    df = df.copy()

    # Returns
    df["Return"] = df["Close"].pct_change()

    # EMAs
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # RSI (safe)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Volatility
    df["Volatility"] = df["Return"].rolling(20).std()

    # Target
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # 🔥 CRITICAL CLEANING
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df

def train_model(df):
    df = add_indicators(df)

    features = [
        "Return",
        "EMA20",
        "EMA50",
        "RSI",
        "MACD",
        "MACD_Signal",
        "Volatility"
    ]

    X = df[features].astype(float)
    y = df["Target"].astype(int)

    # 🚨 Safety check
    if len(X) < 50:
        raise ValueError("Not enough clean data to train the model")

    # 🔥 Scaling (mandatory for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(X_scaled, y)

    # Store scaler inside model (PRO trick)
    model.scaler = scaler

    return model, df, features

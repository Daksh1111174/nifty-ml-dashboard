import numpy as np

def generate_signal(model, df, features, prob_threshold=0.60):
    # Always work on numpy array (not pandas)
    latest = df.iloc[-1:].copy()

    # ---- Convert ALL indicators to Python floats ----
    ema20 = float(latest["EMA20"].values[0])
    ema50 = float(latest["EMA50"].values[0])
    rsi = float(latest["RSI"].values[0])
    macd = float(latest["MACD"].values[0])
    macd_signal = float(latest["MACD_Signal"].values[0])

    # ---- ML Probability ----
    X_latest = model.scaler.transform(latest[features].to_numpy())
    prob_up = float(model.predict_proba(X_latest)[0][1])

    # ---- Trend Filter ----
    trend_up = ema20 > ema50
    trend_down = ema20 < ema50

    # ---- Momentum ----
    rsi_buy = rsi > 55
    rsi_sell = rsi < 45

    macd_buy = macd > macd_signal
    macd_sell = macd < macd_signal

    # ---- Final Signal ----
    if trend_up and rsi_buy and macd_buy and prob_up >= prob_threshold:
        return "BUY", prob_up

    elif trend_down and rsi_sell and macd_sell and prob_up <= (1 - prob_threshold):
        return "SELL", prob_up

    else:
        return "HOLD", prob_up


def risk_management(df, sl_pct=0.02, target_pct=0.04):
    entry = float(df["Close"].values[-1])
    stop_loss = entry * (1 - sl_pct)
    target = entry * (1 + target_pct)
    return entry, stop_loss, target

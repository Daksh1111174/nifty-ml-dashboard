def generate_signal(model, df, features, prob_threshold=0.60):
    latest = df.iloc[-1]

    # Force scalar values (IMPORTANT FIX)
    X_latest = latest[features].to_numpy().reshape(1, -1)
    prob_up = float(model.predict_proba(X_latest)[0][1])

    ema20 = float(latest["EMA20"])
    ema50 = float(latest["EMA50"])
    rsi = float(latest["RSI"])
    macd = float(latest["MACD"])
    macd_signal = float(latest["MACD_Signal"])

    # Trend
    trend_up = ema20 > ema50
    trend_down = ema20 < ema50

    # Momentum
    rsi_buy = rsi > 55
    rsi_sell = rsi < 45

    macd_buy = macd > macd_signal
    macd_sell = macd < macd_signal

    if trend_up and rsi_buy and macd_buy and prob_up >= prob_threshold:
        return "BUY", prob_up

    elif trend_down and rsi_sell and macd_sell and prob_up <= (1 - prob_threshold):
        return "SELL", prob_up

    else:
        return "HOLD", prob_up


def risk_management(df, sl_pct=0.02, target_pct=0.04):
    entry = float(df["Close"].iloc[-1])
    stop_loss = entry * (1 - sl_pct)
    target = entry * (1 + target_pct)
    return entry, stop_loss, target

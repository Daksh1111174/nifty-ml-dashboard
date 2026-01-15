def generate_signal(model, df, features, prob_threshold=0.60):
    latest = df.iloc[-1]
    X_latest = latest[features].values.reshape(1, -1)

    prob_up = model.predict_proba(X_latest)[0][1]

    trend_up = latest["EMA20"] > latest["EMA50"]
    trend_down = latest["EMA20"] < latest["EMA50"]

    rsi_buy = latest["RSI"] > 55
    rsi_sell = latest["RSI"] < 45

    macd_buy = latest["MACD"] > latest["MACD_Signal"]
    macd_sell = latest["MACD"] < latest["MACD_Signal"]

    if trend_up and rsi_buy and macd_buy and prob_up >= prob_threshold:
        return "BUY", prob_up

    elif trend_down and rsi_sell and macd_sell and prob_up <= (1 - prob_threshold):
        return "SELL", prob_up

    else:
        return "HOLD", prob_up


def risk_management(df, sl_pct=0.02, target_pct=0.04):
    entry = df["Close"].iloc[-1]
    stop_loss = entry * (1 - sl_pct)
    target = entry * (1 + target_pct)
    return entry, stop_loss, target

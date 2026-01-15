import pandas as pd
from sklearn.linear_model import LogisticRegression

def add_indicators(df):
    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    df["Volatility"] = df["Return"].rolling(20).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
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

    X = df[features]
    y = df["Target"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model, df, features

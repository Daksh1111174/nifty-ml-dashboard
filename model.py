import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model(df):
    df = df.copy()

    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Return"].rolling(20).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    X = df[["Return", "MA20", "MA50", "Volatility"]]
    y = df["Target"]

    model = LogisticRegression()
    model.fit(X, y)

    return model, X

def predict_trend(model, X):
    latest = X.iloc[-1:].values
    return model.predict(latest)[0]

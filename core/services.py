import yfinance as yf
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def get_stock_data(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    return data


def compute_momentum(data):
    returns = data["Close"].pct_change()
    momentum = returns.rolling(30).mean().iloc[-1]
    return momentum


def rank_stocks(tickers):
    results = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1y", progress=False)

            if data.empty:
                continue

            close = data["Close"]

            # Ensure it's a Series
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            returns = close.pct_change()

            momentum = returns.rolling(30).mean().dropna()

            if len(momentum) == 0:
                continue

            results.append({
                "ticker": ticker,
                "momentum": float(momentum.iloc[-1])
            })

        except Exception as e:
            print("Error with", ticker, e)
            continue

    df = pd.DataFrame(results)

    if df.empty:
        return []

    df = df.sort_values("momentum", ascending=False)

    return df.to_dict(orient="records")





def get_market_news(limit=5):
    try:
        ticker = yf.Ticker("SPY")  # Market proxy
        news = ticker.news

        results = []

        for item in news[:limit]:
            results.append({
                "title": item.get("title"),
                "publisher": item.get("publisher"),
                "link": item.get("link"),
                "date": datetime.fromtimestamp(item.get("providerPublishTime")).strftime("%Y-%m-%d")
            })

        return results

    except Exception as e:
        print("News error:", e)
        return []

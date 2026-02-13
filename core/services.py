import yfinance as yf
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime





def get_stock_data(ticker, period="1y"):
    data = yf.download(ticker, period=period)
    return data


def compute_momentum(data):
    if "Close" not in data.columns:
        return 0

    close = data["Close"]

    # If multi-index or dataframe, squeeze to Series
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()

    if len(close) < 30:
        return 0

    returns = close.pct_change().dropna()
    rolling = returns.rolling(30).mean().dropna()

    if rolling.empty:
        return 0

    value = rolling.iloc[-1]

    return float(value)




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
        ticker = yf.Ticker("SPY")
        news = ticker.news

        if not news:
            return []

        results = []

        for item in news[:limit]:
            content = item.get("content", {})

            title = content.get("title", "No title")
            publisher = content.get("provider", {}).get("displayName", "Unknown")
            link = content.get("canonicalUrl", {}).get("url", "#")

            timestamp = content.get("pubDate")

            if timestamp:
                date = pd.to_datetime(timestamp).strftime("%Y-%m-%d")
            else:
                date = "N/A"

            results.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "date": date
            })

        return results

    except Exception as e:
        print("News error:", e)
        return []


def build_watchlist(tickers, period="1mo"):
    results = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, progress=False)

            if data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
            else:
                close = data["Close"]

                close = close.dropna()


            # Ensure Series
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            current_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2]) if len(close) > 1 else current_price

            change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0

            # 5-day performance
            if len(close) >= 5:
                price_5d = float(close.iloc[-5])
                perf_5d = ((current_price - price_5d) / price_5d) * 100 if price_5d != 0 else 0
            else:
                perf_5d = 0

            # Momentum (reuse your logic)
            momentum = compute_momentum(data)
            momentum = float(momentum) if not pd.isna(momentum) else 0

            results.append({
                "ticker": ticker,
                "price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "perf_5d": round(perf_5d, 2),
                "momentum": round(momentum, 4)
            })

        except Exception as e:
            print("Watchlist error:", ticker, e)
            continue

        print(StockWatchlist)


    return results


def build_watchlist_for_web(universe):
    
    return build_watchlist(universe, period="3mo")




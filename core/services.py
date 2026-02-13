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

            # Handle MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].iloc[:, 0]
                volume_series = data["Volume"].iloc[:, 0]
                high_series = data["High"].iloc[:, 0]
                low_series = data["Low"].iloc[:, 0]
            else:
                close = data["Close"]
                volume_series = data["Volume"]
                high_series = data["High"]
                low_series = data["Low"]

            close = close.dropna()

            if len(close) < 2:
                continue

            current_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])

            change_pct = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0

            # 5D Performance
            if len(close) >= 5:
                price_5d = float(close.iloc[-5])
                perf_5d = ((current_price - price_5d) / price_5d) * 100 if price_5d != 0 else 0
            else:
                perf_5d = 0

            # 20D Performance
            if len(close) >= 20:
                price_20d = float(close.iloc[-20])
                perf_20d = ((current_price - price_20d) / price_20d) * 100 if price_20d != 0 else 0
            else:
                perf_20d = 0

            # Volume
            volume = float(volume_series.iloc[-1])
            avg_volume_20 = float(volume_series.rolling(20).mean().iloc[-1])
            volume_ratio = volume / avg_volume_20 if avg_volume_20 != 0 else 0

            # Volatility
            returns = close.pct_change().dropna()
            vol_5d = returns.tail(5).std() * np.sqrt(252)
            vol_20d = returns.tail(20).std() * np.sqrt(252)

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if not rsi.empty else 0

            # Momentum
            momentum = compute_momentum(data)
            momentum = float(momentum) if not pd.isna(momentum) else 0

            results.append({
                "ticker": ticker,
                "price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "perf_5d": round(perf_5d, 2),
                "perf_20d": round(perf_20d, 2),
                "volume_ratio": round(volume_ratio, 2),
                "volatility_5d": round(vol_5d, 2),
                "volatility_20d": round(vol_20d, 2),
                "rsi": round(float(rsi_value), 2),
                "volume": int(volume),
                "high": round(float(high_series.iloc[-1]), 2),
                "low": round(float(low_series.iloc[-1]), 2),
                "alpha_score": 0,
                "news_sentiment": 0,
                "momentum": round(momentum, 4),
            })

        except Exception as e:
            print("Watchlist error:", ticker, e)
            continue

    return results



def build_watchlist_for_web(universe):
    
    return build_watchlist(universe, period="3mo")




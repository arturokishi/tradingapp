# watchlist.py

# watchlist.py

import yfinance as yf
import pandas as pd
from datetime import datetime

class StockWatchlist:
    def __init__(self, universe):
        self.universe = universe
        self.cache = {}
        self.last_update = None

    def fetch_all_data(self, period="1mo", interval="1d"):
        data = {}

        for ticker in self.universe[:15]:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period, interval=interval)

                if not hist.empty:
                    data[ticker] = hist
            except:
                continue

        self.cache = data
        self.last_update = datetime.now()
        return data
def calculate_metrics(self):
    metrics = []

    for ticker, hist in self.cache.items():
        if len(hist) < 20:
            continue

        try:
            close = hist["Close"].dropna()
            volume = hist["Volume"].dropna()

            current_price = close.iloc[-1]
            prev_close = close.iloc[-2]

            change_pct = ((current_price - prev_close) / prev_close) * 100
            perf_5d = ((current_price - close.iloc[-5]) / close.iloc[-5]) * 100
            perf_20d = ((current_price - close.iloc[-20]) / close.iloc[-20]) * 100

            vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

            vol_5d = close.pct_change().rolling(5).std().iloc[-1] * 100
            vol_20d = close.pct_change().rolling(20).std().iloc[-1] * 100

            # RSI
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]

            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50

            metrics.append({
                "ticker": ticker,
                "price": round(float(current_price), 2),
                "change_pct": round(change_pct, 2),
                "five_day_pct": round(perf_5d, 2),
                "twenty_day_pct": round(perf_20d, 2),
                "volume_ratio": round(vol_ratio, 2),
                "volatility_5d": round(vol_5d, 2),
                "volatility_20d": round(vol_20d, 2),
                "rsi": round(rsi, 2),
                "volume": int(volume.iloc[-1]),
                "high": round(float(hist["High"].iloc[-1]), 2),
                "low": round(float(hist["Low"].iloc[-1]), 2),
                "alpha_score": 0,
                "news_sentiment": 0,
            })

        except Exception as e:
            print("Metric error:", ticker, e)
            continue

    return pd.DataFrame(metrics)



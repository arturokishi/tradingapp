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

    def calculate_metrics(self):   # ← NOW INSIDE CLASS
        metrics = []

        for ticker, hist in self.cache.items():
            if len(hist) < 2:
                continue

            try:
                close = hist['Close'].dropna()

                current_price = close.iloc[-1]
                prev_close = close.iloc[-2]

                day_change = ((current_price - prev_close) / prev_close) * 100

                if len(close) >= 5:
                    price_5d_ago = close.iloc[-5]
                    perf_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
                else:
                    perf_5d = 0

                metrics.append({
                    "ticker": ticker,
                    "price": round(float(current_price), 2),
                    "change_pct": round(float(day_change), 2),
                    "five_day_pct": round(float(perf_5d), 2),
                    "momentum": 0,
                                })


            except Exception as e:
                print("Metric error:", ticker, e)
                continue

        return pd.DataFrame(metrics)


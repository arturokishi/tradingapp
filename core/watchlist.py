# watchlist.py

import yfinance as yf
import pandas as pd
from datetime import datetime
# adjust import to your structure

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
            if len(hist) < 2:
                continue

            try:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2]

                day_change = ((current_price - prev_close) / prev_close) * 100

                if len(hist) >= 5:
                    price_5d_ago = hist['Close'].iloc[-5]
                    perf_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100
                else:
                    perf_5d = 0

                summary, _, _ = get_stock_data(ticker)

                if summary.empty:
                    continue

                metrics.append({
                    "Ticker": ticker,
                    "Price": round(float(current_price), 2),
                    "Change": round(float(day_change), 2),
                    "Perf5D": round(float(perf_5d), 2),
                    "AlphaScore": float(summary.get("Alpha Score", 0)),
                    "Signal": summary.get("Signal", "NEUTRAL"),
                    "IVRank": float(summary.get("IV Rank %", 0)),
                    "NewsSentiment": float(summary.get("News Sentiment", 0)),
                    "MarketState": summary.get("Market State", "UNKNOWN")
                })

            except:
                continue

        return pd.DataFrame(metrics)

    def get_html_table(self):
        df = self.calculate_metrics()

        if df.empty:
            return "<p>No data available</p>"

        return df.to_html(
            classes="table table-striped table-dark",
            index=False
        )

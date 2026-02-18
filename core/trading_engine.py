# core/trading_engine.py
# ============================================================================
# COMPLETE TRADING ENGINE - Copy this entire file
# ============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import warnings
import time
warnings.filterwarnings('ignore')
import feedparser
from urllib.parse import quote_plus
import requests

# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

DEBUG_MODE = True

def debug_print(msg, data=None):
    if DEBUG_MODE:
        print(f"[DEBUG] {msg}")
        if data is not None:
            print(f"[DEBUG DATA] {data}")

# ============================================================================
# UNIVERSE
# ============================================================================

UNIVERSE = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","AVGO","ORCL","CRM","ADBE",
    "JPM","BAC","WFC","GS","MS","V","MA","AXP",
    "XOM","CVX","COP","SLB","OXY",
    "CAT","DE","BA","GE","LMT","RTX",
    "TSLA","HD","LOW","MCD","NKE","COST","WMT",
    "LLY","JNJ","UNH","PFE","MRK","ABBV",
    "AMD","INTC","QCOM","TXN","MU",
    "DIS","NFLX","T","VZ"
]

# ============================================================================
# NEWS API
# ============================================================================

NEWS_API_KEY = "b761a27d78a64c35b5bccfddebcf0732"

def test_news_api():
    print("🔍 Testing NewsAPI...")
    test_tickers = ["AAPL", "TSLA", "NVDA"]
    
    for ticker in test_tickers:
        print(f"\n📰 Testing news for {ticker}:")
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': ticker,
                'apiKey': NEWS_API_KEY,
                'language': 'en',
                'pageSize': 3,
                'from': (date.today() - timedelta(days=7)).isoformat()
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ok' and data.get('articles'):
                    print(f"  ✅ Found {len(data['articles'])} articles")
                else:
                    print(f"  ⚠️ No articles")
            elif response.status_code == 401:
                print(f"  ❌ Invalid API key")
            elif response.status_code == 429:
                print(f"  ⚠️ Rate limit exceeded")
            else:
                print(f"  ❌ Error: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    print("\n" + "="*50)

def get_relevant_news(ticker, company_name=None, n=5):
    try:
        if company_name:
            query = f"{company_name} OR {ticker}"
        else:
            query = ticker
        
        to_date = date.today()
        from_date = to_date - timedelta(days=7)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'apiKey': NEWS_API_KEY,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': n,
            'from': from_date.isoformat(),
            'to': to_date.isoformat()
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return get_relevant_news_fallback(ticker, company_name, n)
        
        data = response.json()
        
        if data.get('status') != 'ok' or not data.get('articles'):
            return get_relevant_news_fallback(ticker, company_name, n)
        
        rows = []
        for article in data['articles'][:n]:
            headline = article.get('title', '')
            if headline is None:
                headline = ''
            
            text = str(headline).lower()
            sentiment = score_headline_sentiment(text)
            
            rows.append({
                "Headline": headline,
                "Source": article.get('source', {}).get('name', 'Unknown') or 'Unknown',
                "Published": pd.to_datetime(article.get('publishedAt'), errors="coerce"),
                "sentiment_score": sentiment,
                "URL": article.get('url', '#') or '#',
                "Description": article.get('description', '') or ''
            })
        
        debug_print(f"Found {len(rows)} news articles for {ticker}")
        return pd.DataFrame(rows)
        
    except Exception as e:
        debug_print(f"NewsAPI error for {ticker}: {e}")
        return get_relevant_news_fallback(ticker, company_name, n)

def get_relevant_news_fallback(ticker, company_name=None, n=5):
    try:
        if company_name:
            query = f"{company_name} stock"
        else:
            query = f"{ticker} stock"

        query_encoded = quote_plus(query)
        url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"
        
        feed = feedparser.parse(url)

        rows = []
        for entry in feed.entries[:n]:
            text = entry.title.lower() if entry.title else ''
            sentiment = score_headline_sentiment(text)
            
            rows.append({
                "Headline": entry.title or '',
                "Source": (entry.source.title if "source" in entry else "Google News") or 'Google News',
                "Published": pd.to_datetime(entry.published, errors="coerce"),
                "sentiment_score": sentiment,
                "URL": entry.link or '#',
                "Description": entry.get('summary', '') or ''
            })
        
        debug_print(f"Fallback: Found {len(rows)} news articles for {ticker}")
        return pd.DataFrame(rows)
        
    except Exception as e:
        debug_print(f"Fallback news error for {ticker}: {e}")
        return pd.DataFrame(columns=["Headline", "Source", "Published", "sentiment_score", "URL", "Description"])

def score_headline_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0

    text = text.lower()
    
    bullish = [
        "beats", "strong", "growth", "record", "surge", "upgrade",
        "expands", "bullish", "profit", "rally", "innovation", "positive",
        "winner", "gain", "increase", "higher", "outperform", "beat",
        "success", "breakthrough", "announces", "partnership", "deal",
        "acquisition", "merger", "buyback", "dividend", "raise", "target",
        "analyst", "recommends", "buy", "outperform", "overweight",
        "strong buy", "initiates", "coverage", "price target", "earnings",
        "profit", "revenue", "soars", "jumps", "climbs", "surges"
    ]
    
    bearish = [
        "miss", "weak", "decline", "fall", "downgrade", "lawsuit",
        "regulatory", "cut", "loss", "delay", "bearish", "negative",
        "loser", "drop", "decrease", "lower", "underperform", "misses",
        "failure", "warning", "caution", "investigation", "probe",
        "fine", "penalty", "settlement", "layoff", "cut", "reduce",
        "sell", "underweight", "underperform", "reduce", "target price cut",
        "terminates", "ends", "cancels", "postpones", "delays", "plunges",
        "tumbles", "slumps", "crashes", "plummets", "drops"
    ]
    
    bull_hits = sum(1 for word in bullish if word in text)
    bear_hits = sum(1 for word in bearish if word in text)
    raw_score = bull_hits - bear_hits
    
    total_hits = bull_hits + bear_hits
    if total_hits > 0:
        score = raw_score / max(2, total_hits)
    else:
        score = 0
    
    return np.clip(score, -1, 1)

def compute_news_score(news_df):
    if news_df is None or news_df.empty:
        return 0.0
    
    try:
        if 'sentiment_score' in news_df.columns:
            valid_scores = news_df['sentiment_score'].dropna()
            if len(valid_scores) > 0:
                avg_score = valid_scores.mean()
                return float(np.clip(avg_score, -1, 1)) if not pd.isna(avg_score) else 0.0
        return 0.0
    except Exception as e:
        debug_print(f"Error computing news score: {e}")
        return 0.0

# ============================================================================
# TRADE LOG FUNCTIONS
# ============================================================================

# Global trade log DataFrame
trade_log = pd.DataFrame(
    columns=[
        "Date",
        "Ticker",
        "Position_Type",
        "Entry_Price",
        "Exit_Price",
        "Quantity",
        "PnL",
        "PnL_Percent",
        "Hold_Days",
        "Strategy",
        "Signal_Type",
        "Signal_Value",
        "Alpha_Score",
        "Market_State",
        "Notes"
    ]
)

def log_trade(ticker, entry_price, exit_price, quantity, 
              position_type="LONG", strategy="Backtest", 
              signal_type="RSI", signal_value=0, alpha_score=0,
              market_state="UNKNOWN", hold_days=0, notes=""):
    """Log a trade to the trade log DataFrame"""
    global trade_log
    
    try:
        pnl = (exit_price - entry_price) * quantity
        if position_type == "SHORT":
            pnl = -pnl
            
        pnl_percent = ((exit_price - entry_price) / entry_price) * 100
        if position_type == "SHORT":
            pnl_percent = -pnl_percent
        
        new_trade = {
            "Date": datetime.now(),
            "Ticker": ticker,
            "Position_Type": position_type,
            "Entry_Price": round(entry_price, 2),
            "Exit_Price": round(exit_price, 2),
            "Quantity": quantity,
            "PnL": round(pnl, 2),
            "PnL_Percent": round(pnl_percent, 2),
            "Hold_Days": hold_days,
            "Strategy": strategy,
            "Signal_Type": signal_type,
            "Signal_Value": round(signal_value, 2),
            "Alpha_Score": round(alpha_score, 2),
            "Market_State": market_state,
            "Notes": notes
        }
        
        trade_log = pd.concat(
            [trade_log, pd.DataFrame([new_trade])],
            ignore_index=True
        )
        
        return True
    except Exception as e:
        print(f"Error logging trade: {e}")
        return False

def get_trade_summary():
    """Get summary statistics from trade log"""
    if trade_log.empty:
        return None
    
    summary = {
        "Total_Trades": len(trade_log),
        "Total_PnL": trade_log["PnL"].sum(),
        "Avg_PnL": trade_log["PnL"].mean(),
        "Win_Rate": (trade_log["PnL"] > 0).mean() * 100,
        "Avg_Hold_Days": trade_log["Hold_Days"].mean(),
        "Max_Win": trade_log["PnL"].max(),
        "Max_Loss": trade_log["PnL"].min(),
        "Sharpe_Ratio": trade_log["PnL"].mean() / trade_log["PnL"].std() if trade_log["PnL"].std() > 0 else 0,
        "Profit_Factor": trade_log[trade_log["PnL"] > 0]["PnL"].sum() / abs(trade_log[trade_log["PnL"] < 0]["PnL"].sum()) if trade_log[trade_log["PnL"] < 0]["PnL"].sum() != 0 else float('inf')
    }
    
    return summary

def analyze_trade_performance():
    """Analyze trade performance by various metrics"""
    if trade_log.empty:
        return None
    
    analysis = {}
    
    if "Strategy" in trade_log.columns:
        strategy_stats = trade_log.groupby("Strategy").agg({
            "PnL": ["count", "sum", "mean", "std"],
            "PnL_Percent": ["mean", "std"],
            "Hold_Days": "mean"
        }).round(2)
        analysis["By_Strategy"] = strategy_stats
    
    if "Signal_Type" in trade_log.columns:
        signal_stats = trade_log.groupby("Signal_Type").agg({
            "PnL": ["count", "sum", "mean"],
            "Win_Rate": lambda x: (x > 0).mean() * 100
        }).round(2)
        analysis["By_Signal"] = signal_stats
    
    if "Market_State" in trade_log.columns:
        market_stats = trade_log.groupby("Market_State").agg({
            "PnL": ["count", "sum", "mean"],
            "Win_Rate": lambda x: (x > 0).mean() * 100
        }).round(2)
        analysis["By_Market_State"] = market_stats
    
    return analysis

def get_market_news():
    """Get general market news"""
    try:
        url = "https://newsapi.org/v2/top-headlines"
        params = {
            'apiKey': NEWS_API_KEY,
            'category': 'business',
            'language': 'en',
            'country': 'us',
            'pageSize': 10
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok' and data.get('articles'):
                rows = []
                for article in data['articles']:
                    rows.append({
                        'title': article.get('title', ''),
                        'publisher': article.get('source', {}).get('name', 'Unknown'),
                        'date': article.get('publishedAt', '')[:10],
                        'link': article.get('url', '#')
                    })
                return rows
    except Exception as e:
        debug_print(f"Market news error: {e}")
    
    return []

# ============================================================================
# STOCK DATA FUNCTION
# ============================================================================

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y").dropna()

        if hist.empty:
            debug_print(f"No history data for {ticker}")
            return pd.Series(), pd.DataFrame(), pd.DataFrame()

        hist["returns"] = hist["Close"].pct_change()
        
        if len(hist) >= 50:
            hist["EMA20"] = hist["Close"].ewm(span=20, adjust=False).mean()
            hist["EMA50"] = hist["Close"].ewm(span=50, adjust=False).mean()
            hist["trend_strength"] = abs(hist["EMA20"] - hist["EMA50"]) / hist["Close"]
        else:
            hist["EMA20"] = hist["Close"]
            hist["EMA50"] = hist["Close"]
            hist["trend_strength"] = 0
        
        if len(hist) >= 21:
            hist["rv_21"] = hist["returns"].rolling(21).std() * np.sqrt(252) * 100
            hist["rv_percentile"] = hist["rv_21"].rank(pct=True)
        else:
            hist["rv_21"] = 0
            hist["rv_percentile"] = 0.5
        
        if len(hist) >= 126:
            hist["mom_3m"] = hist["Close"].pct_change(63)
            hist["mom_6m"] = hist["Close"].pct_change(126)
        else:
            hist["mom_3m"] = 0
            hist["mom_6m"] = 0
        
        hist["cum_max"] = hist["Close"].cummax()
        hist["drawdown"] = hist["Close"] / hist["cum_max"] - 1
        
        try:
            downside = hist.loc[hist["returns"] < 0, "returns"].std()
            upside = hist.loc[hist["returns"] > 0, "returns"].std()
            hist["skew_ratio"] = downside / upside if upside > 0 else 1
        except:
            hist["skew_ratio"] = 1

        try:
            news_df = get_relevant_news(ticker, company_name=info.get("shortName", ticker))
        except:
            news_df = pd.DataFrame()
        
        try:
            regime = classify_market_state(hist)
            state_str = regime["state"]
        except:
            state_str = "UNKNOWN"
        
        try:
            alpha = compute_alpha_score(hist, news_df)
            alpha_score = alpha["alpha_score"]
            signal = alpha["signal"]
        except:
            alpha_score = 0
            signal = "NEUTRAL"
        
        try:
            strategy = map_strategy(state_str, signal)
        except:
            strategy = "NEUTRAL / SMALL SIZE"
        
        try:
            iv_info = calculate_iv_rank(ticker)
        except:
            iv_info = {"IV Rank": 0, "Current IV": 0, "IV Percentile": 0}

        summary = {
            "Ticker": ticker,
            "Company": info.get("shortName", ticker),
            "Sector": info.get("sector", "Unknown"),
            "Market Cap ($B)": round(info.get("marketCap", 0) / 1e9, 2),
            "Price": info.get("regularMarketPrice", hist["Close"].iloc[-1]),
            "Realized Volatility (%)": round(hist["returns"].std() * np.sqrt(252) * 100, 2) if len(hist["returns"].dropna()) > 1 else 0,
            "Max Drawdown (%)": round(hist["drawdown"].min() * 100, 2),
            "Market State": state_str,
            "Alpha Score": alpha_score,
            "Signal": signal,
            "Strategy": strategy,
            "IV Rank %": iv_info["IV Rank"],
            "Current IV %": iv_info["Current IV"],
            "IV Percentile %": iv_info["IV Percentile"],
            "News Sentiment": compute_news_score(news_df)
        }

        return pd.Series(summary), hist, news_df
        
    except Exception as e:
        debug_print(f"Error getting data for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(), pd.DataFrame(), pd.DataFrame()

def classify_market_state(hist):
    if hist.empty or 'rv_percentile' not in hist.columns:
        return {
            "state": "UNKNOWN",
            "vol_regime": "UNKNOWN",
            "trend_dir": "UNKNOWN",
            "trend_strength": "UNKNOWN",
            "momentum": "UNKNOWN"
        }
    
    latest = hist.iloc[-1]

    if latest["rv_percentile"] < 0.25:
        vol_regime = "LOW_VOL"
    elif latest["rv_percentile"] > 0.75:
        vol_regime = "HIGH_VOL"
    else:
        vol_regime = "MID_VOL"

    trend_dir = "UP" if latest["EMA20"] > latest["EMA50"] else "DOWN"
    trend_strength = "STRONG" if latest["trend_strength"] > 0.02 else "WEAK"

    momentum = (
        "MOMENTUM"
        if latest["mom_3m"] > 0.05 and latest["mom_6m"] > 0.05
        else "MEAN_REVERSION"
    )

    if latest["drawdown"] < -0.20 and vol_regime == "HIGH_VOL":
        state = "CRASH"
    else:
        state = f"{trend_dir}_{trend_strength}_{vol_regime}_{momentum}"

    return {
        "state": state,
        "vol_regime": vol_regime,
        "trend_dir": trend_dir,
        "trend_strength": trend_strength,
        "momentum": momentum,
    }

def compute_alpha_score(hist, news_df=None):
    if hist.empty or len(hist) < 20:
        return {
            "alpha_score": 0,
            "signal": "NEUTRAL",
            "components": {
                "trend": 0, "momentum": 0, "volatility": 0,
                "skew": 0, "drawdown": 0, "news": 0
            }
        }
    
    try:
        latest = hist.iloc[-1]
        
        trend_score = 1 if latest.get("EMA20", 0) > latest.get("EMA50", 0) else -1
        momentum_score = np.clip((latest.get("mom_6m", 0) * 100) / 20, -1, 1)
        
        vol_score = (
            1 if latest.get("rv_percentile", 0.5) < 0.25 else
            -1 if latest.get("rv_percentile", 0.5) > 0.75 else
            0
        )
        
        drawdown_score = -1 if latest.get("drawdown", 0) < -0.15 else 1
        
        skew_score = (
            -1 if latest.get("skew_ratio", 1) > 1.2 else
            1 if latest.get("skew_ratio", 1) < 0.8 else
            0
        )

        news_score = compute_news_score(news_df)

        alpha = (
            0.25 * trend_score +
            0.20 * momentum_score +
            0.15 * vol_score +
            0.15 * skew_score +
            0.10 * drawdown_score +
            0.15 * news_score
        )

        if alpha > 0.5:
            signal = "BULLISH"
        elif alpha < -0.5:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return {
            "alpha_score": round(alpha, 2),
            "signal": signal,
            "components": {
                "trend": trend_score,
                "momentum": round(momentum_score, 2),
                "volatility": vol_score,
                "skew": skew_score,
                "drawdown": drawdown_score,
                "news": round(news_score, 2)
            }
        }
    except Exception as e:
        debug_print(f"Error in compute_alpha_score: {e}")
        return {
            "alpha_score": 0,
            "signal": "NEUTRAL",
            "components": {
                "trend": 0, "momentum": 0, "volatility": 0,
                "skew": 0, "drawdown": 0, "news": 0
            }
        }

def map_strategy(state, alpha_signal):
    if state == "CRASH":
        return "LONG VOLATILITY / NO DIRECTIONAL RISK"
    if "LOW_VOL" in state and alpha_signal == "BULLISH":
        return "LONG STOCK + SELL PUTS"
    if "LOW_VOL" in state and "MEAN_REVERSION" in state:
        return "SHORT STRADDLES / IRON CONDORS"
    if "HIGH_VOL" in state and alpha_signal == "BULLISH":
        return "LONG CALLS / DEBIT SPREADS"
    if "HIGH_VOL" in state and alpha_signal == "BEARISH":
        return "LONG PUTS"
    return "NEUTRAL / SMALL SIZE"

def calculate_iv_rank(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty or len(hist) < 50:
            return {"IV Rank": 0, "Current IV": 0, "IV Percentile": 0}
        
        hist['returns'] = hist['Close'].pct_change()
        hist['HV_20'] = hist['returns'].rolling(window=20, min_periods=5).std() * np.sqrt(252) * 100
        
        current_hv = hist['HV_20'].iloc[-1] if not pd.isna(hist['HV_20'].iloc[-1]) else 0
        
        available_days = min(252, len(hist))
        hv_52_high = hist['HV_20'].rolling(window=available_days, min_periods=10).max().iloc[-1]
        hv_52_low = hist['HV_20'].rolling(window=available_days, min_periods=10).min().iloc[-1]
        
        if pd.isna(current_hv) or pd.isna(hv_52_high) or pd.isna(hv_52_low):
            return {"IV Rank": 0, "Current IV": round(current_hv, 2) if not pd.isna(current_hv) else 0, 
                    "IV Percentile": 0}
        
        if (hv_52_high - hv_52_low) == 0:
            iv_rank = 0.5
        else:
            iv_rank = (current_hv - hv_52_low) / (hv_52_high - hv_52_low)
        
        iv_percentile = hist['HV_20'].rank(pct=True).iloc[-1] if not pd.isna(hist['HV_20'].rank(pct=True).iloc[-1]) else 0
        
        return {
            "IV Rank": round(iv_rank * 100, 1),
            "Current IV": round(current_hv, 2),
            "IV Percentile": round(iv_percentile * 100, 1)
        }
        
    except Exception as e:
        debug_print(f"Error calculating IV Rank for {ticker}: {e}")
        return {"IV Rank": 0, "Current IV": 0, "IV Percentile": 0}

# ============================================================================
# WATCHLIST CLASS
# ============================================================================

class StockWatchlist:
    def __init__(self, universe=UNIVERSE):
        self.universe = universe
        self.cache = {}
        self.last_update = None
    
    def fetch_all_data(self, period="1mo", interval="1d"):
        print(f"📊 Fetching data for {len(self.universe)} stocks...")
        
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
        print(f"✅ Updated {len(data)} stocks")
        return data
    
    def calculate_metrics(self):
        metrics = []
        
        for ticker, hist in self.cache.items():
            if len(hist) < 2:
                continue
                
            try:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                day_change = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
                
                if len(hist) >= 5:
                    price_5d_ago = hist['Close'].iloc[-5]
                    perf_5d = ((current_price - price_5d_ago) / price_5d_ago) * 100 if price_5d_ago != 0 else 0
                else:
                    perf_5d = 0
                
                summary, _, _ = get_stock_data(ticker)
                
                if summary.empty:
                    continue
                
                metrics.append({
                    'Ticker': ticker,
                    'Price': round(float(current_price), 2),
                    'Change %': round(float(day_change), 2),
                    '5D %': round(float(perf_5d), 2),
                    'Alpha Score': float(summary.get('Alpha Score', 0)) if not pd.isna(summary.get('Alpha Score', 0)) else 0,
                    'Signal': summary.get('Signal', 'NEUTRAL'),
                    'IV Rank %': float(summary.get('IV Rank %', 0)) if not pd.isna(summary.get('IV Rank %', 0)) else 0,
                    'News Sentiment': float(summary.get('News Sentiment', 0)) if not pd.isna(summary.get('News Sentiment', 0)) else 0,
                    'Market State': summary.get('Market State', 'UNKNOWN')
                })
                    
            except Exception as e:
                debug_print(f"Error calculating metrics for {ticker}: {e}")
                continue
        
        df = pd.DataFrame(metrics)
        return df
    
    def create_dashboard(self, refresh=False):
        if refresh or not self.cache:
            self.fetch_all_data(period="1mo")
        
        metrics_df = self.calculate_metrics()
        
        if metrics_df.empty:
            print("No data available")
        else:
            styled_df = metrics_df.style \
                .background_gradient(subset=['Change %'], cmap='RdYlGn', vmin=-5, vmax=5) \
                .background_gradient(subset=['Alpha Score'], cmap='RdYlGn', vmin=-1, vmax=1) \
                .background_gradient(subset=['IV Rank %'], cmap='RdYlBu_r', vmin=0, vmax=100) \
                .background_gradient(subset=['News Sentiment'], cmap='RdYlGn', vmin=-1, vmax=1) \
                .format({
                    'Price': '${:.2f}',
                    'Change %': '{:.2f}%',
                    '5D %': '{:.2f}%',
                    'Alpha Score': '{:.2f}',
                    'IV Rank %': '{:.1f}%',
                    'News Sentiment': '{:.2f}'
                })
            
            return styled_df

# ============================================================================
# BACKTEST FUNCTION
# ============================================================================

def simple_backtest(
    ticker,
    signal_type="RSI",
    threshold=30,
    lookback_days=20,
    hold_days=5,
    position_size=1.0,
    use_alpha=False,
    alpha_lookback=10,
    use_sentiment=False,
    sentiment_threshold=0.0,
    log_trades=True
):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    if hist.empty or len(hist) < lookback_days + hold_days:
        return None

    try:
        stock_info = stock.info
        ticker_name = stock_info.get("shortName", ticker)
    except:
        ticker_name = ticker
    
    summary, full_hist, _ = get_stock_data(ticker)
    if not full_hist.empty:
        market_state = classify_market_state(full_hist)
        market_state_str = market_state.get("state", "UNKNOWN")
        alpha_info = compute_alpha_score(full_hist)
        alpha_score = alpha_info.get("alpha_score", 0)
    else:
        market_state_str = "UNKNOWN"
        alpha_score = 0

    hist["returns"] = hist["Close"].pct_change()

    if signal_type == "RSI":
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        hist["signal"] = 100 - (100 / (1 + rs))
        signal_mask = hist["signal"] < threshold

    elif signal_type == "Volatility":
        hist["signal"] = hist["returns"].rolling(20).std() * np.sqrt(252) * 100
        signal_mask = hist["signal"] > threshold

    elif signal_type == "Alpha":
        market = yf.download("^GSPC", period="1y")["Close"].pct_change()
        hist["alpha"] = hist["returns"] - market.reindex(hist.index).fillna(0)
        hist["signal"] = hist["alpha"].rolling(lookback_days).mean() * 100
        signal_mask = hist["signal"] > threshold

    elif signal_type == "News Sentiment":
        hist["signal"] = hist["returns"].rolling(3).mean()
        signal_mask = hist["signal"] > sentiment_threshold

    else:
        return None

    if use_alpha:
        hist["alpha_filter"] = hist["returns"].rolling(alpha_lookback).mean()
        signal_mask &= hist["alpha_filter"] > 0

    if use_sentiment:
        hist["sentiment"] = hist["returns"].rolling(3).mean()
        signal_mask &= hist["sentiment"] >= sentiment_threshold

    trades = []

    for i in range(lookback_days, len(hist) - hold_days):
        if signal_mask.iloc[i]:
            entry = hist["Close"].iloc[i]
            exit_ = hist["Close"].iloc[i + hold_days]
            ret = (exit_ - entry) / entry * position_size
            
            if log_trades:
                log_trade(
                    ticker=ticker,
                    entry_price=entry,
                    exit_price=exit_,
                    quantity=position_size,
                    position_type="LONG",
                    strategy="Backtest",
                    signal_type=signal_type,
                    signal_value=hist["signal"].iloc[i],
                    alpha_score=alpha_score,
                    market_state=market_state_str,
                    hold_days=hold_days,
                    notes=f"Backtest: {signal_type} signal"
                )

            trades.append({
                "date": hist.index[i],
                "entry_price": entry,
                "exit_price": exit_,
                "return": ret
            })

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    stats = {
        "total_trades": len(trades_df),
        "avg_return": trades_df["return"].mean(),
        "win_rate": (trades_df["return"] > 0).mean(),
        "total_return": trades_df["return"].sum(),
        "sharpe_ratio": (
            trades_df["return"].mean() / trades_df["return"].std()
            if trades_df["return"].std() > 0 else 0
        )
    }

    return stats, trades_df
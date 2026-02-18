# core/trading_views.py
from django.shortcuts import render
from django.http import JsonResponse
from .trading_engine import (
    UNIVERSE, get_stock_data, StockWatchlist, 
    simple_backtest, calculate_iv_rank, trade_log,
    get_trade_summary, analyze_trade_performance,
    log_trade, get_market_news
)
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize watchlist
watchlist = StockWatchlist(UNIVERSE)

def trading_dashboard(request):
    """Main trading dashboard with tabs"""
    return render(request, 'core/trading_dashboard.html', {
        'universe': UNIVERSE,
    })

def api_watchlist(request):
    """API endpoint for watchlist data"""
    try:
        watchlist.fetch_all_data(period="1mo")
        metrics = watchlist.calculate_metrics()
        data = metrics.to_dict('records')
        # Convert numpy types to Python types
        for item in data:
            for key, value in item.items():
                if isinstance(value, (np.integer, np.floating)):
                    item[key] = float(value)
        return JsonResponse({'success': True, 'data': data})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def api_analysis(request):
    ticker = request.GET.get('ticker', 'AAPL')
    try:
        summary, hist, news_df = get_stock_data(ticker)
        
        summary_dict = {}
        if not summary.empty:
            for k, v in summary.items():
                if isinstance(v, (np.integer, np.floating)):
                    summary_dict[k] = float(v)
                else:
                    summary_dict[k] = v
        
        news_data = []
        if not news_df.empty:
            for _, row in news_df.iterrows():
                news_data.append({
                    'headline': str(row.get('Headline', '')),
                    'source': str(row.get('Source', '')),
                    'published': row.get('Published', '').isoformat() if pd.notna(row.get('Published', '')) else '',
                    'sentiment': float(row.get('sentiment_score', 0)),
                    'url': str(row.get('URL', '#')),
                    'description': str(row.get('Description', ''))[:200] + '...' if len(str(row.get('Description', ''))) > 200 else str(row.get('Description', ''))
                })
        
        price_data = []
        if not hist.empty:
            hist_subset = hist[['Close', 'EMA20', 'EMA50', 'rv_21']].dropna().tail(100)
            for idx, row in hist_subset.iterrows():
                price_data.append({
                    'date': idx.isoformat(),
                    'close': float(row['Close']),
                    'ema20': float(row.get('EMA20', 0)),
                    'ema50': float(row.get('EMA50', 0)),
                    'volatility': float(row.get('rv_21', 0))
                })
        
        return JsonResponse({
            'success': True,
            'summary': summary_dict,
            'news': news_data,
            'price_history': price_data
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def api_ranking(request):
    metric = request.GET.get('metric', 'Alpha Score')
    limit = int(request.GET.get('limit', 20))
    
    try:
        rows = []
        for ticker in UNIVERSE[:30]:
            try:
                summary, _, _ = get_stock_data(ticker)
                if not summary.empty:
                    rows.append(summary)
            except:
                continue
        
        if not rows:
            return JsonResponse({'success': False, 'error': 'No data'})
        
        df = pd.DataFrame(rows)
        df = df.sort_values(metric, ascending=False).head(limit)
        
        data = df.to_dict('records')
        for item in data:
            for key, value in item.items():
                if isinstance(value, (np.integer, np.floating)):
                    item[key] = float(value)
        
        return JsonResponse({'success': True, 'data': data})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def api_backtest(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            stats, trades = simple_backtest(
                ticker=data.get('ticker', 'AAPL'),
                signal_type=data.get('signal_type', 'RSI'),
                threshold=float(data.get('threshold', 30)),
                lookback_days=int(data.get('lookback', 20)),
                hold_days=int(data.get('hold_days', 5)),
                position_size=float(data.get('position_size', 1.0)),
                use_alpha=data.get('use_alpha', False),
                alpha_lookback=int(data.get('alpha_lookback', 10)),
                use_sentiment=data.get('use_sentiment', False),
                sentiment_threshold=float(data.get('sentiment_threshold', 0.0)),
                log_trades=True
            )
            
            if stats is None:
                return JsonResponse({'success': False, 'error': 'No trades generated'})
            
            stats_serializable = {}
            for k, v in stats.items():
                if isinstance(v, (np.integer, np.floating)):
                    stats_serializable[k] = float(v)
                else:
                    stats_serializable[k] = v
            
            trades_data = []
            if trades is not None and not trades.empty:
                trades_data = trades.to_dict('records')
                for trade in trades_data:
                    for key, value in trade.items():
                        if isinstance(value, (np.integer, np.floating, pd.Timestamp)):
                            if isinstance(value, pd.Timestamp):
                                trade[key] = value.isoformat()
                            else:
                                trade[key] = float(value)
            
            return JsonResponse({
                'success': True,
                'stats': stats_serializable,
                'trades': trades_data
            })
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

def api_iv(request):
    ticker = request.GET.get('ticker', 'AAPL')
    
    try:
        iv_info = calculate_iv_rank(ticker)
        iv_serializable = {k: float(v) for k, v in iv_info.items()}
        return JsonResponse({'success': True, 'iv_info': iv_serializable})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def api_trade_log(request):
    global trade_log  # ✅ Declare global FIRST
    action = request.GET.get('action', 'view')
    
    try:
        if action == 'view':
            if trade_log.empty:
                return JsonResponse({'success': True, 'trades': [], 'summary': None})
            
            trades = []
            for _, row in trade_log.iterrows():
                trade = {}
                for col in trade_log.columns:
                    value = row[col]
                    if isinstance(value, (pd.Timestamp, datetime)):
                        trade[col] = value.isoformat()
                    elif isinstance(value, (np.integer, np.floating)):
                        trade[col] = float(value)
                    else:
                        trade[col] = value
                trades.append(trade)
            
            summary = get_trade_summary()
            if summary:
                summary_serializable = {}
                for k, v in summary.items():
                    if isinstance(v, (np.integer, np.floating)):
                        summary_serializable[k] = float(v)
                    else:
                        summary_serializable[k] = v
            else:
                summary_serializable = None
            
            return JsonResponse({
                'success': True,
                'trades': trades,
                'summary': summary_serializable
            })
        
        elif action == 'clear':
            trade_log = pd.DataFrame(columns=trade_log.columns)
            return JsonResponse({'success': True})
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
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
import yfinance as yf
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
    
from .MCForecastTools import MCSimulation

def api_monte_carlo(request):
    """Run Monte Carlo simulation for a stock"""
    ticker = request.GET.get('ticker', 'AAPL')
    num_simulations = int(request.GET.get('simulations', 1000))
    years = int(request.GET.get('years', 5))
    investment = float(request.GET.get('investment', 10000))
    
    try:
        # Get historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{max(years, 1)}y")
        
        if hist.empty or len(hist) < 252:
            return JsonResponse({'success': False, 'error': 'Insufficient historical data'})
        
        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()
        
        # Run Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        simulation_results = []
        
        for _ in range(num_simulations):
            # Randomly sample returns with replacement
            simulated_returns = np.random.choice(returns, size=252*years, replace=True)
            # Calculate cumulative return
            cumulative_return = np.prod(1 + simulated_returns)
            simulation_results.append(cumulative_return)
        
        simulation_results = np.array(simulation_results)
        
        # Calculate statistics
        percentiles = {
            '10': float(np.percentile(simulation_results, 10)),
            '50': float(np.percentile(simulation_results, 50)),
            '90': float(np.percentile(simulation_results, 90))
        }
        
        # Calculate VaR and CVaR
        var_95 = float(np.percentile(simulation_results, 5))
        cvar_95 = float(simulation_results[simulation_results <= var_95].mean())
        
        # Calculate confidence interval
        ci_lower = float(np.percentile(simulation_results, 2.5))
        ci_upper = float(np.percentile(simulation_results, 97.5))
        
        return JsonResponse({
            'success': True,
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper
            },
            'var_95': round(investment * (1 - var_95), 2),
            'cvar_95': round(investment * (1 - cvar_95), 2),
            'percentiles': {
                '10': round(percentiles['10'], 4),
                '50': round(percentiles['50'], 4),
                '90': round(percentiles['90'], 4)
            }
        })
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    

def api_stock_backtest(request):
    """Backtest stock trading strategies"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract parameters
            ticker = data.get('ticker', 'AAPL')
            strategy = data.get('strategy', 'trend_following')
            entry_signal = data.get('entry_signal', 'rsi')
            timeframe = data.get('timeframe', '1d')
            position_type = data.get('position_type', 'long')
            position_size = float(data.get('position_size', 10000))
            stop_loss = float(data.get('stop_loss', 5)) / 100
            take_profit = float(data.get('take_profit', 10)) / 100
            max_hold = int(data.get('max_hold', 30))
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            
            if hist.empty:
                return JsonResponse({'success': False, 'error': 'No data available'})
            
            # Calculate indicators based on strategy
            hist['returns'] = hist['Close'].pct_change()
            hist['SMA20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            
            # RSI calculation
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Run backtest simulation
            trades = []
            in_position = False
            entry_price = 0
            entry_date = None
            
            for i in range(50, len(hist)):
                date = hist.index[i]
                price = hist['Close'].iloc[i]
                
                # Entry signals based on selection
                signal = False
                if entry_signal == 'rsi':
                    if position_type == 'long':
                        signal = hist['RSI'].iloc[i] < 30  # Oversold
                    else:
                        signal = hist['RSI'].iloc[i] > 70  # Overbought
                elif entry_signal == 'sma':
                    signal = (hist['SMA20'].iloc[i] > hist['SMA50'].iloc[i] and 
                             hist['SMA20'].iloc[i-1] <= hist['SMA50'].iloc[i-1])
                
                if signal and not in_position:
                    in_position = True
                    entry_price = price
                    entry_date = date
                    
                elif in_position:
                    days_held = (date - entry_date).days
                    return_pct = (price - entry_price) / entry_price
                    
                    # Exit conditions
                    exit_signal = False
                    exit_reason = ""
                    
                    if position_type == 'long':
                        if return_pct <= -stop_loss:
                            exit_signal = True
                            exit_reason = "stop_loss"
                        elif return_pct >= take_profit:
                            exit_signal = True
                            exit_reason = "take_profit"
                    else:  # short
                        if return_pct >= stop_loss:
                            exit_signal = True
                            exit_reason = "stop_loss"
                        elif return_pct <= -take_profit:
                            exit_signal = True
                            exit_reason = "take_profit"
                    
                    if days_held >= max_hold:
                        exit_signal = True
                        exit_reason = "max_hold"
                    
                    if exit_signal:
                        trades.append({
                            'entry_date': entry_date.strftime('%Y-%m-%d'),
                            'exit_date': date.strftime('%Y-%m-%d'),
                            'direction': position_type,
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(price, 2),
                            'return_pct': round(return_pct, 3),
                            'days_held': days_held,
                            'exit_reason': exit_reason
                        })
                        in_position = False
            
            # Calculate metrics
            if trades:
                returns = [t['return_pct'] for t in trades]
                winning_trades = [r for r in returns if r > 0]
                
                total_trades = len(trades)
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                total_return = sum(returns)
                avg_return = total_return / total_trades if total_trades > 0 else 0
                
                # Calculate max drawdown
                cumulative = np.cumprod(1 + np.array(returns))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
                
                # Sharpe ratio (simplified)
                sharpe = (avg_return / (np.std(returns) + 1e-6)) * np.sqrt(252) if len(returns) > 1 else 0
                
                # Profit factor
                gross_profit = sum([r for r in returns if r > 0])
                gross_loss = abs(sum([r for r in returns if r < 0]))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                return JsonResponse({
                    'success': True,
                    'total_trades': total_trades,
                    'win_rate': round(win_rate, 3),
                    'total_return': round(total_return, 3),
                    'avg_return': round(avg_return, 3),
                    'sharpe_ratio': round(sharpe, 2),
                    'max_drawdown': round(max_drawdown, 3),
                    'profit_factor': round(profit_factor, 2),
                    'trades': trades[-20:]  # Last 20 trades
                })
            else:
                return JsonResponse({'success': False, 'error': 'No trades generated'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

def api_options_backtest(request):
    """Backtest options credit spreads"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Extract parameters
            ticker = data.get('ticker', 'SPY')
            option_type = data.get('option_type', 'put_credit')
            entry_signal = data.get('entry_signal', 'rsi_oversold')
            short_delta = float(data.get('short_delta', 0.30))
            spread_width = float(data.get('spread_width', 5))
            dte = int(data.get('dte', 30))
            exit_dte = int(data.get('exit_dte', 5))
            risk_per_trade = float(data.get('risk_per_trade', 500))
            take_profit = float(data.get('take_profit', 50)) / 100
            stop_loss = float(data.get('stop_loss', 200)) / 100
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            
            if hist.empty:
                return JsonResponse({'success': False, 'error': 'No data available'})
            
            # Calculate RSI for signals
            delta = hist['Close'].diff()
            gain = delta.clip(lower=0).rolling(window=14).mean()
            loss = (-delta.clip(upper=0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            # Simplified backtest - in reality you'd need options chain data
            trades = []
            
            for i in range(50, len(hist) - dte):
                date = hist.index[i]
                price = hist['Close'].iloc[i]
                
                # Check entry signal
                signal = False
                if entry_signal == 'rsi_oversold' and option_type == 'put_credit':
                    signal = hist['RSI'].iloc[i] < 30
                elif entry_signal == 'rsi_overbought' and option_type == 'call_credit':
                    signal = hist['RSI'].iloc[i] > 70
                
                if signal:
                    # Simulated credit spread (simplified)
                    # In reality, you'd need options pricing model
                    credit = price * 0.02  # Simplified 2% credit
                    max_risk = spread_width * 100 - credit
                    
                    # Simulate outcome based on price movement
                    exit_idx = min(i + dte, len(hist) - 1)
                    exit_price = hist['Close'].iloc[exit_idx]
                    
                    # Simplified P&L
                    if option_type == 'put_credit':
                        # Bullish - want price above short strike
                        pnl = credit if exit_price >= price else -max_risk
                    else:
                        # Bearish - want price below short strike
                        pnl = credit if exit_price <= price else -max_risk
                    
                    trades.append({
                        'entry_date': date.strftime('%Y-%m-%d'),
                        'type': option_type,
                        'credit': round(credit, 2),
                        'max_risk': round(max_risk, 2),
                        'pnl': round(pnl, 2),
                        'return_pct': round(pnl / max_risk, 3),
                        'days_held': dte
                    })
            
            # Calculate metrics
            if trades:
                pnls = [t['pnl'] for t in trades]
                winning_trades = [p for p in pnls if p > 0]
                
                total_trades = len(trades)
                win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
                total_pnl = sum(pnls)
                avg_return = total_pnl / total_trades if total_trades > 0 else 0
                
                # Calculate max drawdown
                cumulative = np.cumsum(pnls)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = cumulative - running_max
                max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
                
                # Sharpe ratio
                sharpe = (avg_return / (np.std(pnls) + 1e-6)) * np.sqrt(252) if len(pnls) > 1 else 0
                
                return JsonResponse({
                    'success': True,
                    'total_trades': total_trades,
                    'win_rate': round(win_rate, 3),
                    'total_pnl': round(total_pnl, 2),
                    'avg_return': round(avg_return, 2),
                    'sharpe_ratio': round(sharpe, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'trades': trades[-20:]
                })
            else:
                return JsonResponse({'success': False, 'error': 'No trades generated'})
                
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Invalid method'})

def api_trade_log(request):
    """Handle trade log operations"""
    global trade_log
    
    # Initialize empty trade_log if it doesn't exist
    if 'trade_log' not in globals() or trade_log.empty:
        trade_log = pd.DataFrame(columns=[
            "Date", "Ticker", "Position_Type", "Entry_Price", "Exit_Price",
            "Quantity", "PnL", "PnL_Percent", "Hold_Days", "Strategy",
            "Signal_Type", "Signal_Value", "Alpha_Score", "Market_State", "Notes"
        ])
    
    action = request.GET.get('action', 'view')
    
    # Handle different actions
    if action == 'view':
        if trade_log.empty:
            return JsonResponse({'success': True, 'trades': [], 'summary': None})
        
        # Convert trades to JSON-serializable format
        trades = []
        for _, row in trade_log.iterrows():
            trade = {}
            for col in trade_log.columns:
                value = row[col]
                if isinstance(value, (pd.Timestamp, datetime)):
                    trade[col] = value.strftime('%Y-%m-%d %H:%M')
                elif isinstance(value, (np.integer, np.floating, float, int)):
                    trade[col] = float(value) if not pd.isna(value) else 0
                else:
                    trade[col] = str(value) if value is not None else ''
            trades.append(trade)
        
        # Calculate summary statistics
        summary = {
            'Total Trades': len(trades),
            'Total P&L': round(float(trade_log['PnL'].sum()), 2),
            'Win Rate': round(float((trade_log['PnL'] > 0).mean() * 100), 1),
            'Avg P&L': round(float(trade_log['PnL'].mean()), 2),
            'Max Win': round(float(trade_log['PnL'].max()), 2),
            'Max Loss': round(float(trade_log['PnL'].min()), 2)
        }
        
        return JsonResponse({
            'success': True,
            'trades': trades,
            'summary': summary
        })
    
    elif action == 'add' and request.method == 'POST':
        try:
            data = json.loads(request.body)
            
            # Calculate P&L
            entry = float(data.get('entry_price', 0))
            exit_price = float(data.get('exit_price', 0))
            quantity = float(data.get('quantity', 0))
            position_type = data.get('position_type', 'LONG')
            
            pnl = (exit_price - entry) * quantity
            if position_type == 'SHORT':
                pnl = -pnl
            
            pnl_percent = ((exit_price - entry) / entry) * 100 if entry != 0 else 0
            if position_type == 'SHORT':
                pnl_percent = -pnl_percent
            
            # Create new trade
            new_trade = {
                "Date": datetime.now(),
                "Ticker": data.get('ticker', ''),
                "Position_Type": position_type,
                "Entry_Price": round(entry, 2),
                "Exit_Price": round(exit_price, 2),
                "Quantity": quantity,
                "PnL": round(pnl, 2),
                "PnL_Percent": round(pnl_percent, 2),
                "Hold_Days": int(data.get('hold_days', 0)),
                "Strategy": data.get('strategy', 'Manual'),
                "Signal_Type": data.get('signal_type', 'Manual'),
                "Signal_Value": float(data.get('signal_value', 0)),
                "Alpha_Score": float(data.get('alpha_score', 0)),
                "Market_State": data.get('market_state', 'UNKNOWN'),
                "Notes": data.get('notes', '')
            }
            
            # Add to trade_log
            trade_log = pd.concat([trade_log, pd.DataFrame([new_trade])], ignore_index=True)
            
            return JsonResponse({'success': True, 'message': 'Trade added successfully'})
            
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    elif action == 'clear':
        if request.method == 'POST':
            trade_log = pd.DataFrame(columns=trade_log.columns)
            return JsonResponse({'success': True, 'message': 'Trade log cleared'})
        return JsonResponse({'success': False, 'error': 'Invalid method'})
    
    elif action == 'export':
        # Return CSV data
        csv_data = trade_log.to_csv(index=False)
        return HttpResponse(csv_data, content_type='text/csv')
    
    return JsonResponse({'success': False, 'error': 'Invalid action'})
from django.shortcuts import render
from django.shortcuts import redirect
from django.http import JsonResponse
from .models import UserWatchlist
from .economic_utils import get_all_economic_data
from .services import get_market_news
from .vertical_analyzer import VerticalSpreadAnalyzer
import yfinance as yf
from .services import (
    rank_stocks,
    build_watchlist,
    build_watchlist_for_web,
    get_market_news
)
import logging

logger = logging.getLogger(__name__)

# ---------- WATCHLIST FUNCTIONS ----------
def watchlist_view(request):
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
    stocks = build_watchlist(tickers)
    return render(request, "core/watchlist.html", {"stocks": stocks})

def dashboard(request):
    tickers = list(UserWatchlist.objects.values_list("ticker", flat=True))
    stocks = build_watchlist_for_web(tickers)
    return render(request, "core/dashboard.html", {"stocks": stocks})

def add_stock(request):
    if request.method == "POST":
        ticker = request.POST.get("ticker", "").upper()
        if ticker:
            UserWatchlist.objects.get_or_create(ticker=ticker)
    return redirect("dashboard")

def remove_stock(request, ticker):
    UserWatchlist.objects.filter(ticker=ticker).delete()
    return redirect("dashboard")

# ---------- HOME PAGE ----------
def home(request):
    """Home page - shows economic data and news"""
    eco_data = get_all_economic_data()
    news = get_market_news()
    return render(request, "core/base.html", {
        'eco': eco_data,
        'news': news
    })

# ---------- ANALYSIS PAGE (COMBINED) ----------
def analysis(request):
    """Analysis page - with universe data for templates"""
    universe = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE",
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
        "XOM", "CVX", "COP", "SLB", "OXY",
        "CAT", "DE", "BA", "GE", "LMT", "RTX",
        "TSLA", "HD", "LOW", "MCD", "NKE", "COST", "WMT",
        "LLY", "JNJ", "UNH", "PFE", "MRK", "ABBV",
        "AMD", "INTC", "QCOM", "TXN", "MU",
        "DIS", "NFLX", "T", "VZ"
    ]
    return render(request, "core/analysis.html", {'universe': universe})

# ---------- RANKING PAGE ----------
def ranking(request):
    return render(request, "core/ranking.html")

# ---------- BACKTEST PAGE (COMBINED) ----------
def backtest(request):
    """Backtest page - with universe data for templates"""
    universe = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE",
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
        "XOM", "CVX", "COP", "SLB", "OXY",
        "CAT", "DE", "BA", "GE", "LMT", "RTX",
        "TSLA", "HD", "LOW", "MCD", "NKE", "COST", "WMT",
        "LLY", "JNJ", "UNH", "PFE", "MRK", "ABBV",
        "AMD", "INTC", "QCOM", "TXN", "MU",
        "DIS", "NFLX", "T", "VZ"
    ]
    return render(request, "core/backtest.html", {'universe': universe})

# ---------- IV ANALYSIS PAGE ----------
def iv(request):
    """IV Analysis page"""
    universe = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL", "CRM", "ADBE",
        "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
        "XOM", "CVX", "COP", "SLB", "OXY",
        "CAT", "DE", "BA", "GE", "LMT", "RTX",
        "TSLA", "HD", "LOW", "MCD", "NKE", "COST", "WMT",
        "LLY", "JNJ", "UNH", "PFE", "MRK", "ABBV",
        "AMD", "INTC", "QCOM", "TXN", "MU",
        "DIS", "NFLX", "T", "VZ", "SPY", "QQQ", "IWM"
    ]
    return render(request, "core/iv.html", {'universe': universe})

# ---------- TRADE LOG PAGE ----------
def trade_log(request):
    return render(request, "core/trade_log.html")

# ---------- API IV FUNCTION ----------
def api_iv(request):
    ticker = request.GET.get('ticker', 'AAPL')
    try:
        # You need to define calculate_iv_rank or import it
        # For now, return placeholder
        iv_info = {'iv_rank': 0.5, 'iv_percentile': 0.5}
        iv_serializable = {k: float(v) for k, v in iv_info.items()}
        return JsonResponse({'success': True, 'iv_info': iv_serializable})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

# ---------- VERTICAL SPREAD OPPORTUNITIES ----------
def find_opportunities(request):
    """View to display vertical spread opportunities"""
    try:
        analyzer = VerticalSpreadAnalyzer()
        symbol = request.GET.get('symbol', 'SPY')
        opportunities = analyzer.find_bull_put_spreads(symbol)
        return render(request, 'core/opportunities.html', {
            'symbol': symbol,
            'opportunities': opportunities[:10] if opportunities else [],
        })
    except Exception as e:
        logger.error(f"Error in find_opportunities: {e}")
        return render(request, 'core/opportunities.html', {
            'symbol': request.GET.get('symbol', 'SPY'),
            'opportunities': [],
            'error': str(e)
        })

def api_opportunities(request):
    """API endpoint for opportunities"""
    try:
        analyzer = VerticalSpreadAnalyzer()
        symbol = request.GET.get('symbol', 'SPY')
        opportunities = analyzer.find_bull_put_spreads(symbol)
        return JsonResponse({
            'symbol': symbol,
            'opportunities': opportunities[:10] if opportunities else []
        }, safe=False)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
    
# ---------- ALPACA TEST FUNCTIONS ----------
def test_alpaca_connection(request):
    """Test Alpaca connection"""
    try:
        from .alpaca_client import AlpacaService
        alpaca = AlpacaService()
        account_info = alpaca.get_account_info()
        
        # Create a simple template if it doesn't exist
        html = f"""
        <html>
        <head><title>Alpaca Test</title></head>
        <body>
            <h1>Alpaca Connection Test</h1>
            <div style="padding:20px; background:#f0f0f0;">
                <h2>Account Information:</h2>
                {'<p style="color:green">✅ Connected to Alpaca!</p>' if account_info else '<p style="color:red">❌ Could not connect</p>'}
                {'<ul>' + 
                 '<li>Equity: $' + str(account_info.get('equity', 'N/A')) + '</li>' +
                 '<li>Buying Power: $' + str(account_info.get('buying_power', 'N/A')) + '</li>' +
                 '<li>Cash: $' + str(account_info.get('cash', 'N/A')) + '</li>' +
                 '<li>Status: ' + str(account_info.get('status', 'N/A')) + '</li>' +
                 '</ul>' if account_info else ''}
            </div>
            <p><a href="/opportunities/">View Vertical Spread Opportunities</a></p>
        </body>
        </html>
        """
        from django.http import HttpResponse
        return HttpResponse(html)
    except Exception as e:
        from django.http import HttpResponse
        return HttpResponse(f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>")

def api_test_options(request):
    """API endpoint to test options data"""
    try:
        from .alpaca_client import AlpacaService
        from django.http import JsonResponse
        alpaca = AlpacaService()
        symbol = request.GET.get('symbol', 'SPY')
        chain = alpaca.get_options_chain(symbol)
        
        return JsonResponse({
            'symbol': symbol,
            'underlying_price': float(chain.underlying_price) if chain and chain.underlying_price else None,
            'expiration': chain.expiration_date if chain else None,
            'total_options': (len(chain.puts) + len(chain.calls)) if chain else 0
        })
    except Exception as e:
        from django.http import JsonResponse
        return JsonResponse({'error': str(e)}, status=500)

# Make sure these already exist in your views.py
# If not, add them too:
def find_opportunities(request):
    """View to display vertical spread opportunities with debugging"""
    try:
        from .vertical_analyzer import VerticalSpreadAnalyzer
        from django.http import HttpResponse
        import traceback
        import json
        
        analyzer = VerticalSpreadAnalyzer()
        symbol = request.GET.get('symbol', 'SPY')
        symbol = symbol.upper()  # Force uppercase
        
        # Debug: Check Alpaca connection first
        debug_info = []
        debug_info.append(f"<h3>Debug Info for {symbol}:</h3>")
        
        # Check if we can get options chain with improved error handling
        try:
            from .alpaca_client import AlpacaService
            alpaca = AlpacaService()
            chain = alpaca.get_options_chain(symbol)
            
            if chain:
                debug_info.append(f"✅ Got options chain for {symbol}")
                debug_info.append(f"Underlying price: ${chain.underlying_price if chain.underlying_price else 'N/A'}")
                debug_info.append(f"Expiration: {chain.expiration_date if chain.expiration_date else 'N/A'}")
                
                # Check if puts and calls exist
                puts_count = len(chain.puts) if chain.puts else 0
                calls_count = len(chain.calls) if chain.calls else 0
                debug_info.append(f"Number of puts: {puts_count}")
                debug_info.append(f"Number of calls: {calls_count}")
                
                # Show first few puts as sample if they exist
                if puts_count > 0:
                    debug_info.append("<h4>Sample puts (first 3):</h4>")
                    for i, put in enumerate(chain.puts[:3]):
                        bid = float(put.bid) if put.bid else 0
                        ask = float(put.ask) if put.ask else 0
                        delta = put.greeks.delta if put.greeks else 'N/A'
                        debug_info.append(f"Strike: ${put.strike}, Bid: ${bid}, Ask: ${ask}, Delta: {delta}")
                else:
                    debug_info.append("❌ No put options found in chain")
                    
                # Check if there are options with bids
                if puts_count > 0:
                    has_bids = any(put.bid and float(put.bid) > 0 for put in chain.puts)
                    debug_info.append(f"Puts with bids: {'✅ Yes' if has_bids else '❌ No'}")
            else:
                debug_info.append(f"❌ No options chain returned for {symbol}")
                
        except Exception as e:
            debug_info.append(f"❌ Error fetching options: {str(e)}")
            debug_info.append(f"Traceback: {traceback.format_exc().replace(chr(10), '<br>')}")
        
        # Now try to find spreads
        debug_info.append("<h4>Finding spreads...</h4>")
        try:
            opportunities = analyzer.find_bull_put_spreads(symbol)
            debug_info.append(f"Found {len(opportunities)} opportunities")
            
            # Log the first opportunity as sample if found
            if opportunities:
                debug_info.append("<h4>Sample opportunity (first):</h4>")
                first = opportunities[0]
                debug_info.append(f"Strike: {first['short_strike']:.2f}/{first['long_strike']:.2f}, Credit: ${first['credit_received']:.2f}, ROI: {first['roi_pct']}%")
        except Exception as e:
            debug_info.append(f"❌ Error finding spreads: {str(e)}")
            debug_info.append(f"Traceback: {traceback.format_exc().replace(chr(10), '<br>')}")
            opportunities = []
        
        # Build the HTML
        html = """
        <html>
        <head>
            <title>Debug: Bull Put Opportunities</title>
            <style>
                body { font-family: Arial; padding: 20px; background: #f5f5f5; }
                h1 { color: #333; }
                h3 { color: #555; margin-top: 20px; }
                h4 { color: #666; margin-top: 15px; margin-bottom: 5px; }
                table { width: 100%%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                th { background: #1f2933; color: white; padding: 12px; text-align: left; }
                td { padding: 12px; border-bottom: 1px solid #ddd; }
                tr:hover { background: #f9f9f9; }
                .positive { color: green; font-weight: bold; }
                .symbol-input { padding: 10px; font-size: 16px; margin: 20px 0; width: 200px; }
                .button { padding: 10px 20px; background: #fbbf24; border: none; cursor: pointer; font-weight: bold; }
                .button:hover { background: #f59e0b; }
                .container { max-width: 1200px; margin: 0 auto; }
                .debug { background: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; font-family: monospace; overflow-x: auto; }
                .error { background: #fee; color: #c00; }
                .success { background: #efe; color: #0a0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Bull Put Opportunities Debug</h1>
                
                <form method="get">
                    <input type="text" name="symbol" value=\"""" + symbol + """\" class="symbol-input" placeholder="Enter symbol (e.g., SPY)">
                    <button type="submit" class="button">Analyze</button>
                </form>
                
                <div class="debug">
                    <h3>Debug Information:</h3>
                    """ + "<br>".join(debug_info) + """
                </div>
        """
        
        if opportunities:
            html += """
                <h3>Top Opportunities:</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Strike</th>
                            <th>Credit</th>
                            <th>Max Risk</th>
                            <th>ROI %</th>
                            <th>Prob OTM %</th>
                            <th>Delta</th>
                            <th>IV %</th>
                            <th>Adj Return</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for opp in opportunities[:15]:
                iv_display = f"{opp['short_iv']:.1f}" if opp['short_iv'] else 'N/A'
                html += f"""
                        <tr>
                            <td><strong>{opp['short_strike']:.2f}/{opp['long_strike']:.2f}</strong></td>
                            <td class="positive">${opp['credit_received']:.2f}</td>
                            <td>${opp['max_risk']:.0f}</td>
                            <td class="positive">{opp['roi_pct']}%</td>
                            <td>{opp['probability_otm']}%</td>
                            <td>{opp['short_delta']:.3f}</td>
                            <td>{iv_display}</td>
                            <td><strong>{opp['pop_adjusted_return']}%</strong></td>
                        </tr>
                """
            
            html += """
                    </tbody>
                </table>
            """
        else:
            html += "<p style='color:red; font-size:18px;'>❌ No opportunities found. Check debug info above.</p>"
        
        html += """
                <p style="margin-top:20px; color:#666;">
                    <small>Debug mode - showing API response details</small>
                </p>
                <p>
                    <a href="/opportunities/?symbol=SPY">Try SPY</a> | 
                    <a href="/opportunities/?symbol=QQQ">Try QQQ</a> | 
                    <a href="/opportunities/?symbol=AAPL">Try AAPL</a> | 
                    <a href="/opportunities/?symbol=MSFT">Try MSFT</a>
                </p>
            </div>
        </body>
        </html>
        """
        
        return HttpResponse(html)
        
    except Exception as e:
        import traceback
        from django.http import HttpResponse
        return HttpResponse(f"""
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Analyzing Opportunities</h1>
            <p style="color:red">{str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
            <a href="/opportunities/?symbol=SPY">Try again</a>
        </body>
        </html>
        """)

def api_opportunities(request):
    """API endpoint for opportunities"""
    try:
        from .vertical_analyzer import VerticalSpreadAnalyzer
        from django.http import JsonResponse
        analyzer = VerticalSpreadAnalyzer()
        symbol = request.GET.get('symbol', 'SPY')
        symbol = symbol.upper() 
        opportunities = analyzer.find_bull_put_spreads(symbol)
        
        return JsonResponse({
            'symbol': symbol,
            'opportunities': opportunities[:10] if opportunities else []
        }, safe=False)
    except Exception as e:
        from django.http import JsonResponse
        return JsonResponse({'error': str(e)}, status=500)
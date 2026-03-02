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
    """View to display vertical spread opportunities"""
    try:
        from .vertical_analyzer import VerticalSpreadAnalyzer
        analyzer = VerticalSpreadAnalyzer()
        symbol = request.GET.get('symbol', 'SPY')
        opportunities = analyzer.find_bull_put_spreads(symbol)
        
        # Create a styled HTML table
        html = """
        <html>
        <head>
            <title>Bull Put Opportunities</title>
            <style>
                body { font-family: Arial; padding: 20px; background: #f5f5f5; }
                h1 { color: #333; }
                table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                th { background: #1f2933; color: white; padding: 12px; text-align: left; }
                td { padding: 12px; border-bottom: 1px solid #ddd; }
                tr:hover { background: #f9f9f9; }
                .positive { color: green; font-weight: bold; }
                .symbol-input { padding: 10px; font-size: 16px; margin: 20px 0; width: 200px; }
                .button { padding: 10px 20px; background: #fbbf24; border: none; cursor: pointer; font-weight: bold; }
                .button:hover { background: #f59e0b; }
                .container { max-width: 1200px; margin: 0 auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Bull Put Opportunities for <span id="symbol">""" + symbol + """</span></h1>
                
                <form method="get">
                    <input type="text" name="symbol" value="""" + symbol + """" class="symbol-input" placeholder="Enter symbol (e.g., SPY)">
                    <button type="submit" class="button">Analyze</button>
                </form>
                
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
        
        for opp in opportunities[:15]:  # Show top 15
            html += f"""
                        <tr>
                            <td><strong>{opp['short_strike']:.2f}/{opp['long_strike']:.2f}</strong></td>
                            <td class="positive">${opp['credit_received']:.2f}</td>
                            <td>${opp['max_risk']:.0f}</td>
                            <td class="positive">{opp['roi_pct']}%</td>
                            <td>{opp['probability_otm']}%</td>
                            <td>{opp['short_delta']:.3f}</td>
                            <td>{opp['short_iv'] if opp['short_iv'] else 'N/A'}</td>
                            <td><strong>{opp['pop_adjusted_return']}%</strong></td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
                <p style="margin-top:20px; color:#666;">
                    <small>Sorted by Probability-Adjusted Return (best first)</small>
                </p>
            </div>
        </body>
        </html>
        """
        
        from django.http import HttpResponse
        return HttpResponse(html)
        
    except Exception as e:
        from django.http import HttpResponse
        return HttpResponse(f"""
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Error Analyzing Opportunities</h1>
            <p style="color:red">{str(e)}</p>
            <a href="/opportunities/?symbol=SPY">Try SPY instead</a>
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
        opportunities = analyzer.find_bull_put_spreads(symbol)
        
        return JsonResponse({
            'symbol': symbol,
            'opportunities': opportunities[:10] if opportunities else []
        }, safe=False)
    except Exception as e:
        from django.http import JsonResponse
        return JsonResponse({'error': str(e)}, status=500)
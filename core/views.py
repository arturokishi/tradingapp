from django.shortcuts import render
from django.shortcuts import redirect
from .models import UserWatchlist
from django.shortcuts import render
from .economic_utils import get_all_economic_data  # Import your new function
from .services import get_market_news  # Your existing news function

# trading_views.py - Add this import at the top
import yfinance as yf

from .services import (
    rank_stocks,
    build_watchlist,
    build_watchlist_for_web,
    get_market_news
)


def analysis(request):
    return render(request, "core/analysis.html")

def ranking(request):
    return render(request, "core/ranking.html")

def backtest(request):
    return render(request, "core/backtest.html")

def iv(request):
    return render(request, "core/iv.html")

def trade_log(request):
    return render(request, "core/trade_log.html")

def watchlist_view(request):
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
    stocks = build_watchlist(tickers)

    return render(request, "core/watchlist.html", {
        "stocks": stocks
    })

def dashboard(request):
    tickers = list(UserWatchlist.objects.values_list("ticker", flat=True))

    stocks = build_watchlist_for_web(tickers)

    return render(request, "core/dashboard.html", {
        "stocks": stocks
    })



def add_stock(request):
    if request.method == "POST":
        ticker = request.POST.get("ticker", "").upper()
        if ticker:
            UserWatchlist.objects.get_or_create(ticker=ticker)
    return redirect("dashboard")

def remove_stock(request, ticker):
    UserWatchlist.objects.filter(ticker=ticker).delete()
    return redirect("dashboard")



def home(request):
    """Home page - shows economic data and news"""
    eco_data = get_all_economic_data()
    news = get_market_news()
    
    return render(request, "core/base.html", {  # Changed from home.html to base.html
        'eco': eco_data,
        'news': news
    })

def analysis(request):
    # Define your universe of stocks
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
    
    return render(request, "core/analysis.html", {
        'universe': universe  # This passes the list to the template
    })


def backtest(request):
    # Define your universe of stocks
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
    
    return render(request, "core/backtest.html", {
        'universe': universe  # This passes the list to the template
    })


def api_iv(request):
    ticker = request.GET.get('ticker', 'AAPL')
    
    try:
        iv_info = calculate_iv_rank(ticker)
        iv_serializable = {k: float(v) for k, v in iv_info.items()}
        return JsonResponse({'success': True, 'iv_info': iv_serializable})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
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
    
    return render(request, "core/iv.html", {
        'universe': universe
    })
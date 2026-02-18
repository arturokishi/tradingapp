from django.shortcuts import render
from django.shortcuts import redirect
from .models import UserWatchlist
from django.shortcuts import render
from .economic_utils import get_all_economic_data  # Import your new function
from .services import get_market_news  # Your existing news function


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
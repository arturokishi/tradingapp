from django.shortcuts import render
from .services import (
    rank_stocks,
    build_watchlist,
    build_watchlist_for_web,
    get_market_news
)

def home(request):
    return render(request, "core/base.html")

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
    UNIVERSE = [
        "AAPL","MSFT","NVDA","GOOGL","META",
        "AMZN","AVGO","ORCL","CRM","ADBE",
        "JPM","BAC","WFC","GS","MS"
    ]

    stocks = build_watchlist_for_web(UNIVERSE)
    ranking = rank_stocks(UNIVERSE)
    news = get_market_news()

    return render(request, "core/dashboard.html", {
        "stocks": stocks,
        "ranking": ranking,
        "news": news
    })

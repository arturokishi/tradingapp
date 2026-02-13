from django.shortcuts import render
from .services import rank_stocks
from .services import build_watchlist
from core.services import build_watchlist_for_web
from .services import build_watchlist_for_web, rank_stocks




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




from .services import rank_stocks, get_market_news


def dashboard(request):
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]

    ranking = rank_stocks(tickers)
    news = get_market_news()
    stocks = build_watchlist(tickers)

    context = {
        "ranking": ranking,
        "news": news,
        "stocks": stocks
    }

    return render(request, "core/dashboard.html", context)
   



def analysis(request):
    return render(request, "core/analysis.html")

def watchlist_view(request):
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]

    stocks = build_watchlist(tickers)

    context = {
        "stocks": stocks
    }

    return render(request, "core/watchlist.html", context)









def dashboard(request):
    UNIVERSE = [
        "AAPL","MSFT","NVDA","GOOGL","META",
        "AMZN","AVGO","ORCL","CRM","ADBE",
        "JPM","BAC","WFC","GS","MS"
    ]

    stocks = build_watchlist_for_web(UNIVERSE)
    ranking = rank_stocks(UNIVERSE)

    return render(request, "core/dashboard.html", {
        "stocks": stocks,
        "ranking": ranking,
        "news": None
    })

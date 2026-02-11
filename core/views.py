from django.shortcuts import render
from .services import rank_stocks

def home(request):
    return render(request, "core/base.html")



def dashboard(request):
    return render(request, "core/dashboard.html")

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
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]

    ranking = rank_stocks(tickers)
    news = get_market_news()

    context = {
        "ranking": ranking,
        "news": news
    }

    return render(request, "core/dashboard.html", context)


def analysis(request):
    return render(request, "core/analysis.html")

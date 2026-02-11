from .services import get_market_news

def market_news(request):
    return {
        "news": get_market_news(limit=5)
    }

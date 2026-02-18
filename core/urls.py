

from django.urls import path
from . import views
from . import trading_views  # Add this line

urlpatterns = [
    # Existing paths
    path('', views.home, name='base'),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('analysis/', views.analysis, name="analysis"),
    path('ranking/', views.ranking, name="ranking"),
    path('backtest/', views.backtest, name="backtest"),
    path('iv/', views.iv, name="iv"),
    path('trade-log/', views.trade_log, name="trade_log"),
    path("watchlist/", views.watchlist_view, name="watchlist"),
    path("add/", views.add_stock, name="add_stock"),
    path("remove/<str:ticker>/", views.remove_stock, name="remove_stock"),
    
    # New trading platform paths
    path('trading/', trading_views.trading_dashboard, name='trading_dashboard'),
    path('api/trading/watchlist/', trading_views.api_watchlist, name='api_trading_watchlist'),
    path('api/trading/analysis/', trading_views.api_analysis, name='api_trading_analysis'),
    path('api/trading/ranking/', trading_views.api_ranking, name='api_trading_ranking'),
    path('api/trading/backtest/', trading_views.api_backtest, name='api_trading_backtest'),
    path('api/trading/iv/', trading_views.api_iv, name='api_trading_iv'),
    path('api/trading/trade-log/', trading_views.api_trade_log, name='api_trading_trade_log'),
]



from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='base'),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('analysis/', views.analysis, name="analysis"),
    path('ranking/', views.ranking, name="ranking"),
    path('backtest/', views.backtest, name="backtest"),
    path('iv/', views.iv, name="iv"),
    path('trade-log/', views.trade_log, name="trade_log"),
]



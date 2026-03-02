# core/alpaca_client.py
from django.conf import settings
from alpaca.trading.client import TradingClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AlpacaService:
    """
    Service to handle Alpaca API interactions
    """
    
    def __init__(self):
        self.trading_client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=settings.ALPACA_PAPER_TRADE
        )
        
        self.data_client = OptionHistoricalDataClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY
        )
        
        logger.info("Alpaca service initialized")
    
    def get_account_info(self):
        """Get account information"""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'status': account.status
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_options_chain(self, symbol, expiration_date=None):
        """Get options chain for a symbol"""
        try:
            if not expiration_date:
                # Get next monthly expiration (approx 30-45 days)
                expiration_date = (datetime.now() + timedelta(days=35)).strftime('%Y-%m-%d')
            
            request = OptionChainRequest(
                underlying_symbol=symbol,
                expiration_date=expiration_date
            )
            
            chain = self.data_client.get_option_chain(request)
            return chain
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None
# economic_utils.py
import requests
import yfinance as yf
from datetime import datetime

# Your FRED API key
FRED_API_KEY = "9d2e1939667ce3fbe5cd39210*****"  # Your key here

def get_fred_data(series_id):
    """Fetch latest observation from FRED"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'limit': 1,
        'sort_order': 'desc'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'observations' in data and data['observations']:
            value = data['observations'][0]['value']
            date = data['observations'][0]['date']
            return float(value) if value != '.' else None, date
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
    
    return None, None

def get_sector_performance():
    """Get sector ETF performance from Yahoo Finance"""
    sectors = {
        'XLB': 'Basic Materials',
        'XLY': 'Consumer Cyclical',
        'XLF': 'Financials',
        'VNQ': 'Real Estate',
        'XLP': 'Consumer Defensive',
        'XLV': 'Healthcare',
        'XLU': 'Utilities',
        'XTL': 'Communication',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLK': 'Technology'
    }
    
    sector_data = {}
    for ticker, name in sectors.items():
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                pct_change = ((current - prev) / prev) * 100
                sector_data[ticker] = {
                    'name': name,
                    'change': round(pct_change, 2)
                }
        except:
            pass
    
    return sector_data

def get_market_data():
    """Get VIX, Oil, Dollar Index from Yahoo Finance"""
    market = {}
    
    try:
        vix = yf.Ticker("^VIX")
        market['VIX'] = round(vix.history(period="1d")['Close'].iloc[-1], 2)
    except:
        market['VIX'] = None
    
    try:
        oil = yf.Ticker("CL=F")
        market['OIL'] = round(oil.history(period="1d")['Close'].iloc[-1], 2)
    except:
        market['OIL'] = None
    
    try:
        dx = yf.Ticker("DX-Y.NYB")
        market['DXY'] = round(dx.history(period="1d")['Close'].iloc[-1], 2)
    except:
        market['DXY'] = None
    
    try:
        tnx = yf.Ticker("^TNX")
        yield_val = tnx.history(period="1d")['Close'].iloc[-1]
        market['TNX'] = round(yield_val, 2)
    except:
        market['TNX'] = None
    
    return market

def get_all_economic_data():
    """Fetch all economic data"""
    data = {}
    
    # FRED Series
    fred_series = {
        'GDP': 'GDP',
        'GDP_CHANGE': 'A191RL1Q225SBEA',  # Real GDP percent change
        'CPI': 'CPIAUCSL',
        'CPI_CHANGE': 'CPIAUCSL',  # We'll calculate change
        'UNEMPLOYMENT': 'UNRATE',
        'FEDFUNDS': 'FEDFUNDS',
        'DGS10': 'DGS10'  # 10Y Treasury
    }
    
    # Get latest values
    for key, series_id in fred_series.items():
        value, date = get_fred_data(series_id)
        data[key] = {'value': value, 'date': date}
    
    # Calculate CPI change if we have previous month
    if data.get('CPI') and data['CPI']['value']:
        # Get previous month for comparison
        prev_value, _ = get_fred_data_with_offset('CPIAUCSL', offset=1)
        if prev_value:
            cpi_change = ((data['CPI']['value'] - prev_value) / prev_value) * 100
            data['CPI_CHANGE'] = {'value': round(cpi_change, 1), 'date': data['CPI']['date']}
    
    # Get market data
    data['MARKET'] = get_market_data()
    
    # Get sector performance
    data['SECTORS'] = get_sector_performance()
    
    return data

def get_fred_data_with_offset(series_id, offset=1):
    """Get data from offset months ago"""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'limit': offset + 1,
        'sort_order': 'desc'
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'observations' in data and len(data['observations']) > offset:
            value = data['observations'][offset]['value']
            date = data['observations'][offset]['date']
            return float(value) if value != '.' else None, date
    except:
        pass
    
    return None, None
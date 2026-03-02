# core/vertical_analyzer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .alpaca_client import AlpacaService
import logging

logger = logging.getLogger(__name__)

class VerticalSpreadAnalyzer:
    """
    Analyze and find optimal bull put and bear call vertical spreads
    """
    
    def __init__(self):
        self.alpaca = AlpacaService()
    
    def find_bull_put_spreads(self, symbol, days_to_expiry=35, 
                               min_credit=0.10, max_spread_width=5.0,
                               delta_range=(0.20, 0.35)):
        """
        Find the best bull put spreads (credit spreads) for a symbol
        """
        # Calculate expiration date
        exp_date = (datetime.now() + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')
        
        # Get options chain
        chain = self.alpaca.get_options_chain(symbol, exp_date)
        if not chain:
            logger.error(f"Could not fetch options chain for {symbol}")
            return []
        
        # Convert puts to DataFrame
        puts_data = []
        for put in chain.puts:
            puts_data.append({
                'strike': float(put.strike),
                'bid': float(put.bid) if put.bid else 0,
                'ask': float(put.ask) if put.ask else 0,
                'delta': float(put.greeks.delta) if put.greeks else None,
                'gamma': float(put.greeks.gamma) if put.greeks else None,
                'theta': float(put.greeks.theta) if put.greeks else None,
                'vega': float(put.greeks.vega) if put.greeks else None,
                'iv': float(put.implied_volatility) if put.implied_volatility else None,
                'open_interest': float(put.open_interest) if put.open_interest else 0
            })
        
        puts_df = pd.DataFrame(puts_data)
        
        # Filter out illiquid options
        puts_df = puts_df[puts_df['bid'] > 0.05]
        puts_df = puts_df[puts_df['ask'] > 0.05]
        
        if puts_df.empty:
            return []
        
        # Sort by strike (descending)
        puts_df = puts_df.sort_values('strike', ascending=False)
        
        spreads = []
        underlying_price = float(chain.underlying_price) if chain.underlying_price else 0
        
        # Find potential spreads
        for i, short_put in puts_df.iterrows():
            # Check if short put delta is in target range
            if short_put['delta'] and abs(short_put['delta']) < delta_range[0] or abs(short_put['delta']) > delta_range[1]:
                continue
            
            # Find long puts (lower strike)
            long_puts = puts_df[puts_df['strike'] < short_put['strike']]
            
            for j, long_put in long_puts.iterrows():
                spread_width = short_put['strike'] - long_put['strike']
                
                if spread_width > max_spread_width:
                    continue
                
                # Calculate credit received
                credit = short_put['bid'] - long_put['ask']
                
                if credit < min_credit:
                    continue
                
                # Calculate max risk
                max_risk = (spread_width * 100) - (credit * 100)
                roi_pct = (credit * 100) / max_risk if max_risk > 0 else 0
                
                # Probability of profit (simplified)
                prob_otm = 1 - abs(short_put['delta']) if short_put['delta'] else 0.5
                
                spreads.append({
                    'symbol': symbol,
                    'strategy': 'BULL PUT',
                    'expiration': exp_date,
                    'short_strike': short_put['strike'],
                    'long_strike': long_put['strike'],
                    'spread_width': spread_width,
                    'credit_received': credit,
                    'max_risk': max_risk,
                    'roi_pct': round(roi_pct * 100, 2),
                    'probability_otm': round(prob_otm * 100, 2),
                    'short_delta': round(short_put['delta'], 3) if short_put['delta'] else None,
                    'short_iv': round(short_put['iv'] * 100, 2) if short_put['iv'] else None,
                    'underlying_price': underlying_price,
                    'pop_adjusted_return': round(roi_pct * prob_otm * 100, 2)
                })
        
        # Sort by probability-adjusted return
        return sorted(spreads, key=lambda x: x['pop_adjusted_return'], reverse=True)
    
    def find_bear_call_spreads(self, symbol, days_to_expiry=35,
                                min_credit=0.10, max_spread_width=5.0,
                                delta_range=(0.20, 0.35)):
        """
        Find the best bear call spreads (credit spreads) for a symbol
        """
        # Similar implementation for bear calls
        # We'll add this next
        pass
    
    def analyze_spread_risk(self, spread):
        """
        Analyze risk metrics for a specific spread
        """
        # We'll add risk analysis next
        pass
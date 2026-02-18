# core/MCForecastTools.py - Simplified version without Alpaca dependency
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class MCSimulation:
    """
    A Python class for running Monte Carlo simulation on portfolio price data.
    
    This version is simplified and does not require Alpaca API.
    """
    
    def __init__(self, portfolio_data, num_simulation=1000, num_trading_days=252):
        """
        Initialize the Monte Carlo simulation.
        
        Parameters:
        portfolio_data (pandas.DataFrame): DataFrame with stock price data
        num_simulation (int): Number of simulation runs
        num_trading_days (int): Number of trading days to simulate
        """
        self.portfolio_data = portfolio_data
        self.num_simulation = num_simulation
        self.num_trading_days = num_trading_days
        self.simulated_return = None
        
        # Calculate daily returns
        self.daily_returns = portfolio_data.pct_change().dropna()
        
    def calc_cumulative_return(self):
        """
        Calculate cumulative returns for all simulations.
        """
        simulation_data = []
        
        for n in range(self.num_simulation):
            # Create a random sequence of daily returns
            daily_return_samples = np.random.choice(
                self.daily_returns.values.flatten(),
                size=self.num_trading_days,
                replace=True
            )
            
            # Calculate cumulative return
            cumulative_return = (1 + daily_return_samples).cumprod()
            simulation_data.append(cumulative_return)
        
        self.simulated_return = pd.DataFrame(simulation_data).T
    
    @property
    def confidence_interval(self):
        """
        Return 95% confidence interval for the final cumulative returns.
        """
        if self.simulated_return is None:
            return None
        
        final_returns = self.simulated_return.iloc[-1, :]
        lower_bound = np.percentile(final_returns, 2.5)
        upper_bound = np.percentile(final_returns, 97.5)
        
        return (lower_bound, upper_bound)
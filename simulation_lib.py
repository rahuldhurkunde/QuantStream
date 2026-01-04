from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from datetime import datetime, timedelta
import dateutil.relativedelta

class Wallet:
    def __init__(self, initial_capital, contribution_amount, contribution_freq):
        """
        contribution_freq: 'Monthly', 'Quarterly', 'Annually', 'None'
        """
        self.initial_capital = initial_capital
        self.contribution_amount = contribution_amount
        self.contribution_freq = contribution_freq

    def simulate_portfolio(self, price_df):
        """
        Simulates the portfolio value over time given the price history.
        Assumes all capital is invested into the stock immediately.
        
        price_df: DataFrame with 'Date' and 'Price'.
        Returns: DataFrame with 'Date', 'Portfolio Value', 'Invested Capital'
        """
        df = price_df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        portfolio_values = []
        invested_capital_values = []
        
        current_shares = 0.0
        total_invested = 0.0
        
        # Determine contribution dates
        start_date = df['Date'].iloc[0]
        next_contribution_date = self._get_next_contribution_date(start_date, start_date)
        
        # Initial Investment
        initial_price = df['Price'].iloc[0]
        if initial_price > 0:
            current_shares += self.initial_capital / initial_price
            total_invested += self.initial_capital
        
        for idx, row in df.iterrows():
            current_date = row['Date']
            price = row['Price']
            
            # Check for contributions
            # We add contribution if current_date >= next_contribution_date
            # But strictly speaking we should only add it once per period. 
            # Simple logic: if today matches or passes the scheduled date, and we haven't contributed for this 'slot' yet.
            # Simplified: If current_date >= next_contribution_date
            
            if self.contribution_freq != 'None' and current_date >= next_contribution_date:
                # Invest contribution
                if price > 0:
                    current_shares += self.contribution_amount / price
                    total_invested += self.contribution_amount
                
                # Advance next contribution date
                next_contribution_date = self._get_next_contribution_date(start_date, next_contribution_date)
                
                # Handle case where gaps in data might make us skip multiple contributions? 
                # For now, let's assume simple one-step advance. 
                # If there's a huge gap, we might miss contributions, but for stock data usually gaps are just weekends/holidays.
                # If the gap is huge, we should probably add all missed contributions.
                # Let's keep it simple: one contribution per trigger.
            
            portfolio_value = current_shares * price
            portfolio_values.append(portfolio_value)
            invested_capital_values.append(total_invested)
            
        result = pd.DataFrame({
            'Date': df['Date'],
            'Portfolio Value': portfolio_values,
            'Invested Capital': invested_capital_values
        })
        return result

    def _get_next_contribution_date(self, start_date, last_date):
        if self.contribution_freq == 'Monthly':
            return last_date + dateutil.relativedelta.relativedelta(months=1)
        elif self.contribution_freq == 'Quarterly':
            return last_date + dateutil.relativedelta.relativedelta(months=3)
        elif self.contribution_freq == 'Annually':
            return last_date + dateutil.relativedelta.relativedelta(years=1)
        else:
            return last_date + timedelta(days=36500) # Far future

class PredictionModel(ABC):
    @abstractmethod
    def train(self, dates, prices):
        """
        dates: array-like of datetime objects or ordinal
        prices: array-like of float
        """
        pass

    @abstractmethod
    def predict(self, future_dates, confidence_interval=0.95):
        """
        future_dates: array-like of datetime objects
        confidence_interval: float between 0 and 1
        Returns: (predictions, lower_bound, upper_bound)
        """
        pass

class LinearRegressionPredictor(PredictionModel):
    def __init__(self):
        self.model = LinearRegression()
        self.std_dev = 0.0

    def train(self, dates, prices):
        # Convert dates to ordinal for regression
        dates_ord = np.array([d.toordinal() for d in dates]).reshape(-1, 1)
        self.model.fit(dates_ord, prices)
        
        # Calculate standard deviation of residuals for uncertainty
        predictions = self.model.predict(dates_ord)
        residuals = prices - predictions
        self.std_dev = np.std(residuals)

    def predict(self, future_dates, confidence_interval=0.95):
        dates_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        predictions = self.model.predict(dates_ord)
        
        # Calculate Z-score based on confidence interval
        # e.g., 95% -> 0.975 quantile -> ~1.96
        z_score = norm.ppf((1 + confidence_interval) / 2)
        
        lower_bound = predictions - (z_score * self.std_dev)
        upper_bound = predictions + (z_score * self.std_dev)
        
        return predictions, lower_bound, upper_bound

class PathShadowPredictor(PredictionModel):
    def __init__(self, lookback_window=60, n_neighbors=20):
        self.lookback_window = lookback_window
        self.n_neighbors = n_neighbors
        self.prices = None
        self.dates = None

    def train(self, dates, prices):
        self.dates = np.array(dates)
        self.prices = np.array(prices)

    def predict(self, future_dates, confidence_interval=0.95):
        if self.prices is None or len(self.prices) < self.lookback_window:
            # Fallback if not enough data: simple linear projection or just last price
            # For simplicity, let's just return the last price flat
            last_price = self.prices[-1] if self.prices is not None else 0
            n = len(future_dates)
            return np.full(n, last_price), np.full(n, last_price), np.full(n, last_price)

        n_future = len(future_dates)
        current_pattern = self.prices[-self.lookback_window:]
        current_pattern_norm = current_pattern / current_pattern[0]

        distances = []
        
        # We search through the history up to the point where we still have 'n_future' days after the window
        # to use for projection.
        # Valid start indices for historical windows:
        # 0 to len(prices) - lookback_window - n_future
        
        max_start_idx = len(self.prices) - self.lookback_window - n_future
        
        if max_start_idx < 0:
            # Not enough history to find a window AND a full future path
            # Relax the constraint: we just need *some* future path, even if shorter?
            # Or just limit the search space.
            # If we can't find full paths, we might have to truncate.
            # Let's fallback to linear regression or similar if truly not enough data.
            # But here, let's just use what we have, maybe overlap?
            # If data is really short, this model isn't suitable.
            # Assuming we have enough data for at least 1 match.
            return np.full(n_future, self.prices[-1]), np.full(n_future, self.prices[-1]), np.full(n_future, self.prices[-1])

        # Optimize: sliding window view could be faster, but simple loop is fine for <5000 points
        for i in range(max_start_idx + 1):
            window = self.prices[i : i + self.lookback_window]
            window_norm = window / window[0]
            dist = np.linalg.norm(current_pattern_norm - window_norm)
            distances.append((dist, i))
        
        # Find top k neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.n_neighbors]
        
        future_paths = []
        last_price = self.prices[-1]
        
        for _, idx in neighbors:
            # Get the historical future prices
            hist_future = self.prices[idx + self.lookback_window : idx + self.lookback_window + n_future]
            
            # Calculate returns relative to the end of that historical window
            # future_factor = price[t+k] / price[t]
            hist_end_price = self.prices[idx + self.lookback_window - 1]
            factors = hist_future / hist_end_price
            
            # Apply to current price
            proj_path = last_price * factors
            future_paths.append(proj_path)
            
        future_paths = np.array(future_paths)
        
        # Aggregate
        mean_pred = np.mean(future_paths, axis=0)
        
        # Quantiles for bounds
        alpha = (1 - confidence_interval) / 2
        lower_bound = np.quantile(future_paths, alpha, axis=0)
        upper_bound = np.quantile(future_paths, 1 - alpha, axis=0)
        
        return mean_pred, lower_bound, upper_bound

# Factory or Manager
class SimulationBackend:
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegressionPredictor,
            "Path Shadow": PathShadowPredictor
        }

    def get_model(self, model_name):
        if model_name in self.models:
            return self.models[model_name]()
        raise ValueError(f"Model {model_name} not found.")


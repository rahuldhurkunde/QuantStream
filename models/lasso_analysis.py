import sys
import os
import argparse
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path to import tickers and indicators
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tickers import POPULAR_TICKERS_MAP
    from indicators import calculate_rsi
except ImportError:
    print("Error: Could not import modules from parent directory.")
    sys.exit(1)

def get_change(current, old):
    if old == 0 or pd.isna(old):
        return None
    return (current - old) / old

def get_perf_from_history(hist, current_price, days):
    if hist.empty:
        return None
    last_date = hist.index[-1]
    target_date = last_date - pd.Timedelta(days=days)
    # Find nearest date
    idx = hist.index.get_indexer([target_date], method='nearest')[0]
    if idx < 0 or idx >= len(hist):
        return None
    old_price = hist['Close'].iloc[idx]
    return get_change(current_price, old_price)

def fetch_data(limit=None):
    """
    Fetches data for the provided tickers.
    Returns a DataFrame with features and target.
    """
    data_list = []
    
    ticker_symbols = list(POPULAR_TICKERS_MAP.values())
    if limit:
        ticker_symbols = ticker_symbols[:limit]

    print(f"Fetching data for {len(ticker_symbols)} tickers...")

    for symbol in tqdm(ticker_symbols):
        try:
            stock = yf.Ticker(symbol)
            # Fetch info
            info = stock.info
            
            # Filter: Price > $10
            price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if price is None or price <= 10:
                continue

            # Fetch History for Technicals
            hist = stock.history(period="max")
            if hist.empty:
                continue
            
            # Ensure index is datetime
            hist.index = pd.to_datetime(hist.index)
            
            # Calculate Technicals
            # RSI
            rsi_val = None
            try:
                # calculate_rsi expects a DataFrame with columns as tickers
                temp_df = pd.DataFrame({symbol: hist['Close']})
                rsi_series = calculate_rsi(temp_df)
                if not rsi_series.empty:
                    rsi_val = rsi_series.iloc[-1].item()
            except:
                pass

            # SMAs (Distance from SMA)
            sma20 = hist['Close'].rolling(20).mean().iloc[-1]
            sma20_dist = get_change(price, sma20) if sma20 else None
            
            # Use info for 50/200 if available, else calc
            sma50 = info.get('fiftyDayAverage')
            sma50_dist = get_change(price, sma50) if sma50 else None
            
            sma200 = info.get('twoHundredDayAverage')
            sma200_dist = get_change(price, sma200) if sma200 else None

            # Performance
            perf_week = get_perf_from_history(hist, price, 7)
            perf_month = get_perf_from_history(hist, price, 30)
            perf_quarter = get_perf_from_history(hist, price, 91)
            perf_half_y = get_perf_from_history(hist, price, 182)
            perf_year = info.get('52WeekChange') # Use info for main year perf
            perf_3y = get_perf_from_history(hist, price, 365*3)
            perf_5y = get_perf_from_history(hist, price, 365*5)
            perf_10y = get_perf_from_history(hist, price, 365*10)
            
            # YTD
            last_date = hist.index[-1]
            year_start = hist[hist.index.year == last_date.year]
            perf_ytd = None
            if not year_start.empty:
                 perf_ytd = get_change(price, year_start.iloc[0]['Close'])

            # Features Map (Comprehensive)
            features = {
                'symbol': symbol,
                'price': price,
                
                # --- Valuation ---
                'trailingPE': info.get('trailingPE'),
                'forwardPE': info.get('forwardPE'),
                'pegRatio': info.get('trailingPegRatio'),
                'priceToSales': info.get('priceToSalesTrailing12Months'),
                'priceToBook': info.get('priceToBook'),
                'enterpriseToEbitda': info.get('enterpriseToEbitda'),
                'enterpriseToRevenue': info.get('enterpriseToRevenue'),
                'priceToCash': (price / info.get('totalCashPerShare')) if info.get('totalCashPerShare') else None,
                
                # --- Profitability ---
                'profitMargins': info.get('profitMargins'),
                'grossMargins': info.get('grossMargins'),
                'operatingMargins': info.get('operatingMargins'),
                'returnOnAssets': info.get('returnOnAssets'),
                'returnOnEquity': info.get('returnOnEquity'),
                
                # --- Financial Health ---
                'totalCashPerShare': info.get('totalCashPerShare'),
                'bookValue': info.get('bookValue'),
                'totalDebt': info.get('totalDebt'), # numeric
                'debtToEquity': info.get('debtToEquity'),
                'currentRatio': info.get('currentRatio'),
                'quickRatio': info.get('quickRatio'),
                
                # --- Growth / Earnings ---
                'revenueGrowth': info.get('revenueGrowth'),
                'earningsGrowth': info.get('earningsGrowth'),
                'trailingEps': info.get('trailingEps'),
                'forwardEps': info.get('forwardEps'),
                'epsCurrentYear': info.get('epsCurrentYear'),
                
                # --- Dividends ---
                'dividendRate': info.get('dividendRate'),
                'dividendYield': info.get('dividendYield'),
                'payoutRatio': info.get('payoutRatio'),
                
                # --- Ownership / Short ---
                'heldPercentInsiders': info.get('heldPercentInsiders'),
                'heldPercentInstitutions': info.get('heldPercentInstitutions'),
                'shortRatio': info.get('shortRatio'),
                'shortPercentOfFloat': info.get('shortPercentOfFloat'),
                'sharesOutstanding': info.get('sharesOutstanding'),
                'floatShares': info.get('floatShares'),
                
                # --- Technicals & Momentum ---
                'beta': info.get('beta'),
                'rsi_14': rsi_val,
                'sma20_dist': sma20_dist,
                'sma50_dist': sma50_dist,
                'sma200_dist': sma200_dist,
                'avgVolume': info.get('averageVolume'),
                'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh'),
                'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow'),
                
                # --- Performance History ---
                'perf_week': perf_week,
                'perf_month': perf_month,
                'perf_quarter': perf_quarter,
                'perf_half_y': perf_half_y,
                'perf_ytd': perf_ytd,
                'perf_3y': perf_3y,
                'perf_5y': perf_5y,
                'perf_10y': perf_10y,

                # --- Others ---
                'fullTimeEmployees': info.get('fullTimeEmployees'),
                'recommendationMean': info.get('recommendationMean'),

                # --- Target ---
                'target_52WeekChange': info.get('52WeekChange')
            }
            
            data_list.append(features)
            
        except Exception as e:
            # Silently fail for individual tickers
            continue

    return pd.DataFrame(data_list)

def run_lasso_analysis(df, alpha=0.1):
    """
    Performs LASSO regression on the dataframe.
    """
    if df.empty:
        print("DataFrame is empty.")
        return

    # Drop symbol for analysis but keep for reference
    X_raw = df.drop(columns=['symbol', 'target_52WeekChange'], errors='ignore')
    y_raw = df['target_52WeekChange']

    print("\n--- Data Quality Summary ---")
    print(f"Total tickers fetched: {len(df)}")
    missing_summary = df.isnull().sum()
    print("Missing values per column:")
    print(missing_summary[missing_summary > 0])

    # 1. Drop columns that are > 50% missing
    threshold = len(df) * 0.5
    cols_to_drop = missing_summary[missing_summary > threshold].index.tolist()
    if 'target_52WeekChange' in cols_to_drop:
        cols_to_drop.remove('target_52WeekChange') # Don't drop target unless we have to
    
    X_reduced = X_raw.drop(columns=cols_to_drop, errors='ignore')
    print(f"\nDropped sparse columns: {cols_to_drop}")

    # 2. Drop rows where target is NaN
    valid_target_mask = y_raw.notnull()
    X_valid = X_reduced[valid_target_mask]
    y_valid = y_raw[valid_target_mask]

    # 3. Fill remaining NaNs with median
    X_filled = X_valid.fillna(X_valid.median(numeric_only=True))

    # 4. Final check
    if len(X_filled) < 5:
        print(f"\nToo few samples ({len(X_filled)}) for analysis. Try a larger limit or fewer features.")
        return

    print(f"Final data shape for analysis: {X_filled.shape}")

    feature_cols = X_filled.columns.tolist()
    # Ensure all are numeric
    X_final = X_filled.select_dtypes(include=[np.number])
    feature_cols = X_final.columns.tolist()

    # Standardize features (Important for LASSO)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_valid, test_size=0.2, random_state=42)
    
    # LASSO Model
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    
    # Evaluate
    train_score = lasso.score(X_train, y_train)
    test_score = lasso.score(X_test, y_test)
    y_pred = lasso.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"\n--- LASSO Results (Alpha={alpha}) ---")
    print(f"Train R^2: {train_score:.4f}")
    print(f"Test R^2:  {test_score:.4f}")
    print(f"MSE:       {mse:.4f}")
    
    # Feature Importance
    coefs = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': lasso.coef_
    })
    
    # Sort by absolute value of coefficient
    coefs['Abs_Coefficient'] = coefs['Coefficient'].abs()
    coefs = coefs.sort_values(by='Abs_Coefficient', ascending=False)
    
    print("\nFeature Importance (Non-zero coefficients):")
    print(coefs[coefs['Abs_Coefficient'] > 0].to_string(index=False))
    
    print("\nZeroed-out Features (Removed by LASSO):")
    print(coefs[coefs['Abs_Coefficient'] == 0]['Feature'].tolist())

def main():
    parser = argparse.ArgumentParser(description="Perform LASSO regression on ticker metrics.")
    parser.add_argument('--limit', type=int, default=100, help="Number of tickers to process (default: 100). Use 0 for all.")
    parser.add_argument('--alpha', type=float, default=0.01, help="Lasso regularization strength (default: 0.01).")
    parser.add_argument('--output', type=str, default='lasso_data.csv', help="Path to save the fetched data.")
    
    args = parser.parse_args()
    
    limit = args.limit if args.limit > 0 else None
    
    # Check if data already exists to save time (optional, but good for CLI)
    if os.path.exists(args.output) and args.limit == 0:
         print(f"Found existing data at {args.output}. Loading...")
         df = pd.read_csv(args.output)
    else:
        df = fetch_data(limit=limit)
        if not df.empty:
            print(f"Saving fetched data to {args.output}")
            df.to_csv(args.output, index=False)
    
    if not df.empty:
        run_lasso_analysis(df, alpha=args.alpha)
    else:
        print("No data found matching criteria.")

if __name__ == "__main__":
    main()

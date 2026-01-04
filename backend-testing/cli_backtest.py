import argparse
import sys
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import json

# Add project root to sys.path to import simulation_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import simulation_lib

def get_price_data_cli(ticker, start_date, end_date):
    """
    Fetch price data for a single ticker using yfinance directly.
    """
    try:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if hist.empty:
            return pd.DataFrame()
        df = hist[['Close']].reset_index()
        df = df.rename(columns={'Close': 'Price'})
        # Ensure Date is datetime and timezone-naive
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}", file=sys.stderr)
        return pd.DataFrame()

def run_backtest(args):
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        prediction_date = datetime.strptime(args.prediction_date, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"Error parsing dates: {e}. Use YYYY-MM-DD format.", file=sys.stderr)
        sys.exit(1)

    if start_date >= end_date:
        print("Error: Start Date must be before End Date.", file=sys.stderr)
        sys.exit(1)
    if end_date >= prediction_date:
        print("Error: End Date must be before Prediction Date.", file=sys.stderr)
        sys.exit(1)

    # Initialize backend
    backend = simulation_lib.SimulationBackend()
    
    results = []

    for ticker in args.tickers:
        print(f"Processing {ticker}...")
        
        # 1. Fetch Data
        # Fetch a bit more data to ensure coverage
        fetch_end_date = max(prediction_date, date.today()) + timedelta(days=5)
        df_all = get_price_data_cli(ticker, start_date, fetch_end_date)
        
        if df_all.empty:
            print(f"  No data found for {ticker}.", file=sys.stderr)
            results.append({
                "ticker": ticker,
                "error": "No data found"
            })
            continue

        # 2. Split Data
        mask_train = df_all['Date'].dt.date <= end_date
        df_train = df_all.loc[mask_train].copy()
        
        mask_test = (df_all['Date'].dt.date > end_date) & (df_all['Date'].dt.date <= prediction_date)
        df_test = df_all.loc[mask_test].copy()
        
        if df_train.empty:
            print(f"  Not enough training data for {ticker} before {end_date}.", file=sys.stderr)
            results.append({
                "ticker": ticker,
                "error": "Not enough training data"
            })
            continue

        # 3. Train Model
        try:
            model = backend.get_model(args.model)
            model.train(df_train['Date'].tolist(), df_train['Price'].values)
        except ValueError as e:
            print(f"  Error initializing model: {e}", file=sys.stderr)
            sys.exit(1)

        # 4. Predict
        # Generate prediction dates
        if not df_test.empty:
            # If we have actual future data (backtesting past), use those dates
            future_dates = df_test['Date'].tolist()
        else:
            # If future, generate business days
            future_dates = pd.date_range(start=end_date + timedelta(days=1), end=prediction_date, freq='B').to_pydatetime().tolist()
        
        if not future_dates:
             # Can happen if range is too small (e.g. holidays)
             print(f"  No valid future dates to predict for {ticker}.", file=sys.stderr)
             continue

        pred_prices, lower, upper = model.predict(future_dates, confidence_interval=args.uncertainty_pct / 100.0)
        
        df_pred = pd.DataFrame({
            'Date': future_dates,
            'Predicted': pred_prices
        })

        # 5. Portfolio Simulation
        portfolio = simulation_lib.Wallet(args.initial_investment, args.contribution_amount, args.contribution_freq)
        
        # A. Actual Scenario
        df_history_combined = pd.concat([df_train, df_test]).sort_values('Date').drop_duplicates('Date')
        portfolio_res_actual = portfolio.simulate_portfolio(df_history_combined)
        
        # B. Predicted Scenario
        df_pred_renamed = df_pred[['Date', 'Predicted']].rename(columns={'Predicted': 'Price'})
        df_scenario_predicted = pd.concat([df_train[['Date', 'Price']], df_pred_renamed]).sort_values('Date').reset_index(drop=True)
        portfolio_res_predicted = portfolio.simulate_portfolio(df_scenario_predicted)

        # 6. Calculate Metrics
        # Find the last date common to both or the end of prediction
        comparison_date = df_history_combined['Date'].max()
        
        def get_metrics_at_date(res_df, target_date):
            filtered = res_df[res_df['Date'] <= target_date]
            if filtered.empty:
                return 0, 0, 0, 0
            row = filtered.iloc[-1]
            val = row['Portfolio Value']
            inv = row['Invested Capital']
            prof = val - inv
            roi = (prof / inv * 100) if inv > 0 else 0
            return val, inv, prof, roi

        val_act, inv_act, prof_act, roi_act = get_metrics_at_date(portfolio_res_actual, comparison_date)
        val_pred, inv_pred, prof_pred, roi_pred = get_metrics_at_date(portfolio_res_predicted, comparison_date)
        
        # Value at End Date (for relative delta)
        val_at_end, _, _, _ = get_metrics_at_date(portfolio_res_actual, pd.to_datetime(end_date))

        ticker_result = {
            "ticker": ticker,
            "comparison_date": str(comparison_date.date()),
            "actual": {
                "portfolio_value": val_act,
                "invested_capital": inv_act,
                "profit_loss": prof_act,
                "roi_pct": roi_act,
                "delta_from_end_date": val_act - val_at_end
            },
            "predicted": {
                "portfolio_value": val_pred,
                "invested_capital": inv_pred,
                "profit_loss": prof_pred,
                "roi_pct": roi_pred,
                "delta_from_end_date": val_pred - val_at_end
            }
        }
        results.append(ticker_result)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Stock Backtesting Tool")
    
    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols (e.g. NVDA AAPL)")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date for simulation cutoff (YYYY-MM-DD)")
    parser.add_argument("--prediction-date", required=True, help="Prediction target date (YYYY-MM-DD)")
    
    parser.add_argument("--initial-investment", type=float, default=10000.0, help="Initial investment amount")
    parser.add_argument("--contribution-amount", type=float, default=0.0, help="Regular contribution amount")
    parser.add_argument("--contribution-freq", choices=["None", "Monthly", "Quarterly", "Annually"], default="None", help="Contribution frequency")
    
    parser.add_argument("--model", choices=["Linear Regression", "Path Shadow"], default="Linear Regression", help="Prediction model")
    parser.add_argument("--uncertainty-pct", type=float, default=95.0, help="Uncertainty interval percentage")
    
    parser.add_argument("--output", help="Path to save output JSON file")

    args = parser.parse_args()
    run_backtest(args)

import argparse
import os
from polygon import RESTClient
from datetime import date

def load_api_key(filepath="api_key.txt"):
    """Loads the API key from a file."""
    try:
        with open(filepath, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file '{filepath}' not found.")
        print("Please create this file and paste your Polygon API key into it.")
        exit(1)

def fetch_data(ticker, start_date, end_date, multiplier=1, timespan="day", limit=50000):
    """Fetches stock data using Polygon API."""
    
    # Try to find the key in the current directory or the script's directory
    key_path = "api_key.txt"
    if not os.path.exists(key_path):
        # Check in the temp directory if we are running from root
        potential_path = os.path.join("temp", "api_key.txt")
        if os.path.exists(potential_path):
            key_path = potential_path
            
    api_key = load_api_key(key_path)
    client = RESTClient(api_key=api_key)

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    try:
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=limit
        ):
            aggs.append(a)
        
        print(f"Successfully fetched {len(aggs)} data points.")
        
        # Simple output to stdout, can be redirected to a file
        print("timestamp,open,high,low,close,volume,vwap,transactions")
        for agg in aggs:
             print(f"{agg.timestamp},{agg.open},{agg.high},{agg.low},{agg.close},{agg.volume},{agg.vwap},{agg.transactions}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock data using Polygon.io API.")
    
    parser.add_argument("--ticker", type=str, help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--multiplier", type=int, default=1, help="Timespan multiplier (default: 1)")
    parser.add_argument("--timespan", type=str, default="day", choices=["minute", "hour", "day", "week", "month", "quarter", "year"], help="Timespan of the data (default: day)")
    parser.add_argument("--limit", type=int, default=50000, help="Limit number of results")

    args = parser.parse_args()

    fetch_data(
        args.ticker,
        args.start_date,
        args.end_date,
        args.multiplier,
        args.timespan,
        args.limit
    )

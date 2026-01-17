import streamlit as st
import pandas as pd
import yfinance as yf
from polygon import RESTClient
from datetime import date, timedelta
import re

def set_page_config(page_title='Stock Prices dashboard', page_icon=':chart_with_upwards_trend:'):
    # Set the title and favicon that appear in the Browser's tab bar.
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
    )

def get_polygon_data(ticker, start_date, end_date, api_key, interval='1d'):
    """Fetch historical data using Polygon.io REST API."""
    if not api_key:
        return pd.DataFrame()
    
    try:
        client = RESTClient(api_key=api_key)
        # Convert intervals to Polygon format
        multiplier = 1
        timespan = 'day'
        if interval == '1wk':
            timespan = 'week'
        elif interval == '1mo':
            timespan = 'month'
        
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date
        )
        
        data = []
        for agg in aggs:
            data.append({
                'Date': pd.to_datetime(agg.timestamp, unit='ms'),
                'Open': agg.open,
                'High': agg.high,
                'Low': agg.low,
                'Price': agg.close, # Polygon uses 'close', we map to 'Price'
                'Ticker': ticker
            })
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching Polygon data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60 * 24)
def get_price_data(tickers, start_date, end_date, interval='1d', source='yfinance', api_key=None):
    """Download historical Close prices for the requested tickers.

    Returns a DataFrame with columns: Date, Ticker, Price
    Cached for 24 hours by default.
    """
    frames = []
    for ticker in tickers:
        try:
            if source == 'polygon':
                df = get_polygon_data(ticker, start_date, end_date, api_key, interval)
                if df.empty:
                    continue
                frames.append(df)
            else:
                # Default to yfinance
                hist = yf.Ticker(ticker).history(start=start_date, end=end_date, interval=interval)
                if hist.empty:
                    continue
                df = hist[['Open', 'High', 'Low', 'Close']].reset_index()
                # Normalize date column name (intraday often returns 'Datetime')
                if 'Datetime' in df.columns:
                    df = df.rename(columns={'Datetime': 'Date'})
                
                df = df.rename(columns={'Close': 'Price'})
                df['Ticker'] = ticker
                frames.append(df)
        except Exception:
            # don't break the whole app if one ticker fails
            continue

    if not frames:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Price'])

    result = pd.concat(frames, ignore_index=True)
    
    # Only convert to date object if we are using daily interval to preserve time for intraday
    if interval == '1d':
        result['Date'] = pd.to_datetime(result['Date']).dt.date
    else:
        result['Date'] = pd.to_datetime(result['Date'])
        
    return result

# --- Ticker Mapping & Helper Functions ---
from tickers import POPULAR_TICKERS_MAP, POPULAR_TICKERS_LIST

def extract_ticker(selection):
    """
    Extracts the ticker symbol from a selection string like "Nvidia (NVDA)".
    If the selection is just a raw ticker or not in the map, returns it as is (uppercased).
    """
    if selection in POPULAR_TICKERS_MAP:
        return POPULAR_TICKERS_MAP[selection]
    
    # Fallback: try to extract content inside parentheses if present
    match = re.search(r'\((.*?)\)', selection)
    if match:
        return match.group(1).upper()
    
    # Default: assume the whole string is the ticker
    return selection.strip().upper()

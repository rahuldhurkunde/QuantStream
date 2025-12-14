import pandas as pd

def add_moving_average(df, ma_period):
    """Adds a moving average column to the DataFrame."""
    for ticker in df.columns:
        if ticker.endswith(('_MA', '_52w_high', '_52w_low')):
            continue
        df[f'{ticker}_MA{ma_period}'] = df[ticker].rolling(window=ma_period, min_periods=1).mean()
    return df

def add_52w_high_low(df):
    """Adds 52-week high and low columns to the DataFrame."""
    win = 252  # 52 weeks ~ 252 trading days
    for ticker in df.columns:
        if ticker.endswith(('_MA', '_52w_high', '_52w_low')):
            continue
        df[f'{ticker}_52w_high'] = df[ticker].rolling(window=win, min_periods=1).max()
        df[f'{ticker}_52w_low'] = df[ticker].rolling(window=win, min_periods=1).min()
    return df

def calculate_rsi(df):
    """Calculates the RSI for each ticker in the DataFrame."""
    rsi_period = 14
    rsi_frames = {}
    for ticker in df.columns:
        if ticker.endswith(('_MA', '_52w_high', '_52w_low')):
            continue
        series = df[ticker].dropna()
        if series.empty:
            continue

        delta = series.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        # Wilder's smoothing (EMA with alpha = 1/n)
        ma_up = up.ewm(alpha=1/rsi_period, adjust=False).mean()
        ma_down = down.ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        rsi.index = series.index
        rsi_frames[ticker] = rsi

    if rsi_frames:
        return pd.DataFrame(rsi_frames)
    return pd.DataFrame()

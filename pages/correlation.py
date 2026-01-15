import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from utils import get_price_data, extract_ticker
from tickers import POPULAR_TICKERS_LIST

st.set_page_config(page_title="Ticker Correlation", page_icon="📈")

st.markdown("# 🔗 Ticker Correlation Analysis")
st.markdown("Select two tickers to analyze their correlation via convolution over different time periods.")

# Ticker Selection
col1, col2 = st.columns(2)
with col1:
    ticker1_select = st.selectbox(
        "Select Ticker 1", 
        POPULAR_TICKERS_LIST, 
        index=POPULAR_TICKERS_LIST.index('Nvidia Corp (NVDA)') if 'Nvidia Corp (NVDA)' in POPULAR_TICKERS_LIST else 0
    )
with col2:
    ticker2_select = st.selectbox(
        "Select Ticker 2", 
        POPULAR_TICKERS_LIST, 
        index=POPULAR_TICKERS_LIST.index('Amazon Com Inc (AMZN)') if 'Amazon Com Inc (AMZN)' in POPULAR_TICKERS_LIST else 1
    )

# Time Range Selection
st.markdown("### Select Time Range")
time_range = st.radio(
    "Range",
    options=["1 Week", "15 Days", "1 Month", "3 Months", "1 Year"],
    horizontal=True,
    label_visibility="collapsed"
)

# Calculate Dates and Interval
end_date = date.today()
interval = "1d" # Default

if time_range == "1 Week":
    start_date = end_date - timedelta(weeks=1)
    interval = "5m" # High resolution for short period
elif time_range == "15 Days":
    start_date = end_date - timedelta(days=15)
    interval = "30m" # Medium resolution
elif time_range == "1 Month":
    start_date = end_date - timedelta(days=30)
    interval = "60m" # Hourly resolution
elif time_range == "3 Months":
    start_date = end_date - timedelta(days=90)
    interval = "1d"
else: # 1 Year
    start_date = end_date - timedelta(days=365)
    interval = "1d"

# Extract symbols
ticker1 = extract_ticker(ticker1_select)
ticker2 = extract_ticker(ticker2_select)

if ticker1 and ticker2:
    # Fetch Data
    # Fetch with a slight buffer to ensure we get enough data points if weekends are involved
    # but utils.get_price_data handles dates pretty well.
    # Pass interval to get_price_data
    df = get_price_data([ticker1, ticker2], start_date.isoformat(), (end_date + timedelta(days=1)).isoformat(), interval=interval)

    if df.empty:
        st.warning("No data found for the selected range.")
    else:
        # Pivot to get aligned dates
        # Note: 'Date' column might be datetime now if interval is intraday
        pivot_df = df.pivot(index='Date', columns='Ticker', values='Price')
        pivot_df = pivot_df.dropna() # Ensure we have data for both on all days

        if pivot_df.empty or ticker1 not in pivot_df.columns or ticker2 not in pivot_df.columns:
            st.warning("Insufficient overlapping data for both tickers.")
        else:
            # Get series
            series1 = pivot_df[ticker1].values
            series2 = pivot_df[ticker2].values
            
            # Normalize (Area under curve = 1)
            # Assuming unit spacing for time
            norm1 = series1 / np.sum(series1)
            norm2 = series2 / np.sum(series2)

            # Convolution
            convolution = np.convolve(norm1, norm2, mode='full')

            # Create Plots
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=False, 
                vertical_spacing=0.15,
                row_heights=[0.7, 0.3],
                subplot_titles=("Normalized Price History", "Convolution")
            )

            # Top Plot: Normalized Prices
            fig.add_trace(go.Scatter(x=pivot_df.index, y=norm1, name=f"{ticker1} (Norm)", mode='lines'), row=1, col=1)
            fig.add_trace(go.Scatter(x=pivot_df.index, y=norm2, name=f"{ticker2} (Norm)", mode='lines'), row=1, col=1)

            # Bottom Plot: Convolution
            # Create an x-axis for convolution (lags)
            # Convolution size is len1 + len2 - 1
            conv_x = np.arange(len(convolution))
            fig.add_trace(go.Scatter(x=conv_x, y=convolution, name="Convolution", mode='lines', line=dict(color='purple')), row=2, col=1)

            fig.update_layout(height=700, showlegend=True)
            
            # Update axes titles
            fig.update_xaxes(title_text="Date/Time", row=1, col=1)
            fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
            
            fig.update_xaxes(title_text="Lag / Displacement", row=2, col=1)
            fig.update_yaxes(title_text="Magnitude", row=2, col=1)

            # Use the 'width' parameter as requested by the user's warning message
            # If this fails on older versions, the user should update Streamlit or revert to use_container_width=True
            st.plotly_chart(fig, width="stretch")

            # Display stats
            st.markdown("### Statistics")
            col_a, col_b = st.columns(2)
            col_a.metric(f"{ticker1} Data Points", len(series1))
            col_b.metric(f"{ticker2} Data Points", len(series2))

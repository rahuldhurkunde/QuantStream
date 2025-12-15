import yfinance as yf
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data
def get_news(ticker_symbol):
    """
    Fetches news for a given stock ticker.
    """
    logging.info(f"Fetching news for {ticker_symbol}")
    ticker = yf.Ticker(ticker_symbol)
    company_news = ticker.news
    
    if not company_news:
        logging.info(f"No news found for {ticker_symbol}")
        return []
    
    news_list = []
    for article in company_news:
        content = article.get('content', {})
        headline = content.get('title', 'N/A - No Title Found')
        canonical_url = content.get('canonicalUrl', {})
        link = canonical_url.get('url', 'N/A - No Link Found')
        publisher = article.get('publisher', 'N/A - No Publisher')
        
        logging.info(f"Fetched headline for {ticker_symbol}: {headline}")

        news_list.append({
            'headline': headline,
            'link': link,
            'publisher': publisher
        })
        
    return news_list

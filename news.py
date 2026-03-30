import yfinance as yf
import streamlit as st
import logging
from polygon import RESTClient
from datetime import datetime

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data(ttl=3600)
def get_news(ticker_symbol, source='yfinance', api_key=None):
    """
    Fetches news for a given stock ticker.
    """
    logging.info(f"Fetching news for {ticker_symbol} using {source}")
    
    news_list = []
    
    if source == 'polygon' and api_key:
        try:
            client = RESTClient(api_key=api_key)
            # Fetch most recent news
            articles = client.list_ticker_news(ticker_symbol, limit=10)
            
            for article in articles:
                news_list.append({
                    'headline': article.title,
                    'link': article.article_url,
                    'publisher': article.author if hasattr(article, 'author') else 'Polygon.io',
                    'published_utc': article.published_utc
                })
            
            if not news_list:
                logging.info(f"No news found for {ticker_symbol} on Polygon")
            
            return news_list

        except Exception as e:
            logging.error(f"Error fetching news from Polygon: {e}")
            return []

    # Default to yfinance
    try:
        ticker = yf.Ticker(ticker_symbol)
        company_news = ticker.news
        
        if not company_news:
            logging.info(f"No news found for {ticker_symbol} on yfinance")
            return []
        
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
    except Exception as e:
        logging.error(f"Error fetching news from yfinance: {e}")
        return []

import pytest
from unittest.mock import MagicMock, patch
from news import get_news

@patch('news.RESTClient')
def test_get_news_polygon(mock_rest_client):
    # Setup mock return data for Polygon News
    mock_article1 = MagicMock()
    mock_article1.title = "Polygon News 1"
    mock_article1.article_url = "http://polygon.io/news1"
    mock_article1.author = "Polygon Author"
    mock_article1.published_utc = "2023-01-01T10:00:00Z"

    mock_instance = MagicMock()
    mock_instance.list_ticker_news.return_value = [mock_article1]
    mock_rest_client.return_value = mock_instance
    
    ticker = "TEST_POLY"
    api_key = "fake_key"
    
    news_list = get_news(ticker, source='polygon', api_key=api_key)
    
    # Verification
    mock_rest_client.assert_called_with(api_key=api_key)
    mock_instance.list_ticker_news.assert_called_with(ticker, limit=10)
    
    assert len(news_list) == 1
    assert news_list[0]['headline'] == "Polygon News 1"
    assert news_list[0]['link'] == "http://polygon.io/news1"
    assert news_list[0]['publisher'] == "Polygon Author"
    assert news_list[0]['published_utc'] == "2023-01-01T10:00:00Z"

@patch('news.RESTClient')
def test_get_news_polygon_error(mock_rest_client):
    mock_instance = MagicMock()
    mock_instance.list_ticker_news.side_effect = Exception("API Error")
    mock_rest_client.return_value = mock_instance
    
    news_list = get_news("TEST_ERR", source='polygon', api_key="key")
    
    assert news_list == []

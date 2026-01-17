import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from utils import get_price_data

@patch('utils.RESTClient')
def test_get_price_data_polygon(mock_rest_client):
    # Setup mock return data for Polygon
    # Polygon returns a list of Agg objects
    mock_agg1 = MagicMock()
    mock_agg1.timestamp = 1672531200000 # 2023-01-01
    mock_agg1.open = 100
    mock_agg1.high = 105
    mock_agg1.low = 99
    mock_agg1.close = 101

    mock_agg2 = MagicMock()
    mock_agg2.timestamp = 1672617600000 # 2023-01-02
    mock_agg2.open = 102
    mock_agg2.high = 106
    mock_agg2.low = 101
    mock_agg2.close = 105

    mock_instance = MagicMock()
    mock_instance.get_aggs.return_value = [mock_agg1, mock_agg2]
    mock_rest_client.return_value = mock_instance
    
    start = '2023-01-01'
    end = '2023-01-03'
    tickers = ['TEST_POLY']
    api_key = 'fake_key'
    
    df = get_price_data(tickers, start, end, source='polygon', api_key=api_key)
    
    # Verification
    mock_rest_client.assert_called_with(api_key=api_key)
    mock_instance.get_aggs.assert_called()
    
    assert not df.empty
    assert 'Ticker' in df.columns
    assert 'Price' in df.columns
    assert df['Ticker'].iloc[0] == 'TEST_POLY'
    assert df['Price'].iloc[0] == 101
    assert df['Price'].iloc[1] == 105
    
    # Check date conversion
    # 1672531200000 is 2023-01-01 00:00:00 UTC
    # Since we use '1d' interval, it should be a date object
    assert df['Date'].iloc[0] == pd.to_datetime('2023-01-01').date()

@patch('utils.RESTClient')
def test_get_price_data_polygon_no_key(mock_rest_client):
    start = '2023-01-01'
    end = '2023-01-03'
    tickers = ['TEST_POLY']
    
    # Should handle missing key gracefully (return empty DF or handle internally)
    # The current implementation returns empty DF if no key is provided in get_polygon_data
    # But get_price_data calls it.
    
    df = get_price_data(tickers, start, end, source='polygon', api_key=None)
    
    assert df.empty

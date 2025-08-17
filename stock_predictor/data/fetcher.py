"""Data fetcher for retrieving stock data from Yahoo Finance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time
import random
from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import DataFetchError


class DataFetcher:
    """Fetches historical stock data from Yahoo Finance API."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.logger = get_logger('data.fetcher')
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLC stock data for a given symbol and date range.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLC data
            
        Raises:
            DataFetchError: If data fetching fails after retries
        """
        self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        
        for attempt in range(self.max_retries):
            try:
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                data = ticker.history(start=start_date, end=end_date)
                
                if data.empty:
                    raise DataFetchError(f"No data found for symbol {symbol}")
                
                # Reset index to make Date a column
                data.reset_index(inplace=True)
                
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                # Add symbol column
                data['symbol'] = symbol
                
                # Validate data completeness
                if not self.validate_data_completeness(data):
                    self.logger.warning(f"Data completeness validation failed for {symbol}")
                
                self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    raise DataFetchError(f"Failed to fetch data for {symbol} after {self.max_retries} attempts: {str(e)}")
    
    def fetch_stock_data_years_back(self, symbol: str, years: int, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data for a specified number of years back from end date.
        
        Args:
            symbol: Stock symbol
            years: Number of years to look back
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        start_dt = end_dt - timedelta(days=years * 365)
        start_date = start_dt.strftime('%Y-%m-%d')
        
        return self.fetch_stock_data(symbol, start_date, end_date)
    
    def validate_data_completeness(self, data: pd.DataFrame) -> bool:
        """
        Validate that the fetched data is complete and consistent.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            True if data passes validation, False otherwise
        """
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        
        # Check required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for excessive missing values
        missing_percentage = data[required_columns].isnull().sum() / len(data)
        if (missing_percentage > 0.1).any():  # More than 10% missing
            self.logger.warning(f"High percentage of missing values: {missing_percentage}")
            return False
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} records with invalid OHLC relationships")
            return False
        
        # Check for negative values
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        negative_values = (data[numeric_columns] < 0).any()
        if negative_values.any():
            self.logger.warning("Found negative values in price/volume data")
            return False
        
        return True
    
    def get_available_symbols(self) -> list:
        """
        Get a list of commonly traded stock symbols for testing.
        
        Returns:
            List of stock symbols
        """
        return [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
            'META', 'NVDA', 'JPM', 'JNJ', 'V',
            'PG', 'UNH', 'HD', 'MA', 'DIS'
        ]
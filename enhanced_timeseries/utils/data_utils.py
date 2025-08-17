"""
Data utility functions for the enhanced time series system.
"""

import numpy as np
import pandas as pd
import torch
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Advanced data processing utilities."""
    
    def __init__(self, scaler_type: str = 'standard'):
        self.scaler_type = scaler_type
        self.scalers = {}
        self.is_fitted = False
        
    def get_scaler(self, scaler_type: str = None):
        """Get appropriate scaler instance."""
        scaler_type = scaler_type or self.scaler_type
        
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def fit_scalers(self, data: Dict[str, np.ndarray]) -> 'DataProcessor':
        """Fit scalers on training data."""
        for key, values in data.items():
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            
            scaler = self.get_scaler()
            scaler.fit(values)
            self.scalers[key] = scaler
            
        self.is_fitted = True
        return self
    
    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before transformation")
            
        transformed = {}
        for key, values in data.items():
            if key in self.scalers:
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                    transformed[key] = self.scalers[key].transform(values).flatten()
                else:
                    transformed[key] = self.scalers[key].transform(values)
            else:
                transformed[key] = values
                
        return transformed
    
    def inverse_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Inverse transform data using fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Scalers must be fitted before inverse transformation")
            
        inverse_transformed = {}
        for key, values in data.items():
            if key in self.scalers:
                if values.ndim == 1:
                    values = values.reshape(-1, 1)
                    inverse_transformed[key] = self.scalers[key].inverse_transform(values).flatten()
                else:
                    inverse_transformed[key] = self.scalers[key].inverse_transform(values)
            else:
                inverse_transformed[key] = values
                
        return inverse_transformed


class SequenceGenerator:
    """Generate sequences for time series models."""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from features and targets."""
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")
            
        X, y = [], []
        
        for i in range(self.sequence_length, len(features) - self.prediction_horizon + 1):
            # Input sequence
            X.append(features[i-self.sequence_length:i])
            
            # Target (can be single value or sequence)
            if self.prediction_horizon == 1:
                y.append(targets[i + self.prediction_horizon - 1])
            else:
                y.append(targets[i:i + self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def create_prediction_sequence(self, features: np.ndarray) -> np.ndarray:
        """Create sequence for prediction (last sequence_length points)."""
        if len(features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
            
        return features[-self.sequence_length:].reshape(1, self.sequence_length, -1)


class DataDownloader:
    """Download and manage financial data."""
    
    @staticmethod
    def download_stock_data(symbol: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """Download stock data from Yahoo Finance."""
        try:
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to download data for {symbol}: {str(e)}")
    
    @staticmethod
    def download_multiple_stocks(symbols: List[str], period: str = '2y', 
                                interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Download data for multiple stocks."""
        data_dict = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                data_dict[symbol] = DataDownloader.download_stock_data(symbol, period, interval)
            except Exception as e:
                failed_symbols.append(symbol)
                print(f"Failed to download {symbol}: {str(e)}")
        
        if failed_symbols:
            print(f"Failed to download data for: {failed_symbols}")
            
        return data_dict


class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> List[str]:
        """Validate OHLCV data and return list of issues."""
        issues = []
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")
        
        if not missing_columns:  # Only check if we have all columns
            # Check for missing values
            if data[required_columns].isnull().any().any():
                issues.append("Missing values detected in OHLCV data")
            
            # Check price relationships
            if (data['High'] < data['Low']).any():
                issues.append("High prices lower than low prices detected")
            
            if (data['High'] < data['Open']).any() or (data['High'] < data['Close']).any():
                issues.append("High prices lower than open/close prices detected")
                
            if (data['Low'] > data['Open']).any() or (data['Low'] > data['Close']).any():
                issues.append("Low prices higher than open/close prices detected")
            
            # Check for negative values
            if (data[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
                issues.append("Non-positive price values detected")
                
            if (data['Volume'] < 0).any():
                issues.append("Negative volume values detected")
            
            # Check for extreme price movements (>50% in one day)
            returns = data['Close'].pct_change().abs()
            if (returns > 0.5).any():
                issues.append("Extreme price movements (>50%) detected")
        
        return issues
    
    @staticmethod
    def clean_data(data: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        cleaned_data = data.copy()
        
        # Handle missing values
        if method == 'forward_fill':
            cleaned_data = cleaned_data.fillna(method='ffill')
        elif method == 'backward_fill':
            cleaned_data = cleaned_data.fillna(method='bfill')
        elif method == 'interpolate':
            cleaned_data = cleaned_data.interpolate()
        elif method == 'drop':
            cleaned_data = cleaned_data.dropna()
        
        # Remove any remaining NaN values
        cleaned_data = cleaned_data.dropna()
        
        return cleaned_data


class TensorUtils:
    """Utilities for PyTorch tensor operations."""
    
    @staticmethod
    def to_tensor(data: Union[np.ndarray, List], device: torch.device = None, 
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Convert data to PyTorch tensor."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype).to(device)
        else:
            tensor = torch.tensor(data, dtype=dtype, device=device)
            
        return tensor
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array."""
        return tensor.detach().cpu().numpy()
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')


def split_time_series(data: pd.DataFrame, train_ratio: float = 0.8, 
                     validation_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time series data into train, validation, and test sets."""
    n = len(data)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data


def calculate_returns(prices: Union[pd.Series, np.ndarray], method: str = 'simple') -> np.ndarray:
    """Calculate returns from price series."""
    if isinstance(prices, pd.Series):
        prices = prices.values
        
    if method == 'simple':
        returns = np.diff(prices) / prices[:-1]
    elif method == 'log':
        returns = np.diff(np.log(prices))
    else:
        raise ValueError(f"Unknown return calculation method: {method}")
    
    return returns


def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
    """Detect outliers in data."""
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outliers = z_scores > threshold
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    return outliers
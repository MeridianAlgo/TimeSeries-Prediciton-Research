"""
Data Processor for Time Series Prediction
========================================

Handles data loading, preprocessing, and feature engineering for time series data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Comprehensive data processor for time series prediction.
    
    Handles data fetching, cleaning, feature engineering, and preprocessing.
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 feature_columns: List[str] = None,
                 target_column: str = 'close',
                 sequence_length: int = 60):
        """
        Initialize data processor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            feature_columns: List of feature column names
            target_column: Name of target column
            sequence_length: Length of sequences for time series models
        """
        self.scaler_type = scaler_type
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.scaler = None
        self.target_scaler = None
        
    def fetch_data(self, symbol: str, period: str = '2y', interval: str = '1d') -> pd.DataFrame:
        """
        Fetch financial data using yfinance.
        
        Args:
            symbol: Stock symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            print(f"✅ Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"❌ Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            data: Raw financial data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing values
        data = data.dropna()
        
        # Remove duplicate dates
        data = data.drop_duplicates(subset=['date'])
        
        # Sort by date
        data = data.sort_values('date').reset_index(drop=True)
        
        # Handle outliers using IQR method
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"✅ Cleaned data: {len(data)} records")
        return data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features.
        
        Args:
            data: Clean financial data
            
        Returns:
            DataFrame with additional features
        """
        features = data.copy()
        
        # Price-based features
        features['returns'] = features['close'].pct_change()
        features['log_returns'] = np.log(features['close'] / features['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = features['close'].rolling(window).mean()
            features[f'ema_{window}'] = features['close'].ewm(span=window).mean()
            features[f'price_sma_ratio_{window}'] = features['close'] / features[f'sma_{window}']
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
        
        # RSI
        for period in [7, 14, 21]:
            delta = features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = features['close'].ewm(span=12).mean()
        exp2 = features['close'].ewm(span=26).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for window in [20]:
            sma = features['close'].rolling(window).mean()
            std = features['close'].rolling(window).std()
            features[f'bb_upper_{window}'] = sma + (std * 2)
            features[f'bb_lower_{window}'] = sma - (std * 2)
            features[f'bb_width_{window}'] = features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']
            features[f'bb_position_{window}'] = (features['close'] - features[f'bb_lower_{window}']) / features[f'bb_width_{window}']
        
        # Volume features
        features['volume_sma_20'] = features['volume'].rolling(20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_sma_20']
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            features[f'close_lag_{lag}'] = features['close'].shift(lag)
            features[f'volume_lag_{lag}'] = features['volume'].shift(lag)
        
        # Remove NaN values
        features = features.dropna()
        
        print(f"✅ Created {len(features.columns)} features")
        return features
    
    def prepare_sequences(self, data: pd.DataFrame, target_col: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series models.
        
        Args:
            data: Feature DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays
        """
        if target_col is None:
            target_col = self.target_column
        
        # Select feature columns
        if self.feature_columns is None:
            # Use all numeric columns except target
            feature_cols = [col for col in data.columns if col != target_col and data[col].dtype in ['float64', 'int64']]
        else:
            feature_cols = [col for col in self.feature_columns if col in data.columns]
        
        # Prepare data
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X)):
            X_sequences.append(X[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"✅ Created {len(X_sequences)} sequences")
        print(f"📊 X shape: {X_sequences.shape}")
        print(f"📊 y shape: {y_sequences.shape}")
        
        return X_sequences, y_sequences
    
    def scale_data(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale the data using the specified scaler.
        
        Args:
            X: Feature array
            y: Target array (optional)
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of scaled (X, y) arrays
        """
        # Initialize scalers
        if self.scaler is None:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
        
        if y is not None and self.target_scaler is None:
            if self.scaler_type == 'standard':
                self.target_scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.target_scaler = MinMaxScaler()
            else:
                self.target_scaler = StandardScaler()
        
        # Scale features
        if fit:
            # Reshape for 3D data (samples, timesteps, features)
            if len(X.shape) == 3:
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.fit_transform(X_reshaped)
                X_scaled = X_scaled.reshape(X.shape)
            else:
                X_scaled = self.scaler.fit_transform(X)
        else:
            if len(X.shape) == 3:
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = self.scaler.transform(X_reshaped)
                X_scaled = X_scaled.reshape(X.shape)
            else:
                X_scaled = self.scaler.transform(X)
        
        # Scale target
        y_scaled = None
        if y is not None:
            if fit:
                y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def inverse_scale_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled target values.
        
        Args:
            y_scaled: Scaled target values
            
        Returns:
            Original scale target values
        """
        if self.target_scaler is None:
            return y_scaled
        
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature array
            y: Target array
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_columns or []
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers."""
        import joblib
        scalers = {
            'feature_scaler': self.scaler,
            'target_scaler': self.target_scaler
        }
        joblib.dump(scalers, filepath)
        print(f"✅ Scalers saved to {filepath}")
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers."""
        import joblib
        scalers = joblib.load(filepath)
        self.scaler = scalers['feature_scaler']
        self.target_scaler = scalers['target_scaler']
        print(f"✅ Scalers loaded from {filepath}")

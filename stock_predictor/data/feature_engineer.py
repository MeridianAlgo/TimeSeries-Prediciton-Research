"""Feature engineering for stock price data."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import DataPreprocessingError


class FeatureEngineer:
    """Creates technical indicators and engineered features for stock data."""
    
    def __init__(self):
        self.logger = get_logger('data.feature_engineer')
        self.feature_columns = []
    
    def create_moving_averages(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Create simple and exponential moving averages.
        
        Args:
            data: DataFrame with stock data
            windows: List of window sizes for moving averages
            
        Returns:
            DataFrame with moving average features
        """
        self.logger.info(f"Creating moving averages with windows: {windows}")
        data_with_ma = data.copy()
        
        if 'close' not in data.columns:
            raise DataPreprocessingError("'close' column required for moving averages")
        
        for window in windows:
            # Simple Moving Average
            sma_col = f'sma_{window}'
            data_with_ma[sma_col] = data_with_ma['close'].rolling(window=window).mean()
            self.feature_columns.append(sma_col)
            
            # Exponential Moving Average
            ema_col = f'ema_{window}'
            data_with_ma[ema_col] = data_with_ma['close'].ewm(span=window).mean()
            self.feature_columns.append(ema_col)
        
        return data_with_ma
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate price volatility using rolling standard deviation.
        
        Args:
            data: DataFrame with stock data
            window: Window size for volatility calculation
            
        Returns:
            DataFrame with volatility features
        """
        self.logger.info(f"Calculating volatility with window: {window}")
        data_with_vol = data.copy()
        
        if 'close' not in data.columns:
            raise DataPreprocessingError("'close' column required for volatility")
        
        # Calculate returns
        data_with_vol['returns'] = data_with_vol['close'].pct_change()
        self.feature_columns.append('returns')
        
        # Rolling volatility (standard deviation of returns)
        vol_col = f'volatility_{window}'
        data_with_vol[vol_col] = data_with_vol['returns'].rolling(window=window).std()
        self.feature_columns.append(vol_col)
        
        # Realized volatility (annualized)
        realized_vol_col = f'realized_vol_{window}'
        data_with_vol[realized_vol_col] = data_with_vol[vol_col] * np.sqrt(252)  # 252 trading days
        self.feature_columns.append(realized_vol_col)
        
        return data_with_vol
    
    def create_lagged_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
        """
        Create lagged price features.
        
        Args:
            data: DataFrame with stock data
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        self.logger.info(f"Creating lagged features with lags: {lags}")
        data_with_lags = data.copy()
        
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in data.columns:
                for lag in lags:
                    lag_col = f'{col}_lag_{lag}'
                    data_with_lags[lag_col] = data_with_lags[col].shift(lag)
                    self.feature_columns.append(lag_col)
        
        return data_with_lags
    
    def create_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive technical indicators.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            DataFrame with technical indicators
        """
        self.logger.info("Creating technical indicators")
        data_with_indicators = data.copy()
        
        # RSI (Relative Strength Index)
        data_with_indicators = self._calculate_rsi(data_with_indicators)
        
        # MACD (Moving Average Convergence Divergence)
        data_with_indicators = self._calculate_macd(data_with_indicators)
        
        # Bollinger Bands
        data_with_indicators = self._calculate_bollinger_bands(data_with_indicators)
        
        # Price position within daily range
        data_with_indicators = self._calculate_price_position(data_with_indicators)
        
        # Volume indicators
        data_with_indicators = self._calculate_volume_indicators(data_with_indicators)
        
        return data_with_indicators
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        if 'close' not in data.columns:
            return data
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        data[f'rsi_{period}'] = rsi
        self.feature_columns.append(f'rsi_{period}')
        
        return data
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator."""
        if 'close' not in data.columns:
            return data
        
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        data['macd_histogram'] = macd_histogram
        
        self.feature_columns.extend(['macd', 'macd_signal', 'macd_histogram'])
        
        return data
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        if 'close' not in data.columns:
            return data
        
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        
        data['bb_upper'] = rolling_mean + (rolling_std * num_std)
        data['bb_lower'] = rolling_mean - (rolling_std * num_std)
        data['bb_middle'] = rolling_mean
        
        # Bollinger Band position (0 = lower band, 1 = upper band)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Bollinger Band width (volatility measure)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        self.feature_columns.extend(['bb_upper', 'bb_lower', 'bb_middle', 'bb_position', 'bb_width'])
        
        return data
    
    def _calculate_price_position(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate price position within daily range."""
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return data
        
        # Price position within daily range (0 = low, 1 = high)
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Daily range as percentage of close
        data['daily_range'] = (data['high'] - data['low']) / data['close']
        
        self.feature_columns.extend(['price_position', 'daily_range'])
        
        return data
    
    def _calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        if 'volume' not in data.columns:
            return data
        
        # Volume moving averages
        for window in [10, 20]:
            vol_ma_col = f'volume_ma_{window}'
            data[vol_ma_col] = data['volume'].rolling(window=window).mean()
            self.feature_columns.append(vol_ma_col)
        
        # Volume ratio (current volume / average volume)
        data['volume_ratio'] = data['volume'] / data['volume_ma_20']
        self.feature_columns.append('volume_ratio')
        
        # Price-Volume Trend
        if 'close' in data.columns:
            data['pvt'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1) * data['volume']).cumsum()
            self.feature_columns.append('pvt')
        
        return data
    
    def create_target_variable(self, data: pd.DataFrame, target_type: str = 'next_close', 
                              horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable for prediction.
        
        Args:
            data: DataFrame with stock data
            target_type: Type of target ('next_close', 'return', 'direction')
            horizon: Number of periods ahead to predict
            
        Returns:
            DataFrame with target variable
        """
        self.logger.info(f"Creating target variable: {target_type}, horizon: {horizon}")
        data_with_target = data.copy()
        
        if target_type == 'next_close':
            data_with_target['target'] = data_with_target['close'].shift(-horizon)
        
        elif target_type == 'return':
            data_with_target['target'] = (
                data_with_target['close'].shift(-horizon) / data_with_target['close'] - 1
            )
        
        elif target_type == 'direction':
            future_price = data_with_target['close'].shift(-horizon)
            data_with_target['target'] = (future_price > data_with_target['close']).astype(int)
        
        else:
            raise DataPreprocessingError(f"Unknown target type: {target_type}")
        
        return data_with_target
    
    def get_feature_importance_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get data prepared for feature importance analysis.
        
        Args:
            data: DataFrame with engineered features
            
        Returns:
            Dictionary with feature data and metadata
        """
        # Check if features exist in the data
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        if not available_features:
            # If no features available, return empty result
            return {
                'features': pd.DataFrame(),
                'feature_names': [],
                'n_features': 0,
                'n_samples': 0
            }
        
        # Separate features from other columns
        feature_data = data[available_features].copy()
        
        # Remove rows with NaN values (common after feature engineering)
        feature_data = feature_data.dropna()
        
        return {
            'features': feature_data,
            'feature_names': available_features,
            'n_features': len(available_features),
            'n_samples': len(feature_data)
        }
    
    def engineer_all_features(self, data: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            data: DataFrame with stock data
            config: Configuration dictionary with feature parameters
            
        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("Engineering all features")
        
        if config is None:
            config = {
                'moving_averages': [5, 10, 20, 50],
                'volatility_window': 20,
                'lag_periods': [1, 2, 3, 5]
            }
        
        # Reset feature columns list
        self.feature_columns = []
        
        # Apply all feature engineering steps
        data_engineered = data.copy()
        data_engineered = self.create_moving_averages(data_engineered, config.get('moving_averages', [5, 10, 20, 50]))
        data_engineered = self.calculate_volatility(data_engineered, config.get('volatility_window', 20))
        data_engineered = self.create_lagged_features(data_engineered, config.get('lag_periods', [1, 2, 3, 5]))
        data_engineered = self.create_technical_indicators(data_engineered)
        
        self.logger.info(f"Created {len(self.feature_columns)} features")
        
        return data_engineered
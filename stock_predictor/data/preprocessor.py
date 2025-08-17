"""Data preprocessing for stock price data."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import DataPreprocessingError


class DataPreprocessor:
    """Preprocesses stock data for machine learning models."""
    
    def __init__(self):
        self.logger = get_logger('data.preprocessor')
        self.scalers = {}
        self.is_fitted = False
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in stock data.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            DataFrame with missing values handled
        """
        self.logger.info("Handling missing values")
        data_clean = data.copy()
        
        # Forward fill for OHLC values (common for stock data)
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        for col in price_columns:
            if col in data_clean.columns:
                data_clean[col] = data_clean[col].ffill()
        
        # Handle volume separately (can use median)
        if 'volume' in data_clean.columns:
            median_volume = data_clean['volume'].median()
            data_clean['volume'] = data_clean['volume'].fillna(median_volume)
        
        # Drop rows with remaining NaN values
        initial_rows = len(data_clean)
        data_clean = data_clean.dropna()
        final_rows = len(data_clean)
        
        if initial_rows != final_rows:
            self.logger.info(f"Dropped {initial_rows - final_rows} rows with missing values")
        
        return data_clean
    
    def normalize_prices(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize price data using MinMax scaling.
        
        Args:
            data: DataFrame with stock data
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            DataFrame with normalized prices
        """
        self.logger.info("Normalizing price data")
        data_normalized = data.copy()
        
        price_columns = ['open', 'high', 'low', 'close', 'adj_close']
        
        for col in price_columns:
            if col in data_normalized.columns:
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = MinMaxScaler()
                    data_normalized[col] = self.scalers[col].fit_transform(
                        data_normalized[[col]]
                    ).flatten()
                else:
                    if col not in self.scalers:
                        raise DataPreprocessingError(f"Scaler for {col} not fitted")
                    data_normalized[col] = self.scalers[col].transform(
                        data_normalized[[col]]
                    ).flatten()
        
        # Normalize volume separately (different scale)
        if 'volume' in data_normalized.columns:
            if fit:
                if 'volume' not in self.scalers:
                    self.scalers['volume'] = MinMaxScaler()
                data_normalized['volume'] = self.scalers['volume'].fit_transform(
                    data_normalized[['volume']]
                ).flatten()
            else:
                if 'volume' not in self.scalers:
                    raise DataPreprocessingError("Volume scaler not fitted")
                data_normalized['volume'] = self.scalers['volume'].transform(
                    data_normalized[['volume']]
                ).flatten()
        
        if fit:
            self.is_fitted = True
        
        return data_normalized
    
    def denormalize_prices(self, data: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """
        Denormalize price data back to original scale.
        
        Args:
            data: DataFrame with normalized data
            columns: List of columns to denormalize (default: all price columns)
            
        Returns:
            DataFrame with denormalized prices
        """
        if not self.is_fitted:
            raise DataPreprocessingError("Scalers not fitted")
        
        data_denormalized = data.copy()
        
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        for col in columns:
            if col in data_denormalized.columns and col in self.scalers:
                data_denormalized[col] = self.scalers[col].inverse_transform(
                    data_denormalized[[col]]
                ).flatten()
        
        return data_denormalized
    
    def create_time_splits(self, data: pd.DataFrame, train_ratio: float = 0.7, 
                          val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based splits for training, validation, and testing.
        
        Args:
            data: DataFrame with stock data (must be sorted by date)
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        self.logger.info(f"Creating time splits: train={train_ratio}, val={val_ratio}")
        
        if train_ratio + val_ratio >= 1.0:
            raise DataPreprocessingError("train_ratio + val_ratio must be < 1.0")
        
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date').reset_index(drop=True)
        
        n_samples = len(data)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        self.logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def detect_outliers(self, data: pd.DataFrame, columns: list = None, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in the data.
        
        Args:
            data: DataFrame with stock data
            columns: Columns to check for outliers
            method: Method to use ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        data_with_outliers = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
            
            else:
                raise DataPreprocessingError(f"Unknown outlier detection method: {method}")
            
            data_with_outliers[f'{col}_outlier'] = outliers
        
        return data_with_outliers
    
    def handle_outliers(self, data: pd.DataFrame, columns: list = None, 
                       method: str = 'cap', threshold: float = 3.0) -> pd.DataFrame:
        """
        Handle outliers in the data.
        
        Args:
            data: DataFrame with stock data
            columns: Columns to process
            method: Method to handle outliers ('cap', 'remove', or 'winsorize')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        self.logger.info(f"Handling outliers using method: {method}")
        
        if columns is None:
            columns = ['volume']  # Typically only handle volume outliers for stock data
        
        data_clean = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            # Detect outliers using IQR method
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                data_clean[col] = data_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            elif method == 'remove':
                outlier_mask = (data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)
                data_clean = data_clean[outlier_mask]
            
            elif method == 'winsorize':
                data_clean[col] = data_clean[col].clip(
                    lower=data_clean[col].quantile(0.05),
                    upper=data_clean[col].quantile(0.95)
                )
            
            else:
                raise DataPreprocessingError(f"Unknown outlier handling method: {method}")
        
        return data_clean
    
    def get_preprocessing_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get preprocessing statistics for the data.
        
        Args:
            data: DataFrame with stock data
            
        Returns:
            Dictionary with preprocessing statistics
        """
        stats = {
            'total_records': len(data),
            'date_range': {
                'start': data['date'].min() if 'date' in data.columns else None,
                'end': data['date'].max() if 'date' in data.columns else None
            },
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Add price statistics
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                stats[f'{col}_stats'] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                }
        
        return stats
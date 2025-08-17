"""
Unit tests for market microstructure feature extraction.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from enhanced_timeseries.features.microstructure_features import MicrostructureFeatures


class TestMicrostructureFeatures(unittest.TestCase):
    """Test MicrostructureFeatures class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='T')  # Minute data
        
        # Generate realistic intraday price data
        base_price = 100
        returns = np.random.normal(0, 0.001, 100)  # Small returns for minute data
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data with realistic spreads
        self.data = pd.DataFrame({
            'Date': dates,
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100, 5000, 100)
        })
        
        self.data.set_index('Date', inplace=True)
        
        self.microstructure = MicrostructureFeatures()
    
    def test_microstructure_creation(self):
        """Test microstructure features creation."""
        self.assertIsNotNone(self.microstructure)
        self.assertIsInstance(self.microstructure.feature_cache, dict)
    
    def test_calculate_all_microstructure_features(self):
        """Test calculating all microstructure features."""
        features = self.microstructure.calculate_all_microstructure_features(self.data)
        
        # Should return a dictionary
        self.assertIsInstance(features, dict)
        
        # Should have some features
        self.assertGreater(len(features), 0)
        
        # Each feature should be an array with same length as data
        for feature_name, feature_values in features.items():
            self.assertEqual(len(feature_values), len(self.data))
    
    def test_spread_features(self):
        """Test spread feature calculation."""
        # Test the private method through the public interface
        features = self.microstructure.calculate_all_microstructure_features(self.data)
        
        # Should have spread-related features
        spread_features = [k for k in features.keys() if 'spread' in k.lower()]
        self.assertGreater(len(spread_features), 0)
    
    def test_feature_values_reasonable(self):
        """Test that calculated features have reasonable values."""
        features = self.microstructure.calculate_all_microstructure_features(self.data)
        
        # Check that features don't have all NaN values
        for feature_name, feature_values in features.items():
            # At least some values should not be NaN
            non_nan_count = np.sum(~np.isnan(feature_values))
            self.assertGreater(non_nan_count, 0, f"Feature {feature_name} has all NaN values")
    
    def test_different_lookback_periods(self):
        """Test with different lookback periods."""
        custom_periods = [3, 7, 14]
        features = self.microstructure.calculate_all_microstructure_features(
            self.data, lookback_periods=custom_periods
        )
        
        # Should still return features
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Should handle empty data gracefully
        try:
            features = self.microstructure.calculate_all_microstructure_features(empty_data)
            # If it doesn't raise an exception, features should be empty or have zero-length arrays
            for feature_values in features.values():
                self.assertEqual(len(feature_values), 0)
        except Exception:
            # It's also acceptable to raise an exception for empty data
            pass
    
    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        incomplete_data = self.data[['Open', 'Close']].copy()  # Missing High, Low, Volume
        
        # Should handle missing columns gracefully or raise appropriate error
        try:
            features = self.microstructure.calculate_all_microstructure_features(incomplete_data)
            # If successful, should return some features (maybe fewer)
            self.assertIsInstance(features, dict)
        except (KeyError, ValueError):
            # It's acceptable to raise an error for missing required columns
            pass


if __name__ == '__main__':
    unittest.main()
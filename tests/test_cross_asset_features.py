"""
Unit tests for cross-asset and regime detection features.
"""

import unittest
import numpy as np
import pandas as pd
from enhanced_timeseries.features.cross_asset_features import CrossAssetFeatures


class TestCrossAssetFeatures(unittest.TestCase):
    """Test cases for cross-asset features."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample data for multiple assets
        n_days = 150
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
        
        self.data_dict = {}
        
        for i, symbol in enumerate(symbols):
            # Create correlated price series
            base_price = 100 + i * 10
            
            # Generate returns with some correlation structure
            market_factor = np.random.normal(0, 0.01, n_days)
            idiosyncratic = np.random.normal(0, 0.015, n_days)
            returns = 0.7 * market_factor + 0.3 * idiosyncratic
            
            # Generate prices
            prices = [base_price]
            for ret in returns:
                prices.append(prices[-1] * (1 + ret))
            
            prices = np.array(prices[1:])
            
            # Create OHLCV data
            self.data_dict[symbol] = pd.DataFrame({
                'Open': prices * (1 + np.random.normal(0, 0.002, n_days)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
                'Close': prices,
                'Volume': np.random.lognormal(15, 0.3, n_days).astype(int)
            })
            
            # Ensure price relationships
            data = self.data_dict[symbol]
            data['High'] = np.maximum(data['High'], 
                                    np.maximum(data['Open'], data['Close']))
            data['Low'] = np.minimum(data['Low'], 
                                   np.minimum(data['Open'], data['Close']))
        
        self.cross_asset = CrossAssetFeatures()
    
    def test_correlation_features(self):
        """Test correlation-based features."""
        features = self.cross_asset._correlation_features(
            self.data_dict, 'AAPL', [5, 10, 20]
        )
        
        # Check that correlation features are present
        expected_features = []
        for period in [5, 10, 20]:
            expected_features.extend([
                f'avg_correlation_{period}',
                f'max_correlation_{period}',
                f'min_correlation_{period}',
                f'correlation_dispersion_{period}',
                f'correlation_momentum_{period}'
            ])
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check correlation bounds
        for period in [5, 10, 20]:
            avg_corr = features[f'avg_correlation_{period}']
            max_corr = features[f'max_correlation_{period}']
            min_corr = features[f'min_correlation_{period}']
            
            # Remove NaN values for testing
            valid_avg = avg_corr[~np.isnan(avg_corr)]
            valid_max = max_corr[~np.isnan(max_corr)]
            valid_min = min_corr[~np.isnan(min_corr)]
            
            if len(valid_avg) > 0:
                self.assertTrue(np.all(valid_avg >= -1))
                self.assertTrue(np.all(valid_avg <= 1))
            
            if len(valid_max) > 0:
                self.assertTrue(np.all(valid_max >= -1))
                self.assertTrue(np.all(valid_max <= 1))
            
            if len(valid_min) > 0:
                self.assertTrue(np.all(valid_min >= -1))
                self.assertTrue(np.all(valid_min <= 1))
    
    def test_sector_momentum_features(self):
        """Test sector momentum features."""
        features = self.cross_asset._sector_momentum_features(
            self.data_dict, 'AAPL', [5, 10, 20]
        )
        
        # Check sector momentum features
        for period in [5, 10, 20]:
            expected_features = [
                f'own_sector_momentum_{period}',
                f'sector_relative_momentum_{period}',
                f'sector_dispersion_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check that values are finite
        for feature_name, values in features.items():
            finite_mask = np.isfinite(values)
            finite_ratio = np.sum(finite_mask) / len(values)
            self.assertGreater(finite_ratio, 0.5, 
                             f"Feature {feature_name} has too many non-finite values")
    
    def test_market_wide_features(self):
        """Test market-wide features."""
        features = self.cross_asset._market_wide_features(
            self.data_dict, 'AAPL', [5, 10, 20]
        )
        
        # Check market-wide features
        for period in [5, 10, 20]:
            expected_features = [
                f'market_momentum_{period}',
                f'market_volatility_{period}',
                f'market_volume_{period}',
                f'market_breadth_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check market breadth bounds (should be between 0 and 1)
        for period in [5, 10, 20]:
            breadth = features[f'market_breadth_{period}']
            valid_breadth = breadth[~np.isnan(breadth)]
            
            if len(valid_breadth) > 0:
                self.assertTrue(np.all(valid_breadth >= 0))
                self.assertTrue(np.all(valid_breadth <= 1))
        
        # Check volatility is non-negative
        for period in [5, 10, 20]:
            volatility = features[f'market_volatility_{period}']
            valid_vol = volatility[~np.isnan(volatility)]
            
            if len(valid_vol) > 0:
                self.assertTrue(np.all(valid_vol >= 0))
    
    def test_pca_features(self):
        """Test PCA features."""
        features = self.cross_asset._pca_features(
            self.data_dict, 'AAPL', [10, 20]
        )
        
        # Check PCA features
        for period in [10, 20]:
            expected_features = [
                f'pca_loading_1_{period}',
                f'pca_loading_2_{period}',
                f'pca_explained_var_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check explained variance bounds (should be between 0 and 1)
        for period in [10, 20]:
            explained_var = features[f'pca_explained_var_{period}']
            valid_var = explained_var[~np.isnan(explained_var)]
            
            if len(valid_var) > 0:
                self.assertTrue(np.all(valid_var >= 0))
                self.assertTrue(np.all(valid_var <= 1))
    
    def test_cross_volatility_features(self):
        """Test cross-asset volatility features."""
        features = self.cross_asset._cross_volatility_features(
            self.data_dict, 'AAPL', [5, 10, 20]
        )
        
        # Check cross-volatility features
        for period in [5, 10, 20]:
            expected_features = [
                f'cross_volatility_avg_{period}',
                f'cross_volatility_max_{period}',
                f'volatility_spillover_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check volatility is non-negative
        for period in [5, 10, 20]:
            avg_vol = features[f'cross_volatility_avg_{period}']
            max_vol = features[f'cross_volatility_max_{period}']
            
            valid_avg_vol = avg_vol[~np.isnan(avg_vol)]
            valid_max_vol = max_vol[~np.isnan(max_vol)]
            
            if len(valid_avg_vol) > 0:
                self.assertTrue(np.all(valid_avg_vol >= 0))
            
            if len(valid_max_vol) > 0:
                self.assertTrue(np.all(valid_max_vol >= 0))
        
        # Check spillover correlation bounds
        for period in [5, 10, 20]:
            spillover = features[f'volatility_spillover_{period}']
            valid_spillover = spillover[~np.isnan(spillover)]
            
            if len(valid_spillover) > 0:
                self.assertTrue(np.all(valid_spillover >= -1))
                self.assertTrue(np.all(valid_spillover <= 1))
    
    def test_market_regime_classification(self):
        """Test market regime classification."""
        features = self.cross_asset._market_regime_classification(
            self.data_dict['AAPL'], [20, 50]
        )
        
        # Check regime features
        for period in [20, 50]:
            expected_features = [
                f'market_regime_{period}',
                f'regime_persistence_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check regime values are valid
        for period in [20, 50]:
            regimes = features[f'market_regime_{period}']
            unique_regimes = np.unique(regimes)
            
            # Should be in range [-1, 2] (bear, neutral, bull, crisis)
            self.assertTrue(np.all(unique_regimes >= -1))
            self.assertTrue(np.all(unique_regimes <= 2))
            
            # Persistence should be positive
            persistence = features[f'regime_persistence_{period}']
            self.assertTrue(np.all(persistence >= 0))
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection."""
        features = self.cross_asset._volatility_regime_detection(
            self.data_dict['AAPL'], [20, 50]
        )
        
        # Check volatility regime features
        for period in [20, 50]:
            expected_features = [
                f'volatility_regime_{period}',
                f'vol_regime_change_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check regime values
        for period in [20, 50]:
            vol_regimes = features[f'volatility_regime_{period}']
            unique_regimes = np.unique(vol_regimes)
            
            # Should be 0, 1, or 2 (low, normal, high volatility)
            self.assertTrue(np.all(unique_regimes >= 0))
            self.assertTrue(np.all(unique_regimes <= 2))
    
    def test_trend_regime_detection(self):
        """Test trend regime detection."""
        features = self.cross_asset._trend_regime_detection(
            self.data_dict['AAPL'], [20, 50]
        )
        
        # Check trend regime features
        for period in [20, 50]:
            expected_features = [
                f'trend_regime_{period}',
                f'trend_strength_{period}'
            ]
            
            for feature in expected_features:
                self.assertIn(feature, features)
                self.assertEqual(len(features[feature]), len(self.data_dict['AAPL']))
        
        # Check trend regime values
        for period in [20, 50]:
            trend_regimes = features[f'trend_regime_{period}']
            unique_regimes = np.unique(trend_regimes)
            
            # Should be -1, 0, or 1 (downtrend, no trend, uptrend)
            self.assertTrue(np.all(unique_regimes >= -1))
            self.assertTrue(np.all(unique_regimes <= 1))
            
            # Trend strength should be between 0 and 1
            trend_strength = features[f'trend_strength_{period}']
            valid_strength = trend_strength[~np.isnan(trend_strength)]
            
            if len(valid_strength) > 0:
                self.assertTrue(np.all(valid_strength >= 0))
                self.assertTrue(np.all(valid_strength <= 1))
    
    def test_structural_break_detection(self):
        """Test structural break detection."""
        features = self.cross_asset._structural_break_detection(
            self.data_dict['AAPL'], [20, 50]
        )
        
        # Check structural break features
        for period in [20, 50]:
            feature_name = f'structural_break_{period}'
            self.assertIn(feature_name, features)
            self.assertEqual(len(features[feature_name]), len(self.data_dict['AAPL']))
            
            # Should be 0 or 1 (no break, break detected)
            breaks = features[feature_name]
            unique_breaks = np.unique(breaks)
            self.assertTrue(np.all(unique_breaks >= 0))
            self.assertTrue(np.all(unique_breaks <= 1))
    
    def test_regime_transition_features(self):
        """Test regime transition features."""
        features = self.cross_asset._regime_transition_features(
            self.data_dict['AAPL'], [20, 50]
        )
        
        # Check transition probability features
        for period in [20, 50]:
            feature_name = f'regime_stay_prob_{period}'
            self.assertIn(feature_name, features)
            self.assertEqual(len(features[feature_name]), len(self.data_dict['AAPL']))
            
            # Probabilities should be between 0 and 1
            probs = features[feature_name]
            valid_probs = probs[~np.isnan(probs)]
            
            if len(valid_probs) > 0:
                self.assertTrue(np.all(valid_probs >= 0))
                self.assertTrue(np.all(valid_probs <= 1))
    
    def test_calculate_cross_asset_features(self):
        """Test calculation of all cross-asset features."""
        features = self.cross_asset.calculate_cross_asset_features(
            self.data_dict, 'AAPL', [5, 10, 20]
        )
        
        # Should have many cross-asset features
        self.assertGreater(len(features), 15)
        
        # All features should have correct length
        for feature_name, values in features.items():
            self.assertEqual(len(values), len(self.data_dict['AAPL']), 
                           f"Feature {feature_name} has wrong length")
        
        # Check that most values are finite
        for feature_name, values in features.items():
            finite_mask = np.isfinite(values)
            finite_ratio = np.sum(finite_mask) / len(values)
            self.assertGreater(finite_ratio, 0.3, 
                             f"Feature {feature_name} has too many non-finite values")
    
    def test_calculate_regime_features(self):
        """Test calculation of all regime features."""
        features = self.cross_asset.calculate_regime_features(
            self.data_dict['AAPL'], [20, 50]
        )
        
        # Should have regime features
        self.assertGreater(len(features), 10)
        
        # All features should have correct length
        for feature_name, values in features.items():
            self.assertEqual(len(values), len(self.data_dict['AAPL']), 
                           f"Feature {feature_name} has wrong length")
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with single asset (no cross-asset features possible)
        single_asset_dict = {'AAPL': self.data_dict['AAPL']}
        features = self.cross_asset.calculate_cross_asset_features(
            single_asset_dict, 'AAPL', [5, 10]
        )
        
        # Should still return features (mostly zeros)
        self.assertGreater(len(features), 0)
        
        # Test with minimal data
        small_data = self.data_dict['AAPL'].head(10)
        regime_features = self.cross_asset.calculate_regime_features(small_data, [5])
        
        # Should handle small datasets gracefully
        self.assertGreater(len(regime_features), 0)
        
        # Test with constant prices
        constant_data = self.data_dict['AAPL'].copy()
        constant_data['Close'] = 100.0
        
        regime_features = self.cross_asset.calculate_regime_features(constant_data, [20])
        
        # Should handle constant prices without crashing
        self.assertGreater(len(regime_features), 0)


if __name__ == '__main__':
    unittest.main()
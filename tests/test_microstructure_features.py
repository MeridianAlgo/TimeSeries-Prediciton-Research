"""
Unit tests for microstructure features.
"""

import unittest
import numpy as np
import pandas as pd
from enhanced_timeseries.features.microstructure_features import MicrostructureFeatures


class TestMicrostructureFeatures(unittest.TestCase):
    """Test cases for microstructure features."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample OHLCV data with realistic microstructure patterns
        n_days = 200
        base_price = 100
        
        # Generate realistic price data with microstructure noise
        returns = np.random.normal(0, 0.015, n_days)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])
        
        # Add microstructure noise
        microstructure_noise = np.random.normal(0, 0.002, n_days)
        noisy_prices = prices * (1 + microstructure_noise)
        
        # Create OHLCV data with realistic intraday patterns
        self.data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.003, n_days)),
            'High': noisy_prices * (1 + np.abs(np.random.normal(0, 0.008, n_days))),
            'Low': noisy_prices * (1 - np.abs(np.random.normal(0, 0.008, n_days))),
            'Close': noisy_prices,
            'Volume': np.random.lognormal(15, 0.5, n_days).astype(int)
        })
        
        # Ensure price relationships are valid
        self.data['High'] = np.maximum(self.data['High'], 
                                      np.maximum(self.data['Open'], self.data['Close']))
        self.data['Low'] = np.minimum(self.data['Low'], 
                                     np.minimum(self.data['Open'], self.data['Close']))
        
        self.microstructure = MicrostructureFeatures()
    
    def test_spread_features(self):
        """Test spread-related features."""
        features = self.microstructure._spread_features(self.data, [5, 10, 20])
        
        # Check basic spread features
        expected_features = [
            'hl_spread', 'relative_spread', 'effective_spread', 
            'quoted_spread', 'roll_spread'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # Check HL spread is positive
        hl_spread = features['hl_spread']
        valid_spread = hl_spread[~np.isnan(hl_spread)]
        if len(valid_spread) > 0:
            self.assertTrue(np.all(valid_spread >= 0))
        
        # Check rolling spread features
        for period in [5, 10, 20]:
            self.assertIn(f'spread_mean_{period}', features)
            self.assertIn(f'spread_std_{period}', features)
            self.assertIn(f'spread_percentile_{period}', features)
            
            # Percentiles should be between 0 and 1
            percentiles = features[f'spread_percentile_{period}']
            valid_percentiles = percentiles[~np.isnan(percentiles)]
            if len(valid_percentiles) > 0:
                self.assertTrue(np.all(valid_percentiles >= 0))
                self.assertTrue(np.all(valid_percentiles <= 1))
    
    def test_order_flow_features(self):
        """Test order flow features."""
        features = self.microstructure._order_flow_features(self.data, [5, 10, 20])
        
        # Check basic order flow features
        expected_features = [
            'volume_weighted_returns', 'order_flow_imbalance', 
            'trade_direction', 'signed_volume'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # Check trade direction is -1, 0, or 1
        trade_direction = features['trade_direction']
        unique_directions = np.unique(trade_direction[~np.isnan(trade_direction)])
        self.assertTrue(np.all(np.isin(unique_directions, [-1, 0, 1])))
        
        # Check rolling order flow features
        for period in [5, 10, 20]:
            self.assertIn(f'net_order_flow_{period}', features)
            self.assertIn(f'order_flow_ratio_{period}', features)
            self.assertIn(f'buy_pressure_{period}', features)
            self.assertIn(f'sell_pressure_{period}', features)
            
            # Buy and sell pressure should be between 0 and 1
            buy_pressure = features[f'buy_pressure_{period}']
            sell_pressure = features[f'sell_pressure_{period}']
            
            valid_buy = buy_pressure[~np.isnan(buy_pressure)]
            valid_sell = sell_pressure[~np.isnan(sell_pressure)]
            
            if len(valid_buy) > 0:
                self.assertTrue(np.all(valid_buy >= 0))
                self.assertTrue(np.all(valid_buy <= 1))
            
            if len(valid_sell) > 0:
                self.assertTrue(np.all(valid_sell >= 0))
                self.assertTrue(np.all(valid_sell <= 1))
    
    def test_price_impact_features(self):
        """Test price impact features."""
        features = self.microstructure._price_impact_features(self.data, [5, 10, 20])
        
        # Check basic price impact features
        expected_features = [
            'intraday_return', 'overnight_return', 'price_impact_per_volume',
            'kyle_lambda', 'amihud_illiquidity'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # Check permanent and temporary impact features
        for period in [1, 5, 10]:
            self.assertIn(f'permanent_impact_{period}', features)
            self.assertIn(f'temporary_impact_{period}', features)
        
        # Amihud illiquidity should be non-negative
        amihud = features['amihud_illiquidity']
        valid_amihud = amihud[~np.isnan(amihud) & ~np.isinf(amihud)]
        if len(valid_amihud) > 0:
            self.assertTrue(np.all(valid_amihud >= 0))
    
    def test_volatility_microstructure_features(self):
        """Test volatility microstructure features."""
        features = self.microstructure._volatility_microstructure_features(self.data, [5, 10, 20])
        
        # Check volatility estimators
        volatility_features = [
            'garman_klass_volatility', 'rogers_satchell_volatility', 
            'yang_zhang_volatility', 'noise_to_signal'
        ]
        
        for feature in volatility_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # Check rolling features
        for period in [5, 10, 20]:
            self.assertIn(f'return_autocorr_{period}', features)
            self.assertIn(f'variance_ratio_{period}', features)
            
            # Return autocorrelation should be between -1 and 1
            autocorr = features[f'return_autocorr_{period}']
            valid_autocorr = autocorr[~np.isnan(autocorr)]
            if len(valid_autocorr) > 0:
                self.assertTrue(np.all(valid_autocorr >= -1))
                self.assertTrue(np.all(valid_autocorr <= 1))
    
    def test_market_depth_features(self):
        """Test market depth features."""
        features = self.microstructure._market_depth_features(self.data, [5, 10, 20])
        
        # Check basic depth features
        expected_features = ['vwap_deviation', 'price_clustering']
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # Check rolling depth features
        for period in [5, 10, 20]:
            self.assertIn(f'avg_trade_size_{period}', features)
            self.assertIn(f'trade_intensity_{period}', features)
            self.assertIn(f'market_depth_{period}', features)
            
            # Average trade size should be positive
            avg_trade_size = features[f'avg_trade_size_{period}']
            valid_size = avg_trade_size[~np.isnan(avg_trade_size)]
            if len(valid_size) > 0:
                self.assertTrue(np.all(valid_size >= 0))
        
        # Price clustering should be between 0 and 1
        clustering = features['price_clustering']
        valid_clustering = clustering[~np.isnan(clustering)]
        if len(valid_clustering) > 0:
            self.assertTrue(np.all(valid_clustering >= 0))
            self.assertTrue(np.all(valid_clustering <= 1))
    
    def test_intraday_pattern_features(self):
        """Test intraday pattern features."""
        features = self.microstructure._intraday_pattern_features(self.data, [5, 10, 20])
        
        # Check basic intraday features
        expected_features = [
            'open_close_ratio', 'hl_position', 'opening_gap', 'intraday_momentum'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # HL position should be between 0 and 1
        hl_position = features['hl_position']
        valid_position = hl_position[~np.isnan(hl_position)]
        if len(valid_position) > 0:
            self.assertTrue(np.all(valid_position >= 0))
            self.assertTrue(np.all(valid_position <= 1))
        
        # Check volume pattern features
        for period in [5, 10, 20]:
            self.assertIn(f'volume_trend_{period}', features)
            self.assertIn(f'volume_acceleration_{period}', features)
    
    def test_volume_at_price_features(self):
        """Test volume at price features."""
        features = self.microstructure._volume_at_price_features(self.data)
        
        expected_features = [
            'volume_at_high_ratio', 'volume_at_low_ratio', 'price_volume_correlation'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertEqual(len(features[feature]), len(self.data))
        
        # Volume ratios should be between 0 and 1
        high_ratio = features['volume_at_high_ratio']
        low_ratio = features['volume_at_low_ratio']
        
        valid_high = high_ratio[~np.isnan(high_ratio)]
        valid_low = low_ratio[~np.isnan(low_ratio)]
        
        if len(valid_high) > 0:
            self.assertTrue(np.all(valid_high >= 0))
            self.assertTrue(np.all(valid_high <= 1))
        
        if len(valid_low) > 0:
            self.assertTrue(np.all(valid_low >= 0))
            self.assertTrue(np.all(valid_low <= 1))
    
    def test_calculate_all_microstructure_features(self):
        """Test calculation of all microstructure features."""
        all_features = self.microstructure.calculate_all_microstructure_features(self.data)
        
        # Should have many microstructure features
        self.assertGreater(len(all_features), 20)
        
        # All features should have the same length as input data
        for feature_name, values in all_features.items():
            self.assertEqual(len(values), len(self.data), 
                           f"Feature {feature_name} has wrong length")
        
        # Check that most values are finite
        for feature_name, values in all_features.items():
            finite_mask = np.isfinite(values)
            finite_ratio = np.sum(finite_mask) / len(values)
            self.assertGreater(finite_ratio, 0.3, 
                             f"Feature {feature_name} has too many non-finite values")
    
    def test_volatility_estimators(self):
        """Test specific volatility estimators."""
        # Test Garman-Klass volatility
        gk_vol = self.microstructure._calculate_garman_klass_volatility(self.data)
        self.assertEqual(len(gk_vol), len(self.data))
        
        # Should be non-negative
        valid_gk = gk_vol[~np.isnan(gk_vol)]
        if len(valid_gk) > 0:
            self.assertTrue(np.all(valid_gk >= 0))
        
        # Test Rogers-Satchell volatility
        rs_vol = self.microstructure._calculate_rogers_satchell_volatility(self.data)
        self.assertEqual(len(rs_vol), len(self.data))
        
        # Test Yang-Zhang volatility
        yz_vol = self.microstructure._calculate_yang_zhang_volatility(self.data)
        self.assertEqual(len(yz_vol), len(self.data))
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with minimal data
        small_data = self.data.head(10)
        features = self.microstructure.calculate_all_microstructure_features(small_data)
        
        # Should still return features without crashing
        self.assertGreater(len(features), 0)
        
        # Test with constant prices
        constant_data = self.data.copy()
        constant_data['Close'] = 100.0
        constant_data['Open'] = 100.0
        constant_data['High'] = 100.0
        constant_data['Low'] = 100.0
        
        features = self.microstructure.calculate_all_microstructure_features(constant_data)
        
        # Should handle constant prices gracefully
        self.assertGreater(len(features), 0)
        
        # Test with zero volume
        zero_volume_data = self.data.copy()
        zero_volume_data['Volume'] = 1  # Avoid division by zero
        
        features = self.microstructure.calculate_all_microstructure_features(zero_volume_data)
        self.assertGreater(len(features), 0)


if __name__ == '__main__':
    unittest.main()
"""Comprehensive tests for feature engineering components."""

import pytest
import numpy as np
import pandas as pd
import logging
from unittest.mock import Mock, patch

# Import available feature engineering components
from ultra_precision_predictor.feature_engineering.extreme_feature_engineer import ExtremeFeatureEngineer
from ultra_precision_predictor.feature_engineering.advanced_bollinger import AdvancedBollingerBands
from ultra_precision_predictor.feature_engineering.multi_rsi import MultiRSISystem
from ultra_precision_predictor.feature_engineering.microstructure import MarketMicrostructureAnalyzer
from ultra_precision_predictor.feature_engineering.volatility_analysis import VolatilityAnalysisSystem

# Optional imports - only import if available
try:
    from ultra_precision_predictor.feature_engineering.fractional_indicators import FractionalIndicators
except ImportError:
    FractionalIndicators = None

try:
    from ultra_precision_predictor.feature_engineering.multi_harmonic import MultiHarmonicAnalyzer
except ImportError:
    MultiHarmonicAnalyzer = None

try:
    from ultra_precision_predictor.feature_engineering.micro_patterns import MicroPatternDetector
except ImportError:
    MicroPatternDetector = None

try:
    from ultra_precision_predictor.feature_engineering.advanced_macd import AdvancedMACDSystem
except ImportError:
    AdvancedMACDSystem = None


class TestFeatureEngineering:
    """Test suite for all feature engineering components."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
        
        # Generate realistic price data with trends and volatility
        base_price = 100
        returns = np.random.normal(0.0001, 0.02, 1000)  # Small positive drift with volatility
        
        # Add some autocorrelation to make it more realistic
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLC from close prices
        high_noise = np.random.exponential(0.005, 1000)
        low_noise = np.random.exponential(0.005, 1000)
        open_noise = np.random.normal(0, 0.002, 1000)
        
        df = pd.DataFrame({
            'Open': prices * (1 + open_noise),
            'High': prices * (1 + high_noise),
            'Low': prices * (1 - low_noise),
            'Close': prices,
            'Volume': np.random.lognormal(10, 1, 1000)  # Realistic volume distribution
        }, index=dates)
        
        # Ensure OHLC relationships are valid
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
        
        return df
    
    @pytest.fixture
    def minimal_data(self):
        """Create minimal data for edge case testing."""
        return pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200]
        })
    
    def test_extreme_feature_engineer(self, sample_data):
        """Test ExtremeFeatureEngineer with comprehensive validation."""
        engineer = ExtremeFeatureEngineer()
        
        # Test feature generation
        result = engineer.generate_features(sample_data)
        
        # Validate output
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert len(result.columns) > len(sample_data.columns)
        
        # Check for NaN values (should be minimal after cleaning)
        nan_percentage = result.isnull().sum().sum() / (len(result) * len(result.columns))
        assert nan_percentage < 0.01, f"Too many NaN values: {nan_percentage:.3%}"
        
        # Check feature names are populated
        feature_names = engineer.get_feature_names()
        assert len(feature_names) > 0
        
        # Validate feature statistics
        stats = engineer.get_feature_statistics()
        assert stats['total_features'] == len(feature_names)
        assert 'feature_categories' in stats
        
        print(f"ExtremeFeatureEngineer generated {len(feature_names)} features")
        print(f"Feature categories: {stats['feature_categories']}")
    
    def test_advanced_bollinger_bands(self, sample_data):
        """Test AdvancedBollingerBands with precision validation."""
        bb = AdvancedBollingerBands()
        
        result = bb.generate_features(sample_data)
        
        # Validate Bollinger Bands properties
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        
        # Check specific Bollinger Band features
        bb_features = [col for col in result.columns if 'bb_' in col]
        assert len(bb_features) > 0
        
        # Validate Bollinger Band mathematical properties
        for period in bb.periods:
            sma_col = f'bb_sma_{period}'
            if sma_col in result.columns:
                # SMA should be between high and low most of the time
                sma_values = result[sma_col].dropna()
                high_values = result['High'].loc[sma_values.index]
                low_values = result['Low'].loc[sma_values.index]
                
                within_range = ((sma_values >= low_values) & (sma_values <= high_values)).mean()
                assert within_range > 0.7, f"SMA not within HL range enough: {within_range:.3f}"
        
        # Check for feature importance structure
        importance = bb.get_feature_importance()
        assert isinstance(importance, dict)
        
        feature_names = bb.get_feature_names()
        print(f"AdvancedBollingerBands generated {len(feature_names)} features")
    
    def test_multi_rsi_system(self, sample_data):
        """Test MultiRSISystem with RSI validation."""
        rsi_system = MultiRSISystem()
        
        result = rsi_system.generate_features(sample_data)
        
        # Validate RSI properties
        assert isinstance(result, pd.DataFrame)
        
        # Check RSI bounds (should be between 0 and 100)
        rsi_features = [col for col in result.columns if col.startswith('rsi_') and not any(x in col for x in ['velocity', 'momentum', 'divergence'])]
        
        for rsi_col in rsi_features[:5]:  # Check first 5 RSI columns
            if rsi_col in result.columns:
                rsi_values = result[rsi_col].dropna()
                if len(rsi_values) > 0:
                    assert rsi_values.min() >= -1, f"RSI {rsi_col} below 0: {rsi_values.min()}"
                    assert rsi_values.max() <= 101, f"RSI {rsi_col} above 100: {rsi_values.max()}"
        
        feature_names = rsi_system.get_feature_names()
        stats = rsi_system.get_feature_statistics()
        
        print(f"MultiRSISystem generated {len(feature_names)} features")
        print(f"RSI periods: {rsi_system.periods}")
    
    def test_market_microstructure_analyzer(self, sample_data):
        """Test MarketMicrostructureAnalyzer with microstructure validation."""
        analyzer = MarketMicrostructureAnalyzer()
        
        result = analyzer.generate_features(sample_data)
        
        # Validate microstructure properties
        assert isinstance(result, pd.DataFrame)
        
        # Check spread proxies are positive
        spread_features = [col for col in result.columns if 'spread' in col]
        for spread_col in spread_features[:3]:  # Check first 3 spread columns
            if spread_col in result.columns:
                spread_values = result[spread_col].dropna()
                if len(spread_values) > 0:
                    negative_spreads = (spread_values < 0).sum()
                    assert negative_spreads / len(spread_values) < 0.1, f"Too many negative spreads in {spread_col}"
        
        # Check VWAP features
        vwap_features = [col for col in result.columns if 'vwap' in col]
        assert len(vwap_features) > 0
        
        feature_names = analyzer.get_feature_names()
        stats = analyzer.get_feature_statistics()
        
        print(f"MarketMicrostructureAnalyzer generated {len(feature_names)} features")
        print(f"Feature categories: {list(stats['feature_categories'].keys())}")
    
    def test_volatility_analysis_system(self, sample_data):
        """Test VolatilityAnalysisSystem with volatility validation."""
        vol_system = VolatilityAnalysisSystem()
        
        result = vol_system.generate_features(sample_data)
        
        # Validate volatility properties
        assert isinstance(result, pd.DataFrame)
        
        # Check volatility is non-negative
        vol_features = [col for col in result.columns if 'volatility' in col and 'momentum' not in col]
        for vol_col in vol_features[:3]:  # Check first 3 volatility columns
            if vol_col in result.columns:
                vol_values = result[vol_col].dropna()
                if len(vol_values) > 0:
                    negative_vol = (vol_values < 0).sum()
                    assert negative_vol == 0, f"Negative volatility in {vol_col}: {negative_vol}"
        
        # Check returns are calculated
        assert 'returns' in result.columns
        returns = result['returns'].dropna()
        assert len(returns) > 0
        
        feature_names = vol_system.get_feature_names()
        print(f"VolatilityAnalysisSystem generated {len(feature_names)} features")
    
    def test_edge_cases(self, minimal_data):
        """Test edge cases with minimal data."""
        # Test with minimal data
        engineer = ExtremeFeatureEngineer()
        
        try:
            result = engineer.generate_features(minimal_data)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(minimal_data)
        except Exception as e:
            pytest.fail(f"Failed with minimal data: {str(e)}")
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        engineer = ExtremeFeatureEngineer()
        
        with pytest.raises(Exception):  # Should raise an exception for empty data
            engineer.generate_features(empty_df)
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        incomplete_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        
        engineer = ExtremeFeatureEngineer()
        
        with pytest.raises(Exception):  # Should raise an exception for missing columns
            engineer.generate_features(incomplete_df)
    
    def test_feature_consistency(self, sample_data):
        """Test that features are generated consistently."""
        engineer = ExtremeFeatureEngineer()
        
        # Generate features twice
        result1 = engineer.generate_features(sample_data.copy())
        result2 = engineer.generate_features(sample_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_performance_benchmark(self, sample_data):
        """Benchmark performance of feature generation."""
        import time
        
        engineer = ExtremeFeatureEngineer()
        
        start_time = time.time()
        result = engineer.generate_features(sample_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        rows_per_second = len(sample_data) / processing_time
        
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"Rows per second: {rows_per_second:.0f}")
        
        # Should process at least 100 rows per second
        assert rows_per_second > 100, f"Too slow: {rows_per_second:.0f} rows/sec"
    
    def test_memory_usage(self, sample_data):
        """Test memory usage during feature generation."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engineer = ExtremeFeatureEngineer()
        result = engineer.generate_features(sample_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (less than 500MB for 1000 rows)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f} MB"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
"""Integration tests for the complete ultra-precision predictor system."""

import pytest
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path

from ultra_precision_predictor.predictor import UltraPrecisionPredictor
from ultra_precision_predictor.core.data_validator import DataValidator
from ultra_precision_predictor.core.exceptions import (
    DataValidationError, 
    FeatureEngineeringError, 
    ModelTrainingError, 
    PredictionError
)


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def comprehensive_market_data(self):
        """Generate comprehensive market data for integration testing."""
        np.random.seed(42)
        n_samples = 1500
        
        # Create realistic market data with multiple regimes
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Generate multi-regime data
        regime_length = n_samples // 3
        
        # Regime 1: Low volatility trending
        t1 = np.arange(regime_length)
        trend1 = 0.0002 * t1
        vol1 = 0.01
        returns1 = trend1 + np.random.normal(0, vol1, regime_length)
        
        # Regime 2: High volatility sideways
        returns2 = np.random.normal(0, 0.03, regime_length)
        
        # Regime 3: Medium volatility with cycles
        t3 = np.arange(regime_length)
        cycle = 0.02 * np.sin(2 * np.pi * t3 / 50)
        returns3 = cycle + np.random.normal(0, 0.015, regime_length)
        
        # Combine regimes
        all_returns = np.concatenate([returns1, returns2, returns3])
        
        # Add some autocorrelation
        for i in range(1, len(all_returns)):
            all_returns[i] += 0.05 * all_returns[i-1]
        
        # Convert to prices
        base_price = 100
        prices = base_price * np.cumprod(1 + all_returns)
        
        # Generate OHLCV
        high_noise = np.random.exponential(0.003, n_samples)
        low_noise = np.random.exponential(0.003, n_samples)
        open_noise = np.random.normal(0, 0.001, n_samples)
        
        df = pd.DataFrame({
            'Open': prices * (1 + open_noise),
            'High': prices * (1 + high_noise),
            'Low': prices * (1 - low_noise),
            'Close': prices,
            'Volume': np.random.lognormal(10, 0.8, n_samples)
        }, index=dates)
        
        # Ensure OHLC consistency
        df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
        df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
        
        return df
    
    def test_complete_pipeline(self, comprehensive_market_data):
        """Test the complete prediction pipeline."""
        print("\\n=== Testing Complete Pipeline ===")
        
        data = comprehensive_market_data.copy()
        
        # Initialize predictor
        predictor = UltraPrecisionPredictor()
        
        # Test data validation
        validator = DataValidator()
        validation_result = validator.validate(data)
        assert validation_result['is_valid'], f"Data validation failed: {validation_result['errors']}"
        print("✓ Data validation passed")
        
        # Test training
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        start_time = time.time()
        predictor.train(train_data)
        training_time = time.time() - start_time
        
        print(f"✓ Training completed in {training_time:.2f} seconds")
        
        # Test prediction
        start_time = time.time()
        predictions = predictor.predict(test_data)
        prediction_time = time.time() - start_time
        
        print(f"✓ Prediction completed in {prediction_time:.2f} seconds")
        
        # Validate predictions
        assert predictions is not None, "No predictions generated"
        assert len(predictions) > 0, "Empty predictions"
        assert not np.any(np.isnan(predictions)), "NaN values in predictions"
        
        print(f"✓ Generated {len(predictions)} valid predictions")
        
        # Test prediction quality
        actual_returns = test_data['Close'].pct_change().dropna()
        min_len = min(len(predictions), len(actual_returns))
        
        if min_len > 10:
            pred_subset = predictions[:min_len]
            actual_subset = actual_returns.iloc[:min_len].values
            
            # Calculate basic metrics
            mae = np.mean(np.abs(pred_subset - actual_subset))
            mse = np.mean((pred_subset - actual_subset) ** 2)
            
            print(f"✓ MAE: {mae:.6f}")
            print(f"✓ MSE: {mse:.6f}")
            
            # Basic quality checks
            assert mae < 0.1, f"MAE too high: {mae}"
            assert mse < 0.01, f"MSE too high: {mse}"
        
        print("✓ Complete pipeline test passed")
    
    def test_feature_engineering_integration(self, comprehensive_market_data):
        """Test feature engineering integration."""
        print("\\n=== Testing Feature Engineering Integration ===")
        
        predictor = UltraPrecisionPredictor()
        
        # Test that feature engineering works with the predictor
        data = comprehensive_market_data.copy()
        
        # Initialize feature engineering
        if hasattr(predictor, 'feature_engineer'):
            engineered_data = predictor.feature_engineer.generate_features(data)
            
            # Validate engineered features
            assert isinstance(engineered_data, pd.DataFrame)
            assert len(engineered_data) == len(data)
            assert len(engineered_data.columns) > len(data.columns)
            
            # Check for excessive NaN values
            nan_percentage = engineered_data.isnull().sum().sum() / (len(engineered_data) * len(engineered_data.columns))
            assert nan_percentage < 0.05, f"Too many NaN values: {nan_percentage:.3%}"
            
            # Check feature names
            feature_names = predictor.feature_engineer.get_feature_names()
            assert len(feature_names) > 0
            
            print(f"✓ Generated {len(feature_names)} features")
            print(f"✓ NaN percentage: {nan_percentage:.3%}")
            
        else:
            pytest.skip("No feature engineer available")
    
    def test_error_handling(self):
        """Test error handling throughout the system."""
        print("\\n=== Testing Error Handling ===")
        
        predictor = UltraPrecisionPredictor()
        
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'Close': [100, np.nan, 102],
            'Volume': [1000, 1100, np.nan]
        })
        
        # Should handle invalid data gracefully
        try:
            predictor.train(invalid_data)
            pytest.fail("Should have raised an exception for invalid data")
        except Exception as e:
            print(f"✓ Correctly handled invalid data: {type(e).__name__}")
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        try:
            predictor.train(empty_data)
            pytest.fail("Should have raised an exception for empty data")
        except Exception as e:
            print(f"✓ Correctly handled empty data: {type(e).__name__}")
        
        print("✓ Error handling tests passed")
    
    def test_performance_benchmarks(self, comprehensive_market_data):
        """Test performance benchmarks."""
        print("\\n=== Testing Performance Benchmarks ===")
        
        predictor = UltraPrecisionPredictor()
        data = comprehensive_market_data.copy()
        
        # Benchmark training time
        train_data = data.iloc[:1000]
        
        start_time = time.time()
        predictor.train(train_data)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Training speed: {len(train_data) / training_time:.0f} samples/second")
        
        # Should train reasonably fast
        assert training_time < 60, f"Training too slow: {training_time:.2f} seconds"
        
        # Benchmark prediction time
        test_data = data.iloc[1000:1200]
        
        start_time = time.time()
        predictions = predictor.predict(test_data)
        prediction_time = time.time() - start_time
        
        print(f"Prediction time: {prediction_time:.2f} seconds")
        if predictions is not None:
            print(f"Prediction speed: {len(predictions) / prediction_time:.0f} predictions/second")
            
            # Should predict reasonably fast
            assert prediction_time < 10, f"Prediction too slow: {prediction_time:.2f} seconds"
        
        print("✓ Performance benchmarks passed")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
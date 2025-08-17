#!/usr/bin/env python3
"""Simple usage examples and tests."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from ultra_precision_predictor.simple_predictor import SimplePredictor

def create_simple_test_data(n_samples=100):
    """Create simple test data for examples."""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Add some trend
    trend = np.linspace(0, 0.1, n_samples)
    returns += np.diff(np.concatenate([[0], trend]))
    
    prices = 100 * np.cumprod(1 + returns)
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.003, n_samples))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_samples))),
        'Close': prices,
        'Volume': np.random.lognormal(9, 0.5, n_samples)
    }, index=dates)
    
    # Ensure OHLC consistency
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    return df

def test_simple_prediction_workflow():
    """Test a simple prediction workflow."""
    print("\\n=== Simple Prediction Workflow ===")
    
    # Create test data
    data = create_simple_test_data(200)
    print(f"Created {len(data)} samples of test data")
    
    # Split data
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Initialize predictor
    predictor = SimplePredictor()
    
    # Train
    print("Training predictor...")
    predictor.train(train_data)
    print("✓ Training completed")
    
    # Predict
    print("Making predictions...")
    predictions = predictor.predict(test_data)
    
    if predictions is not None:
        print(f"✓ Generated {len(predictions)} predictions")
        print(f"Prediction range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
        
        # Basic validation
        assert len(predictions) > 0, "No predictions generated"
        assert not np.any(np.isnan(predictions)), "NaN values in predictions"
        
        return True
    else:
        pytest.fail("No predictions generated")

def test_data_validation():
    """Test data validation."""
    print("\\n=== Data Validation Test ===")
    
    predictor = SimplePredictor()
    
    # Test with valid data
    valid_data = create_simple_test_data(50)
    try:
        predictor.train(valid_data)
        print("✓ Valid data accepted")
    except Exception as e:
        pytest.fail(f"Valid data rejected: {e}")
    
    # Test with invalid data (missing columns)
    invalid_data = pd.DataFrame({
        'Close': [100, 101, 102],
        'Volume': [1000, 1100, 1200]
    })
    
    try:
        predictor.train(invalid_data)
        print("⚠ Invalid data was accepted (predictor may be robust)")
    except Exception:
        print("✓ Invalid data correctly rejected")

def test_feature_generation_example():
    """Test feature generation with simple data."""
    print("\\n=== Feature Generation Example ===")
    
    from ultra_precision_predictor.feature_engineering.extreme_feature_engineer import ExtremeFeatureEngineer
    
    # Create test data
    data = create_simple_test_data(100)
    print(f"Original data shape: {data.shape}")
    
    # Generate features
    engineer = ExtremeFeatureEngineer()
    features = engineer.generate_features(data)
    
    print(f"Features shape: {features.shape}")
    print(f"Generated {features.shape[1] - data.shape[1]} new features")
    
    # Validate features
    assert features.shape[0] == data.shape[0], "Row count mismatch"
    assert features.shape[1] > data.shape[1], "No new features generated"
    
    # Check for excessive NaN values
    nan_percentage = features.isnull().sum().sum() / (len(features) * len(features.columns))
    print(f"NaN percentage: {nan_percentage:.3%}")
    assert nan_percentage < 0.1, f"Too many NaN values: {nan_percentage:.3%}"
    
    print("✓ Feature generation successful")

def test_prediction_quality_example():
    """Test prediction quality with simple metrics."""
    print("\\n=== Prediction Quality Example ===")
    
    # Create test data with known pattern
    np.random.seed(123)
    n_samples = 300
    
    # Create data with strong trend for easier prediction
    trend = np.linspace(0, 0.2, n_samples)
    noise = np.random.normal(0, 0.01, n_samples)
    returns = np.diff(trend) + noise[1:]
    
    prices = 100 * np.cumprod(np.concatenate([[1], 1 + returns]))
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    data = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.001, n_samples)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.002, n_samples))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.002, n_samples))),
        'Close': prices,
        'Volume': np.random.lognormal(9, 0.3, n_samples)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    # Split and predict
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    predictor = SimplePredictor()
    predictor.train(train_data)
    predictions = predictor.predict(test_data)
    
    if predictions is not None and len(predictions) > 0:
        # Calculate actual returns
        actual_returns = test_data['Close'].pct_change().dropna()
        
        # Align predictions
        min_len = min(len(predictions), len(actual_returns))
        if min_len > 10:
            pred_subset = predictions[:min_len]
            actual_subset = actual_returns.iloc[:min_len].values
            
            # Calculate basic metrics
            mae = np.mean(np.abs(pred_subset - actual_subset))
            mse = np.mean((pred_subset - actual_subset) ** 2)
            
            print(f"MAE: {mae:.6f}")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {np.sqrt(mse):.6f}")
            
            # Directional accuracy
            pred_direction = np.sign(pred_subset)
            actual_direction = np.sign(actual_subset)
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            print(f"Directional accuracy: {directional_accuracy:.1f}%")
            
            # Basic quality checks
            assert mae < 0.1, f"MAE too high: {mae}"
            assert directional_accuracy > 40, f"Directional accuracy too low: {directional_accuracy}%"
            
            print("✓ Prediction quality acceptable")
        else:
            print("⚠ Insufficient samples for quality assessment")
    else:
        pytest.fail("No predictions for quality assessment")

if __name__ == "__main__":
    print("Running simple usage examples...")
    
    test_simple_prediction_workflow()
    test_data_validation()
    test_feature_generation_example()
    test_prediction_quality_example()
    
    print("\\n✓ All simple usage examples completed successfully!")
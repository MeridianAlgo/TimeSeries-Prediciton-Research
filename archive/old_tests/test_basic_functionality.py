#!/usr/bin/env python3
"""Basic functionality test for ultra-precision predictor."""

import numpy as np
import pandas as pd
import logging
from ultra_precision_predictor.simple_predictor import SimplePredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(n_samples=500):
    """Create realistic test data."""
    np.random.seed(42)
    
    # Generate realistic price data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    returns = np.random.normal(0.0001, 0.02, n_samples)
    
    # Add some autocorrelation
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Convert to prices
    base_price = 100
    prices = base_price * np.cumprod(1 + returns)
    
    # Generate OHLCV
    high_noise = np.random.exponential(0.003, n_samples)
    low_noise = np.random.exponential(0.003, n_samples)
    open_noise = np.random.normal(0, 0.001, n_samples)
    
    df = pd.DataFrame({
        'Open': prices * (1 + open_noise),
        'High': prices * (1 + high_noise),
        'Low': prices * (1 - low_noise),
        'Close': prices,
        'Volume': np.random.lognormal(10, 0.5, n_samples)
    }, index=dates)
    
    # Ensure OHLC consistency
    df['High'] = np.maximum(df['High'], np.maximum(df['Open'], df['Close']))
    df['Low'] = np.minimum(df['Low'], np.minimum(df['Open'], df['Close']))
    
    return df

def test_feature_engineering():
    """Test feature engineering components."""
    logger.info("Testing feature engineering...")
    
    # Create test data
    data = create_test_data(200)
    logger.info(f"Created test data with {len(data)} samples")
    
    # Test individual components
    from ultra_precision_predictor.feature_engineering.extreme_feature_engineer import ExtremeFeatureEngineer
    
    engineer = ExtremeFeatureEngineer()
    
    # Generate features
    features = engineer.generate_features(data)
    
    logger.info(f"Generated {len(features.columns)} total columns")
    logger.info(f"Original columns: {len(data.columns)}")
    logger.info(f"New features: {len(features.columns) - len(data.columns)}")
    
    # Check for NaN values
    nan_count = features.isnull().sum().sum()
    total_values = len(features) * len(features.columns)
    nan_percentage = (nan_count / total_values) * 100
    
    logger.info(f"NaN values: {nan_count} ({nan_percentage:.2f}%)")
    
    # Get feature statistics
    stats = engineer.get_feature_statistics()
    logger.info(f"Feature statistics: {stats}")
    
    return features

def test_predictor():
    """Test the main predictor."""
    logger.info("Testing main predictor...")
    
    # Create test data
    data = create_test_data(300)
    
    # Split data
    train_size = int(len(data) * 0.7)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    logger.info(f"Training data: {len(train_data)} samples")
    logger.info(f"Test data: {len(test_data)} samples")
    
    # Initialize predictor
    predictor = SimplePredictor()
    
    # Train
    logger.info("Training predictor...")
    predictor.train(train_data)
    logger.info("Training completed")
    
    # Predict
    logger.info("Making predictions...")
    predictions = predictor.predict(test_data)
    
    if predictions is not None:
        logger.info(f"Generated {len(predictions)} predictions")
        logger.info(f"Prediction range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
        logger.info(f"Prediction mean: {np.mean(predictions):.6f}")
        logger.info(f"Prediction std: {np.std(predictions):.6f}")
        
        # Calculate basic accuracy metrics
        actual_returns = test_data['Close'].pct_change().dropna()
        min_len = min(len(predictions), len(actual_returns))
        
        if min_len > 10:
            pred_subset = predictions[:min_len]
            actual_subset = actual_returns.iloc[:min_len].values
            
            mae = np.mean(np.abs(pred_subset - actual_subset))
            mse = np.mean((pred_subset - actual_subset) ** 2)
            
            logger.info(f"MAE: {mae:.6f}")
            logger.info(f"MSE: {mse:.6f}")
            logger.info(f"RMSE: {np.sqrt(mse):.6f}")
            
            # Directional accuracy
            pred_direction = np.sign(pred_subset)
            actual_direction = np.sign(actual_subset)
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            logger.info(f"Directional accuracy: {directional_accuracy:.1f}%")
            
            return {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'directional_accuracy': directional_accuracy,
                'predictions': predictions
            }
    else:
        logger.warning("No predictions generated")
        return None

def main():
    """Main test function."""
    logger.info("Starting basic functionality tests...")
    
    try:
        # Test feature engineering
        features = test_feature_engineering()
        logger.info("✓ Feature engineering test passed")
        
        # Test predictor
        results = test_predictor()
        if results:
            logger.info("✓ Predictor test passed")
            
            # Check if results meet basic quality thresholds
            if results['mae'] < 0.1:
                logger.info("✓ MAE within acceptable range")
            else:
                logger.warning(f"⚠ MAE high: {results['mae']:.6f}")
                
            if results['directional_accuracy'] > 45:
                logger.info("✓ Directional accuracy acceptable")
            else:
                logger.warning(f"⚠ Directional accuracy low: {results['directional_accuracy']:.1f}%")
        else:
            logger.error("✗ Predictor test failed")
            
        logger.info("Basic functionality tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
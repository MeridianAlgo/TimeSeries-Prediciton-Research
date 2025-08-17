#!/usr/bin/env python3
"""
System Health Check for Enhanced Time Series Prediction System
=============================================================

This script tests the basic functionality of all major components
to ensure the system is working correctly.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import torch
        import sklearn
        print("✅ Basic imports successful")
        
        # Test enhanced_timeseries imports
        try:
            from enhanced_timeseries.ensemble.ensemble_framework import EnsembleFramework
            print("✅ EnsembleFramework import successful")
        except ImportError as e:
            print(f"⚠️  EnsembleFramework import failed: {e}")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_generation():
    """Test data generation and preprocessing."""
    print("\n📊 Testing data generation...")
    
    try:
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Create realistic financial data
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0, 1, n_samples))
        
        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices * (1 + np.random.normal(0, 0.01, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.02, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.02, n_samples))),
            'volume': np.random.lognormal(10, 0.5, n_samples)
        })
        
        print(f"✅ Generated {len(data)} samples of financial data")
        print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return data
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return None

def test_feature_engineering(data):
    """Test feature engineering capabilities."""
    print("\n🔧 Testing feature engineering...")
    
    try:
        # Create basic features
        features = pd.DataFrame()
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20]:
            features[f'sma_{window}'] = data['close'].rolling(window).mean()
            features[f'price_sma_ratio_{window}'] = data['close'] / features[f'sma_{window}']
        
        # Volatility
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Remove NaN values
        features = features.dropna()
        
        print(f"✅ Created {len(features.columns)} features")
        print(f"📊 Feature shape: {features.shape}")
        
        return features
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None

def test_model_training(features, target):
    """Test model training capabilities."""
    print("\n🧠 Testing model training...")
    
    try:
        # Ensure features and target have the same length
        min_length = min(len(features), len(target))
        features = features.iloc[:min_length]
        target = target.iloc[:min_length]
        
        X = features.values
        y = target.values
        
        print(f"📊 Total samples: {len(X)}")
        print(f"📊 Feature dimensions: {X.shape[1]}")
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"📊 Training samples: {len(X_train)}")
        print(f"📊 Test samples: {len(X_test)}")
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Train Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        models = {
            'Random Forest': rf_model,
            'Linear Regression': lr_model
        }
        
        results = []
        for name, model in models.items():
            if name == 'Random Forest':
                pred = rf_pred
            else:
                pred = lr_pred
                
            mse = mean_squared_error(y_test, pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            results.append({
                'model': name,
                'mse': mse,
                'mae': mae,
                'r2': r2
            })
            
            print(f"✅ {name}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        return models, results, X_test
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None, None, None

def test_ensemble_framework(models, X_test):
    """Test ensemble framework."""
    print("\n🎯 Testing ensemble framework...")
    
    try:
        from enhanced_timeseries.ensemble.ensemble_framework import EnsembleFramework
        
        # Create ensemble
        ensemble = EnsembleFramework()
        
        # Add models
        for name, model in models.items():
            ensemble.add_model(name, model)
        
        # Create dummy performance metrics
        performance_dict = {
            'Random Forest': {'rmse': 0.1, 'r2_score': 0.8},
            'Linear Regression': {'rmse': 0.15, 'r2_score': 0.6}
        }
        
        # Calculate weights
        weights = ensemble.calculate_weights(performance_dict)
        print(f"✅ Ensemble weights: {weights}")
        
        # Make predictions
        predictions = ensemble.predict_ensemble(X_test)
        print(f"✅ Ensemble predictions shape: {predictions['ensemble_prediction'].shape}")
        
        return ensemble
    except Exception as e:
        print(f"❌ Ensemble framework failed: {e}")
        return None

def test_main_application():
    """Test the main application."""
    print("\n🚀 Testing main application...")
    
    try:
        # Test if main.py can be imported and run
        import subprocess
        result = subprocess.run([sys.executable, 'main.py', '--symbol', 'AAPL', '--years', '1'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Main application runs successfully")
            return True
        else:
            print(f"⚠️  Main application had issues: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Main application test failed: {e}")
        return False

def main():
    """Run all system health checks."""
    print("🏥 ENHANCED TIME SERIES PREDICTION SYSTEM - HEALTH CHECK")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("❌ System health check failed at imports")
        return False
    
    # Test 2: Data generation
    data = test_data_generation()
    if data is None:
        print("❌ System health check failed at data generation")
        return False
    
    # Test 3: Feature engineering
    features = test_feature_engineering(data)
    if features is None:
        print("❌ System health check failed at feature engineering")
        return False
    
    # Test 4: Model training
    target = data['close'].shift(-1).dropna()  # Next day's close
    # Align features and target
    features = features.iloc[:-1]  # Remove last row since target is shifted
    target = target.iloc[:-1]      # Remove last row to match features
    
    models, results, X_test = test_model_training(features, target)
    if models is None:
        print("❌ System health check failed at model training")
        return False
    
    # Test 5: Ensemble framework
    ensemble = test_ensemble_framework(models, X_test)
    if ensemble is None:
        print("❌ System health check failed at ensemble framework")
        return False
    
    # Test 6: Main application (optional)
    print("\n📋 Summary:")
    print("✅ All core components working")
    print("✅ Data generation and preprocessing functional")
    print("✅ Feature engineering operational")
    print("✅ Model training successful")
    print("✅ Ensemble framework working")
    
    print("\n🎉 SYSTEM HEALTH CHECK PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

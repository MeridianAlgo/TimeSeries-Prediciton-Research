#!/usr/bin/env python3
"""Live historical simulation with maximum accuracy ML system."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from maximum_accuracy_predictor import MaximumAccuracyPredictor


def run_live_max_accuracy():
    """Run live simulation with maximum accuracy predictor."""
    print("🚀 Live Maximum Accuracy Simulation")
    print("=" * 40)
    
    # Load data
    from stock_predictor.data.fetcher import DataFetcher
    fetcher = DataFetcher()
    data = fetcher.fetch_stock_data_years_back('AAPL', years=2.0)
    
    if data is None or data.empty:
        print("❌ No data")
        return
    
    # Prepare data
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
    
    for old, new in {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}.items():
        if old in data.columns:
            data = data.rename(columns={old: new})
    
    data = data.sort_index()
    
    # Initialize predictor
    predictor = MaximumAccuracyPredictor()
    
    # Create features and train
    print("🧠 Training...")
    enhanced_data = predictor.create_ultimate_features(data)
    numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
    enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Split data
    split_idx = int(len(enhanced_data) * 0.7)
    train_data = enhanced_data.iloc[:split_idx]
    test_data = enhanced_data.iloc[split_idx:]
    
    X_train = train_data[feature_cols].values
    y_train = train_data['close'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['close'].values
    
    # Train models
    predictor.train_maximum_accuracy_models(X_train, y_train)
    print("✅ Training complete")
    
    # Setup live simulation
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    dates = test_data.index
    actual_prices = []
    predicted_prices = []
    errors = []
    
    print("📈 Starting live simulation...")
    
    # Simulate live predictions
    for i in range(0, len(X_test), 10):  # Every 10 points
        end_idx = min(i + 10, len(X_test))
        X_batch = X_test[i:end_idx]
        y_batch = y_test[i:end_idx]
        
        # Make predictions
        pred_batch, _ = predictor.predict_maximum_accuracy(X_batch)
        
        # Update data
        actual_prices.extend(y_batch)
        predicted_prices.extend(pred_batch)
        
        # Calculate errors
        batch_errors = np.abs((y_batch - pred_batch) / y_batch) * 100
        errors.extend(batch_errors)
        
        # Update plots
        ax1.clear()
        ax2.clear()
        
        # Ensure arrays are same length
        min_len = min(len(actual_prices), len(predicted_prices), len(errors))
        current_dates = dates[:min_len]
        current_actual = actual_prices[:min_len]
        current_pred = predicted_prices[:min_len]
        current_errors = errors[:min_len]
        
        # Price plot
        ax1.plot(current_dates, current_actual, 'k-', label='Actual', linewidth=2)
        ax1.plot(current_dates, current_pred, 'r--', label='Predicted', linewidth=2)
        ax1.set_title('Live Stock Price Prediction')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        accuracy_2pct = np.mean(np.array(current_errors) <= 2.0) * 100
        ax2.plot(current_dates, current_errors, 'b-', alpha=0.7)
        ax2.axhline(y=2.0, color='r', linestyle='--', alpha=0.8, label='2% threshold')
        ax2.set_title(f'Prediction Error - Accuracy: {accuracy_2pct:.1f}%')
        ax2.set_ylabel('Error (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.5)
        
        # Print status
        if len(current_errors) > 0:
            current_error = np.mean(current_errors[-10:]) if len(current_errors) >= 10 else np.mean(current_errors)
            print(f"📊 Batch {i//10+1}: Accuracy {accuracy_2pct:.1f}%, Error {current_error:.2f}%")
    
    # Final results
    final_accuracy = np.mean(np.array(errors) <= 2.0) * 100
    final_rmse = np.sqrt(np.mean((np.array(actual_prices) - np.array(predicted_prices)) ** 2))
    
    print(f"\n🎉 Final Results:")
    print(f"📊 Accuracy (±2%): {final_accuracy:.1f}%")
    print(f"📉 RMSE: ${final_rmse:.3f}")
    print(f"📈 Mean Error: {np.mean(errors):.2f}%")
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_live_max_accuracy()
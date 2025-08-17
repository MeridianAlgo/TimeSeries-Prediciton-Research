#!/usr/bin/env python3
"""Simple live stock prediction - standalone program."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


def fetch_stock_data(symbol='AAPL', period='2y'):
    """Fetch stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        return data
    except:
        print("❌ Error fetching data")
        return None


def create_features(data):
    """Create simple technical features."""
    df = data.copy()
    
    # Price features
    df['returns'] = df['Close'].pct_change()
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['price_sma_ratio'] = df['Close'] / df['sma_20']
    
    # Volume features
    df['volume_sma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    
    # Volatility
    df['volatility'] = df['returns'].rolling(10).std()
    
    # Momentum
    df['momentum'] = df['Close'] / df['Close'].shift(5)
    
    # Remove NaN
    df = df.dropna()
    
    return df


def train_model(data):
    """Train simple prediction model."""
    feature_cols = ['returns', 'sma_5', 'sma_20', 'price_sma_ratio', 
                   'volume_ratio', 'volatility', 'momentum']
    
    X = data[feature_cols].values
    y = data['Close'].values
    
    # Split data
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    predictions = model.predict(X_test_scaled)
    
    return model, scaler, X_test, y_test, predictions


def run_live_simulation():
    """Run live prediction simulation."""
    print("🚀 Simple Live Stock Prediction")
    print("=" * 35)
    
    # Fetch data
    print("📊 Fetching data...")
    data = fetch_stock_data()
    if data is None:
        return
    
    # Create features
    print("🔧 Creating features...")
    enhanced_data = create_features(data)
    
    # Train model
    print("🧠 Training model...")
    model, scaler, X_test, y_test, predictions = train_model(enhanced_data)
    
    # Calculate accuracy
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    errors = np.abs((y_test - predictions) / y_test) * 100
    accuracy = np.mean(errors <= 2.0) * 100
    
    print(f"✅ Training complete")
    print(f"📊 Accuracy (±2%): {accuracy:.1f}%")
    print(f"📉 RMSE: ${rmse:.2f}")
    print(f"📈 Mean Error: {np.mean(errors):.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    dates = enhanced_data.index[-len(y_test):]
    plt.plot(dates, y_test, 'k-', label='Actual', linewidth=2)
    plt.plot(dates, predictions, 'r--', label='Predicted', linewidth=2)
    
    plt.title('Live Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return accuracy, rmse


if __name__ == "__main__":
    run_live_simulation()
#!/usr/bin/env python3
"""
Simple Demonstration of Enhanced PyTorch Time Series Prediction System
Shows prediction accuracy and system capabilities
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def generate_realistic_financial_data(n_days=1000, start_price=100):
    """Generate realistic financial time series data."""
    
    print("📊 Generating realistic financial data...")
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate realistic price movements with trends, volatility clustering
    np.random.seed(42)
    
    # Base returns with slight positive drift
    base_returns = np.random.normal(0.0005, 0.015, n_days)
    
    # Add volatility clustering (GARCH-like behavior)
    volatility = np.zeros(n_days)
    volatility[0] = 0.015
    for i in range(1, n_days):
        volatility[i] = 0.1 + 0.8 * volatility[i-1] + 0.1 * abs(base_returns[i-1])
    
    # Apply volatility clustering
    returns = base_returns * np.sqrt(volatility)
    
    # Add market regime changes
    regime_changes = [200, 400, 600, 800]
    for change_point in regime_changes:
        if change_point < n_days:
            returns[change_point:change_point+50] *= 1.5
    
    # Generate prices from returns
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.002, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_days)
    })
    
    print(f"✅ Generated {len(data)} days of financial data")
    print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"📊 Average daily volume: {data['volume'].mean():.0f}")
    
    return data

def create_technical_features(data):
    """Create technical indicators as features."""
    
    print("\n🔧 Creating technical indicators...")
    
    features = pd.DataFrame()
    
    # Price-based features
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        features[f'sma_{window}'] = data['close'].rolling(window).mean()
        features[f'ema_{window}'] = data['close'].ewm(span=window).mean()
    
    # Volatility features
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = features['returns'].rolling(window).std()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = data['close'].ewm(span=12).mean()
    ema26 = data['close'].ewm(span=26).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    for window in [20]:
        sma = data['close'].rolling(window).mean()
        std = data['close'].rolling(window).std()
        features[f'bb_upper_{window}'] = sma + (std * 2)
        features[f'bb_lower_{window}'] = sma - (std * 2)
        features[f'bb_width_{window}'] = features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']
    
    # Volume features
    features['volume_sma'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_sma']
    
    # Price position features
    features['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Drop NaN values
    features = features.dropna()
    
    print(f"✅ Created {len(features.columns)} technical indicators")
    
    return features

def prepare_sequences(data, features, target_col='close', sequence_length=60, prediction_horizon=5):
    """Prepare sequences for time series prediction."""
    
    print(f"\n📈 Preparing sequences (length={sequence_length}, horizon={prediction_horizon})...")
    
    # Combine data and features
    all_data = pd.concat([data, features], axis=1).dropna()
    
    # Select feature columns (exclude date and target)
    feature_cols = [col for col in all_data.columns if col not in ['date', target_col]]
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_data[feature_cols])
    
    # Create sequences
    X, y = [], []
    
    for i in range(sequence_length, len(all_data) - prediction_horizon + 1):
        # Input sequence
        X.append(features_scaled[i-sequence_length:i])
        
        # Target: future price change (percentage)
        current_price = all_data[target_col].iloc[i-1]
        future_price = all_data[target_col].iloc[i+prediction_horizon-1]
        price_change_pct = (future_price - current_price) / current_price
        
        y.append(price_change_pct)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"📊 Prepared {len(X)} sequences")
    print(f"📈 Input shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"🔧 Number of features: {len(feature_cols)}")
    
    return X, y, scaler, feature_cols

class SimpleLSTM(nn.Module):
    """Simple LSTM model for demonstration."""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

def train_lstm_model(X_train, y_train, X_val, y_val, input_dim, epochs=50):
    """Train LSTM model."""
    
    print("\n🧠 Training LSTM model...")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create model
    model = SimpleLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        train_loss = criterion(outputs, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    return model, train_losses, val_losses

def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluate prediction accuracy with multiple metrics."""
    
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.numpy().flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    
    # Directional accuracy
    directional_correct = np.mean((y_true > 0) == (y_pred > 0))
    
    # Correlation
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # R-squared
    r_squared = r2_score(y_true, y_pred)
    
    return {
        'model': model_name,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'directional_accuracy': directional_correct * 100,
        'correlation': correlation,
        'r_squared': r_squared
    }

def plot_results(results_df, y_true, y_pred, model_name):
    """Plot prediction results."""
    
    print(f"\n📊 Generating visualization for {model_name}...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Prediction Results', fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price Change (%)')
    axes[0, 0].set_ylabel('Predicted Price Change (%)')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Time series of predictions
    axes[0, 1].plot(y_true[:100], label='Actual', linewidth=1, color='green')
    axes[0, 1].plot(y_pred[:100], label='Predicted', linewidth=1, color='red')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Price Change (%)')
    axes[0, 1].set_title('Time Series Predictions (First 100)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Prediction errors
    errors = y_true - y_pred
    axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Prediction Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics
    metrics_text = f"""PERFORMANCE METRICS:
R² Score: {results_df['r_squared']:.4f}
Directional Accuracy: {results_df['directional_accuracy']:.2f}%
Correlation: {results_df['correlation']:.4f}
MAE: {results_df['mae']:.4f}
RMSE: {results_df['rmse']:.4f}"""
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                   fontsize=12, verticalalignment='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    axes[1, 1].set_title('Performance Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Results plot saved as '{model_name.lower()}_results.png'")

def main():
    """Main demonstration function."""
    
    print("🚀 Enhanced PyTorch Time Series Prediction System")
    print("=" * 60)
    print("Complete Prediction Accuracy Demonstration")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 1. Generate realistic financial data
    financial_data = generate_realistic_financial_data(n_days=1000)
    
    # 2. Create technical features
    features = create_technical_features(financial_data)
    
    # 3. Prepare sequences
    sequence_length = 60
    prediction_horizon = 5
    X, y, scaler, feature_cols = prepare_sequences(
        financial_data, features,
        target_col='close', 
        sequence_length=sequence_length, 
        prediction_horizon=prediction_horizon
    )
    
    # 4. Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # 5. Train LSTM model
    input_dim = X_train.shape[2]
    lstm_model, train_losses, val_losses = train_lstm_model(
        X_train, y_train, X_val, y_val, input_dim, epochs=50
    )
    
    # 6. Make predictions with LSTM
    lstm_model.eval()
    with torch.no_grad():
        lstm_predictions = lstm_model(torch.FloatTensor(X_test))
    
    # 7. Train linear model for comparison
    print("\n📊 Training linear model for comparison...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    linear_model = LinearRegression()
    linear_model.fit(X_train_flat, y_train)
    linear_predictions = linear_model.predict(X_test_flat)
    
    # 8. Evaluate both models
    print("\n🎯 Evaluating prediction accuracy...")
    
    # LSTM results
    lstm_results = evaluate_predictions(y_test, lstm_predictions, 'LSTM')
    
    # Linear model results
    linear_results = evaluate_predictions(y_test, linear_predictions, 'Linear')
    
    # Create results DataFrame
    results_df = pd.DataFrame([lstm_results, linear_results])
    
    # 9. Print results
    print("\n" + "="*80)
    print("🎯 PREDICTION ACCURACY RESULTS")
    print("="*80)
    print(results_df.round(4))
    
    # Find best model
    best_model = results_df.loc[results_df['r_squared'].idxmax()]
    print(f"\n🏆 BEST MODEL: {best_model['model']}")
    print(f"   R² Score: {best_model['r_squared']:.4f}")
    print(f"   Directional Accuracy: {best_model['directional_accuracy']:.2f}%")
    print(f"   Correlation: {best_model['correlation']:.4f}")
    print(f"   MAE: {best_model['mae']:.4f}")
    print(f"   RMSE: {best_model['rmse']:.4f}")
    
    # 10. Plot results
    plot_results(lstm_results, y_test, lstm_predictions.numpy().flatten(), 'LSTM')
    plot_results(linear_results, y_test, linear_predictions, 'Linear')
    
    # 11. Performance assessment
    print("\n🎯 PREDICTION QUALITY ASSESSMENT:")
    if best_model['r_squared'] > 0.7:
        print("   🏆 EXCELLENT: R² > 0.7 indicates strong predictive power")
    elif best_model['r_squared'] > 0.5:
        print("   ✅ GOOD: R² > 0.5 indicates good predictive power")
    elif best_model['r_squared'] > 0.3:
        print("   ⚠️  FAIR: R² > 0.3 indicates moderate predictive power")
    else:
        print("   ❌ POOR: R² < 0.3 indicates weak predictive power")
    
    if best_model['directional_accuracy'] > 65:
        print("   🎯 EXCELLENT: Directional accuracy > 65% indicates strong trend prediction")
    elif best_model['directional_accuracy'] > 55:
        print("   ✅ GOOD: Directional accuracy > 55% indicates good trend prediction")
    else:
        print("   ⚠️  FAIR: Directional accuracy < 55% indicates weak trend prediction")
    
    print("\n" + "="*80)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*80)
    print("✅ Realistic Financial Data Generation")
    print("✅ Advanced Technical Indicators (15+ features)")
    print("✅ LSTM Neural Network Model")
    print("✅ Linear Model Comparison")
    print("✅ Comprehensive Accuracy Evaluation")
    print("✅ Visualization and Performance Analysis")
    print("\n🎊 SYSTEM CAPABILITIES DEMONSTRATED!")
    print("="*80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ultra-Precision Time Series Predictor
====================================

Advanced PyTorch-based time series prediction system for financial markets.
Tests on historical data with bar-by-bar prediction accuracy tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesTransformer(nn.Module):
    """Advanced Transformer model for time series prediction"""
    
    def __init__(self, input_dim=5, d_model=128, nhead=8, num_layers=6, seq_len=60, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._generate_positional_encoding(seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.output = nn.Linear(d_model // 4, 1)
        
        # Activation
        self.relu = nn.ReLU()
        
    def _generate_positional_encoding(self, seq_len, d_model):
        """Generate positional encoding for transformer"""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use the last time step for prediction
        x = x[:, -1, :]
        
        # Output layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.output(x)
        
        return x

class UltraPrecisionPredictor:
    """Ultra-precision time series predictor using PyTorch"""
    
    def __init__(self, seq_len=60, device=None):
        self.seq_len = seq_len
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        
        print(f"🔥 Using device: {self.device}")
        
    def create_features(self, data):
        """Create advanced technical features"""
        df = data.copy()
        
        # Basic features
        returns = df['Close'].pct_change().fillna(0)
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)
        
        # Moving averages and ratios
        sma_5 = df['Close'].rolling(5).mean()
        sma_10 = df['Close'].rolling(10).mean()
        sma_20 = df['Close'].rolling(20).mean()
        
        price_to_sma_5 = (df['Close'] / sma_5 - 1).fillna(0)
        price_to_sma_10 = (df['Close'] / sma_10 - 1).fillna(0)
        price_to_sma_20 = (df['Close'] / sma_20 - 1).fillna(0)
        
        # Volatility features
        volatility_5 = returns.rolling(5).std().fillna(0)
        volatility_20 = returns.rolling(20).std().fillna(0)
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = (100 - (100 / (1 + rs))).fillna(50)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        macd = (exp1 - exp2).fillna(0)
        macd_signal = macd.ewm(span=9).mean().fillna(0)
        macd_histogram = (macd - macd_signal).fillna(0)
        
        # Bollinger Bands
        bb_middle = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_position = ((df['Close'] - bb_lower) / (bb_upper - bb_lower)).fillna(0.5)
        
        # Price position features
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        price_position = ((df['Close'] - low_20) / (high_20 - low_20)).fillna(0.5)
        
        # Volume features
        volume_sma = df['Volume'].rolling(20).mean()
        volume_ratio = (df['Volume'] / volume_sma).fillna(1)
        
        # Momentum features
        momentum_5 = (df['Close'] / df['Close'].shift(5) - 1).fillna(0)
        momentum_10 = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
        
        # Combine all features
        features = np.column_stack([
            returns.values,
            log_returns.values,
            volatility_5.values,
            volatility_20.values,
            rsi.values / 100,  # Normalize RSI to 0-1
            macd.values,
            macd_histogram.values,
            bb_position.values,
            price_position.values,
            volume_ratio.values,
            momentum_5.values,
            momentum_10.values,
            price_to_sma_5.values,
            price_to_sma_10.values,
            price_to_sma_20.values
        ])
        
        # Replace any remaining NaN or inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features
    
    def prepare_sequences(self, features, targets):
        """Prepare sequences for training"""
        X, y = [], []
        
        for i in range(self.seq_len, len(features)):
            X.append(features[i-self.seq_len:i])
            y.append(targets[i])
        
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the ultra-precision model"""
        print("🚀 Training Ultra-Precision Time Series Predictor...")
        
        # Create features
        features = self.create_features(data)
        
        # Prepare targets (next period return)
        targets = data['Close'].pct_change().shift(-1).values[:-1]
        features = features[:-1]  # Remove last row to match targets
        
        # Scale features and targets
        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled = self.scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.prepare_sequences(features_scaled, targets_scaled)
        
        print(f"📊 Training data shape: {X.shape}, Target shape: {y.shape}")
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.FloatTensor(y_test).to(self.device)
        
        # Create model
        input_dim = X_train.shape[2]
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            seq_len=self.seq_len,
            dropout=0.1
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_test).squeeze()
                val_loss = criterion(val_outputs, y_test).item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {total_loss/len(X_train)*batch_size:.6f}, "
                      f"Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            if patience_counter >= 20:
                print("Early stopping triggered")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.is_trained = True
        
        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_train).squeeze().cpu().numpy()
            test_pred = self.model(X_test).squeeze().cpu().numpy()
        
        # Inverse transform predictions
        train_pred_orig = self.scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
        test_pred_orig = self.scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        y_train_orig = self.scaler.inverse_transform(y_train.cpu().numpy().reshape(-1, 1)).flatten()
        y_test_orig = self.scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train_orig, train_pred_orig)
        test_mae = mean_absolute_error(y_test_orig, test_pred_orig)
        train_rmse = np.sqrt(mean_squared_error(y_train_orig, train_pred_orig))
        test_rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred_orig))
        
        # Directional accuracy
        train_dir_acc = np.mean(np.sign(y_train_orig) == np.sign(train_pred_orig)) * 100
        test_dir_acc = np.mean(np.sign(y_test_orig) == np.sign(test_pred_orig)) * 100
        
        print(f"\n📊 Training Results:")
        print(f"   Train MAE: {train_mae:.6f} ({train_mae*100:.3f}%)")
        print(f"   Test MAE:  {test_mae:.6f} ({test_mae*100:.3f}%)")
        print(f"   Train RMSE: {train_rmse:.6f}")
        print(f"   Test RMSE:  {test_rmse:.6f}")
        print(f"   Train Directional Accuracy: {train_dir_acc:.1f}%")
        print(f"   Test Directional Accuracy:  {test_dir_acc:.1f}%")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_dir_acc': train_dir_acc,
            'test_dir_acc': test_dir_acc
        }
    
    def predict_next(self, data):
        """Predict next period return"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Create features for the latest data
        features = self.create_features(data)
        
        # Scale features
        features_scaled = self.feature_scaler.transform(features)
        
        # Get last sequence
        if len(features_scaled) < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} data points for prediction")
        
        last_sequence = features_scaled[-self.seq_len:]
        
        # Convert to tensor
        X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction_scaled = self.model(X).squeeze().cpu().numpy()
        
        # Inverse transform
        prediction = self.scaler.inverse_transform([[prediction_scaled]])[0][0]
        
        return prediction

def download_data(symbol='AAPL', period='2y'):
    """Download historical data"""
    print(f"📥 Downloading {symbol} data for {period}...")
    data = yf.download(symbol, period=period, progress=False)
    print(f"✅ Downloaded {len(data)} data points")
    return data

def test_historical_predictions(predictor, data, test_days=50):
    """Test predictions on historical data bar by bar"""
    print(f"\n🎯 Testing Historical Predictions (Last {test_days} days)")
    print("=" * 80)
    
    # Use data up to test_days ago for initial training
    train_data = data[:-test_days].copy()
    test_data = data[-test_days:].copy()
    
    predictions = []
    actual_returns = []
    errors = []
    
    for i in range(len(test_data)):
        # Current training data (up to current point)
        current_data = pd.concat([train_data, test_data[:i]]) if i > 0 else train_data
        
        try:
            # Make prediction for next bar
            predicted_return = predictor.predict_next(current_data)
            
            # Get actual return
            if i < len(test_data) - 1:
                current_price = test_data.iloc[i]['Close']
                next_price = test_data.iloc[i + 1]['Close']
                actual_return = (next_price - current_price) / current_price
            else:
                actual_return = 0  # Last bar, no next price
            
            # Calculate error
            error = abs(predicted_return - actual_return)
            
            predictions.append(predicted_return)
            actual_returns.append(actual_return)
            errors.append(error)
            
            # Print results
            date = test_data.index[i].strftime('%Y-%m-%d')
            current_price = test_data.iloc[i]['Close']
            
            if i < len(test_data) - 1:
                next_price = test_data.iloc[i + 1]['Close']
                direction_correct = "✅" if np.sign(predicted_return) == np.sign(actual_return) else "❌"
                
                print(f"📅 {date} | Price: ${current_price:7.2f} → ${next_price:7.2f}")
                print(f"   🔮 Predicted: {predicted_return:+8.4f} ({predicted_return*100:+6.2f}%)")
                print(f"   📊 Actual:    {actual_return:+8.4f} ({actual_return*100:+6.2f}%)")
                print(f"   📏 Error:     {error:8.4f} ({error*100:6.2f}%) {direction_correct}")
                print()
            else:
                print(f"📅 {date} | Price: ${current_price:7.2f}")
                print(f"   🔮 Predicted: {predicted_return:+8.4f} ({predicted_return*100:+6.2f}%)")
                print(f"   📊 Actual:    [Next bar not available]")
                print()
                
        except Exception as e:
            print(f"❌ Error predicting for {date}: {e}")
            continue
    
    # Calculate overall statistics
    if len(errors) > 1:
        avg_error = np.mean(errors[:-1])  # Exclude last prediction (no actual)
        rmse = np.sqrt(np.mean(np.array(errors[:-1])**2))
        
        # Directional accuracy
        correct_directions = sum(1 for p, a in zip(predictions[:-1], actual_returns[:-1]) 
                               if np.sign(p) == np.sign(a))
        directional_accuracy = (correct_directions / len(predictions[:-1])) * 100
        
        print("=" * 80)
        print("📊 PREDICTION PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"📏 Average Error (MAE):     {avg_error:.6f} ({avg_error*100:.3f}%)")
        print(f"📏 Root Mean Square Error:  {rmse:.6f} ({rmse*100:.3f}%)")
        print(f"🎯 Directional Accuracy:    {directional_accuracy:.1f}%")
        print(f"📊 Total Predictions:       {len(predictions)-1}")
        print(f"✅ Correct Directions:      {correct_directions}")
        print(f"❌ Wrong Directions:        {len(predictions)-1-correct_directions}")
        
        return {
            'avg_error': avg_error,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions,
            'actual_returns': actual_returns,
            'errors': errors
        }

def main():
    """Main execution function"""
    print("🚀 Ultra-Precision Time Series Predictor")
    print("📦 Using PyTorch for Advanced Neural Networks")
    print("=" * 60)
    
    # Download data
    symbol = 'AAPL'  # You can change this to any stock symbol
    data = download_data(symbol, period='2y')
    
    # Initialize predictor
    predictor = UltraPrecisionPredictor(seq_len=60)
    
    # Train model on historical data (excluding last 50 days for testing)
    train_data = data[:-50].copy()
    metrics = predictor.train(train_data, epochs=50, batch_size=32)
    
    # Test on recent historical data
    results = test_historical_predictions(predictor, data, test_days=50)
    
    print("\n✨ Ultra-Precision Time Series Prediction Complete!")
    print(f"🎯 Achieved {results['directional_accuracy']:.1f}% directional accuracy")
    print(f"📏 Average prediction error: {results['avg_error']*100:.3f}%")

if __name__ == "__main__":
    main()
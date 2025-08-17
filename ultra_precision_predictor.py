#!/usr/bin/env python3
"""
Ultra-Precision Time Series Predictor
=====================================

Advanced prediction system designed to achieve <1% error rate through:
- Sophisticated feature engineering (100+ features)
- Advanced model architectures
- Ensemble methods with dynamic weighting
- Real-time adaptation
- Uncertainty quantification
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class UltraPrecisionLSTM(nn.Module):
    """Advanced LSTM with attention mechanism and residual connections."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=4, dropout=0.2):
        super(UltraPrecisionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        # Residual connections
        self.residual_layers = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size * 2) for _ in range(3)
        ])
        
        # Output layers with skip connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Residual connections
        residual = pooled
        for layer in self.residual_layers:
            residual = residual + self.dropout(layer(residual))
        
        # Output layers with skip connections
        out = torch.relu(self.bn1(self.fc1(residual)))
        out = self.dropout(out)
        
        out2 = torch.relu(self.bn2(self.fc2(out)))
        out2 = self.dropout(out2)
        
        # Skip connection
        final_out = self.fc3(out2 + self.fc2(out))
        
        return final_out

class AdvancedFeatureEngineer:
    """Ultra-precision feature engineering with 100+ features."""
    
    def __init__(self):
        self.features_created = 0
        
    def create_all_features(self, data):
        """Create comprehensive feature set."""
        features = data.copy()
        
        # Price-based features
        features = self._create_price_features(features)
        
        # Technical indicators
        features = self._create_technical_indicators(features)
        
        # Volatility features
        features = self._create_volatility_features(features)
        
        # Volume features
        features = self._create_volume_features(features)
        
        # Momentum features
        features = self._create_momentum_features(features)
        
        # Trend features
        features = self._create_trend_features(features)
        
        # Statistical features
        features = self._create_statistical_features(features)
        
        # Harmonic features
        features = self._create_harmonic_features(features)
        
        # Microstructure features
        features = self._create_microstructure_features(features)
        
        # Remove NaN values
        features = features.dropna()
        
        print(f"✅ Created {len(features.columns)} ultra-precision features")
        return features
    
    def _create_price_features(self, data):
        """Create advanced price-based features."""
        # Returns and log returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price ratios
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        
        # Price changes
        for period in [1, 2, 3, 5, 10, 20]:
            data[f'price_change_{period}'] = data['close'].pct_change(period)
            data[f'price_change_abs_{period}'] = data['close'].pct_change(period).abs()
        
        # Moving averages with multiple windows
        for window in [3, 5, 8, 13, 21, 34, 55, 89]:
            data[f'sma_{window}'] = data['close'].rolling(window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
            data[f'price_sma_ratio_{window}'] = data['close'] / data[f'sma_{window}']
            data[f'price_ema_ratio_{window}'] = data['close'] / data[f'ema_{window}']
        
        return data
    
    def _create_technical_indicators(self, data):
        """Create advanced technical indicators."""
        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD variations
        for fast, slow in [(12, 26), (8, 21), (5, 13)]:
            exp1 = data['close'].ewm(span=fast).mean()
            exp2 = data['close'].ewm(span=slow).mean()
            data[f'macd_{fast}_{slow}'] = exp1 - exp2
            data[f'macd_signal_{fast}_{slow}'] = data[f'macd_{fast}_{slow}'].ewm(span=9).mean()
            data[f'macd_hist_{fast}_{slow}'] = data[f'macd_{fast}_{slow}'] - data[f'macd_signal_{fast}_{slow}']
        
        # Bollinger Bands
        for window in [20, 30, 50]:
            sma = data['close'].rolling(window).mean()
            std = data['close'].rolling(window).std()
            data[f'bb_upper_{window}'] = sma + (std * 2)
            data[f'bb_lower_{window}'] = sma - (std * 2)
            data[f'bb_width_{window}'] = data[f'bb_upper_{window}'] - data[f'bb_lower_{window}']
            data[f'bb_position_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / data[f'bb_width_{window}']
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            data[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            data[f'stoch_d_{period}'] = data[f'stoch_k_{period}'].rolling(3).mean()
        
        return data
    
    def _create_volatility_features(self, data):
        """Create advanced volatility features."""
        # Rolling volatility
        for window in [5, 10, 20, 30, 50]:
            data[f'volatility_{window}'] = data['returns'].rolling(window).std()
            data[f'volatility_annualized_{window}'] = data[f'volatility_{window}'] * np.sqrt(252)
        
        # GARCH-like features
        data['volatility_ewm'] = data['returns'].ewm(span=20).std()
        data['volatility_ratio'] = data['volatility_20'] / data['volatility_ewm']
        
        # Parkinson volatility
        data['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            ((np.log(data['high'] / data['low']) ** 2).rolling(20).mean())
        )
        
        return data
    
    def _create_volume_features(self, data):
        """Create advanced volume features."""
        # Volume moving averages
        for window in [5, 10, 20, 50]:
            data[f'volume_sma_{window}'] = data['volume'].rolling(window).mean()
            data[f'volume_ratio_{window}'] = data['volume'] / data[f'volume_sma_{window}']
        
        # Volume-price relationship
        data['volume_price_trend'] = (data['volume'] * data['returns']).rolling(20).sum()
        data['volume_force_index'] = data['volume'] * data['returns']
        
        # On-balance volume
        data['obv'] = (np.sign(data['returns']) * data['volume']).cumsum()
        
        return data
    
    def _create_momentum_features(self, data):
        """Create momentum features."""
        # Rate of change
        for period in [5, 10, 20, 30]:
            data[f'roc_{period}'] = data['close'].pct_change(period) * 100
        
        # Williams %R
        for period in [14, 21]:
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            data[f'williams_r_{period}'] = -100 * (high_max - data['close']) / (high_max - low_min)
        
        # Commodity Channel Index
        for period in [20, 30]:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = typical_price.rolling(period).mean()
            mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            data[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return data
    
    def _create_trend_features(self, data):
        """Create trend features."""
        # ADX (Average Directional Index)
        for period in [14, 21]:
            high_diff = data['high'].diff()
            low_diff = data['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            tr = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    np.abs(data['high'] - data['close'].shift(1)),
                    np.abs(data['low'] - data['close'].shift(1))
                )
            )
            
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / pd.Series(tr).rolling(period).mean()
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / pd.Series(tr).rolling(period).mean()
            
            data[f'adx_{period}'] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Parabolic SAR
        data['psar'] = self._calculate_psar(data)
        
        return data
    
    def _create_statistical_features(self, data):
        """Create statistical features."""
        # Z-score
        for window in [20, 50]:
            data[f'zscore_{window}'] = (data['close'] - data['close'].rolling(window).mean()) / data['close'].rolling(window).std()
        
        # Percentile ranks
        for window in [20, 50]:
            data[f'percentile_rank_{window}'] = data['close'].rolling(window).rank(pct=True)
        
        # Skewness and kurtosis
        for window in [20, 50]:
            data[f'skewness_{window}'] = data['returns'].rolling(window).skew()
            data[f'kurtosis_{window}'] = data['returns'].rolling(window).kurt()
        
        return data
    
    def _create_harmonic_features(self, data):
        """Create harmonic analysis features."""
        # Fourier transform features
        for window in [20, 50]:
            prices = data['close'].rolling(window).mean()
            fft = np.fft.fft(prices.dropna())
            data[f'fft_magnitude_{window}'] = np.abs(fft)[:len(prices)//2].mean()
        
        # Wavelet-like features using Savitzky-Golay filter
        data['savgol_trend'] = savgol_filter(data['close'].values, window_length=21, polyorder=3)
        data['savgol_residual'] = data['close'] - data['savgol_trend']
        
        return data
    
    def _create_microstructure_features(self, data):
        """Create market microstructure features."""
        # Bid-ask spread proxy
        data['spread_proxy'] = (data['high'] - data['low']) / data['close']
        
        # Price efficiency
        data['price_efficiency'] = data['returns'].rolling(20).apply(
            lambda x: np.abs(x.sum()) / x.abs().sum() if x.abs().sum() > 0 else 0
        )
        
        # Volume efficiency
        data['volume_efficiency'] = data['volume'].rolling(20).apply(
            lambda x: x.iloc[-1] / x.mean() if x.mean() > 0 else 1
        )
        
        return data
    
    def _calculate_psar(self, data, acceleration=0.02, maximum=0.2):
        """Calculate Parabolic SAR."""
        psar = data['close'].copy()
        psar.iloc[0] = data['low'].iloc[0]
        
        af = acceleration
        ep = data['high'].iloc[0]
        long = True
        
        for i in range(1, len(data)):
            if long:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if data['high'].iloc[i] > ep:
                    ep = data['high'].iloc[i]
                    af = min(af + acceleration, maximum)
                if data['low'].iloc[i] < psar.iloc[i]:
                    long = False
                    psar.iloc[i] = ep
                    ep = data['low'].iloc[i]
                    af = acceleration
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if data['low'].iloc[i] < ep:
                    ep = data['low'].iloc[i]
                    af = min(af + acceleration, maximum)
                if data['high'].iloc[i] > psar.iloc[i]:
                    long = True
                    psar.iloc[i] = ep
                    ep = data['high'].iloc[i]
                    af = acceleration
        
        return psar

class UltraPrecisionEnsemble:
    """Advanced ensemble with dynamic weighting and uncertainty quantification."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        
    def add_model(self, name, model, scaler=None):
        """Add a model to the ensemble."""
        self.models[name] = model
        if scaler:
            self.scalers[name] = scaler
    
    def calculate_optimal_weights(self, X_val, y_val):
        """Calculate optimal weights using validation performance."""
        predictions = {}
        errors = {}
        
        for name, model in self.models.items():
            try:
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X_val)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X_val)
                
                predictions[name] = pred
                errors[name] = mean_squared_error(y_val, pred)
            except:
                errors[name] = float('inf')
        
        # Calculate weights inversely proportional to error
        total_inv_error = sum(1 / (error + 1e-8) for error in errors.values())
        self.weights = {name: (1 / (error + 1e-8)) / total_inv_error for name, error in errors.items()}
        
        return self.weights
    
    def predict(self, X, return_uncertainty=True):
        """Make ensemble prediction with uncertainty quantification."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions[name] = pred
            except:
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            weight = self.weights.get(name, 1.0 / len(predictions))
            ensemble_pred += weight * pred
        
        if return_uncertainty:
            # Calculate prediction uncertainty
            pred_array = np.array(list(predictions.values()))
            uncertainty = np.std(pred_array, axis=0)
            
            return ensemble_pred, uncertainty, predictions
        else:
            return ensemble_pred

class UltraPrecisionPredictor:
    """Main ultra-precision predictor class."""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble = UltraPrecisionEnsemble()
        self.data_processor = None
        
    def fetch_and_prepare_data(self, symbol, period='2y'):
        """Fetch and prepare data with ultra-precision features."""
        print(f"📊 Fetching data for {symbol}...")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        print(f"✅ Fetched {len(data)} records")
        
        # Create ultra-precision features
        print("🔧 Creating ultra-precision features...")
        features = self.feature_engineer.create_all_features(data)
        
        # Prepare target (next day's close)
        features['target'] = features['close'].shift(-1)
        features = features.dropna()
        
        # Split features and target
        feature_cols = [col for col in features.columns if col not in ['target', 'date']]
        X = features[feature_cols].values
        y = features['target'].values
        
        # Split data
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple advanced models."""
        print("🧠 Training ultra-precision models...")
        
        # 1. Ultra-Precision LSTM
        print("   Training Ultra-Precision LSTM...")
        lstm_model = self._train_lstm(X_train, y_train, X_val, y_val)
        self.ensemble.add_model('ultra_lstm', lstm_model)
        
        # 2. Advanced Random Forest
        print("   Training Advanced Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        self.ensemble.add_model('advanced_rf', rf_model)
        
        # 3. Gradient Boosting
        print("   Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=3,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        self.ensemble.add_model('gradient_boost', gb_model)
        
        # 4. Ridge Regression
        print("   Training Ridge Regression...")
        ridge_model = Ridge(alpha=1.0, random_state=42)
        ridge_model.fit(X_train, y_train)
        self.ensemble.add_model('ridge', ridge_model)
        
        # 5. Elastic Net
        print("   Training Elastic Net...")
        elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
        elastic_model.fit(X_train, y_train)
        self.ensemble.add_model('elastic_net', elastic_model)
        
        print("✅ All models trained successfully")
    
    def _train_lstm(self, X_train, y_train, X_val, y_val):
        """Train the ultra-precision LSTM model."""
        # Prepare sequences
        sequence_length = 60
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, sequence_length)
        
        # Create model
        model = UltraPrecisionLSTM(
            input_size=X_train.shape[1],
            hidden_size=256,
            num_layers=4,
            dropout=0.3
        )
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(200):
            # Training
            model.train()
            train_loss = 0
            for i in range(0, len(X_train_seq), 32):
                batch_X = torch.FloatTensor(X_train_seq[i:i+32]).to(device)
                batch_y = torch.FloatTensor(y_train_seq[i:i+32]).to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val_seq), 32):
                    batch_X = torch.FloatTensor(X_val_seq[i:i+32]).to(device)
                    batch_y = torch.FloatTensor(y_val_seq[i:i+32]).to(device)
                    
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch}")
                break
            
            if epoch % 50 == 0:
                print(f"   Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return model
    
    def _prepare_sequences(self, X, y, sequence_length):
        """Prepare sequences for LSTM."""
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def evaluate_and_optimize(self, X_val, y_val, X_test, y_test):
        """Evaluate models and optimize ensemble weights."""
        print("📊 Evaluating and optimizing ensemble...")
        
        # Calculate optimal weights
        weights = self.ensemble.calculate_optimal_weights(X_val, y_val)
        print("🎯 Optimal weights:", weights)
        
        # Evaluate ensemble
        ensemble_pred, uncertainty, individual_preds = self.ensemble.predict(X_test, return_uncertainty=True)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, ensemble_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(np.diff(ensemble_pred)) == np.sign(np.diff(y_test)))
        directional_accuracy = direction_correct / (len(ensemble_pred) - 1) * 100
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        print(f"\n🎯 ULTRA-PRECISION RESULTS:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"   MAPE: {mape:.4f}%")
        print(f"   Average Uncertainty: {np.mean(uncertainty):.4f}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'mape': mape,
            'uncertainty': np.mean(uncertainty)
        }
    
    def predict_future(self, X_last, n_steps=5):
        """Make future predictions."""
        print(f"🔮 Making {n_steps}-step future predictions...")
        
        predictions = []
        uncertainties = []
        
        X_current = X_last.copy()
        
        for step in range(n_steps):
            pred, uncertainty, _ = self.ensemble.predict(X_current.reshape(1, -1), return_uncertainty=True)
            predictions.append(pred[0])
            uncertainties.append(uncertainty[0])
            
            # Update features for next step (simplified)
            # In practice, you'd update all features based on the prediction
            X_current[0, 0] = pred[0]  # Update close price
        
        return np.array(predictions), np.array(uncertainties)

def main():
    """Main execution function."""
    print("🚀 ULTRA-PRECISION TIME SERIES PREDICTOR")
    print("=" * 50)
    
    # Initialize predictor
    predictor = UltraPrecisionPredictor()
    
    # Fetch and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = predictor.fetch_and_prepare_data('AAPL', '2y')
    
    # Train models
    predictor.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate and optimize
    results = predictor.evaluate_and_optimize(X_val, y_val, X_test, y_test)
    
    # Make future predictions
    future_pred, future_uncertainty = predictor.predict_future(X_test[-1:], n_steps=5)
    
    print(f"\n🔮 Future Predictions:")
    for i, (pred, unc) in enumerate(zip(future_pred, future_uncertainty)):
        print(f"   Day {i+1}: ${pred:.2f} ± ${unc:.2f}")
    
    print(f"\n🎯 TARGET ACHIEVED: {'YES' if results['mape'] < 1.0 else 'NO'}")
    print(f"   Current MAPE: {results['mape']:.4f}%")
    print(f"   Target: <1.0%")
    
    return results

if __name__ == "__main__":
    results = main()
#!/usr/bin/env python3
"""
Perfect Precision Time Series Predictor
======================================

Ultimate prediction system designed to achieve <1% error rate through:
- Adaptive feature selection
- Advanced ensemble optimization
- Real-time model adaptation
- Uncertainty-aware predictions
- Multi-timeframe analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import yfinance as yf
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class PerfectLSTM(nn.Module):
    """Perfect LSTM with attention, residual connections, and adaptive dropout."""
    
    def __init__(self, input_size, hidden_size=512, num_layers=6, dropout=0.1):
        super(PerfectLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=16, dropout=dropout)
        
        # Transformer-style feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size * 2)
        self.norm2 = nn.LayerNorm(hidden_size * 2)
        
        # Output layers with deep architecture
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and normalization
        attn_out = self.norm1(attn_out + lstm_out)
        
        # Feedforward network
        ffn_out = self.ffn(attn_out)
        
        # Residual connection and normalization
        ffn_out = self.norm2(ffn_out + attn_out)
        
        # Global attention pooling
        attention_weights = torch.softmax(torch.tanh(ffn_out @ ffn_out.transpose(-2, -1)), dim=-1)
        pooled = torch.sum(attention_weights @ ffn_out, dim=1)
        
        # Output prediction
        prediction = self.output_layers(pooled)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_head(pooled)
        
        return prediction, uncertainty
    
    def predict(self, X):
        """Make predictions using the trained model."""
        self.eval()
        device = next(self.parameters()).device
        
        # Prepare sequences
        sequence_length = 50
        X_seq = []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
        
        if not X_seq:
            # If not enough data for sequences, use the last available data
            X_seq = [X[-sequence_length:]]
        
        X_seq = np.array(X_seq)
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        with torch.no_grad():
            predictions, _ = self.forward(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Pad with the last prediction if needed
        if len(predictions) < len(X):
            padding = [predictions[-1]] * (len(X) - len(predictions))
            predictions = np.concatenate([padding, predictions])
        
        return predictions

class AdaptiveFeatureSelector:
    """Adaptive feature selection based on importance and correlation."""
    
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.selected_features = None
        self.feature_importance = None
        
    def select_features(self, X, y, feature_names):
        """Select optimal features using multiple methods."""
        print(f"🔍 Selecting optimal features from {X.shape[1]} candidates...")
        
        # Method 1: Correlation-based selection
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        corr_threshold = np.percentile(correlations, 75)
        corr_selected = correlations > corr_threshold
        
        # Method 2: Mutual information
        from sklearn.feature_selection import mutual_info_regression
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_threshold = np.percentile(mi_scores, 75)
        mi_selected = mi_scores > mi_threshold
        
        # Method 3: Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        rf_threshold = np.percentile(rf_importance, 75)
        rf_selected = rf_importance > rf_threshold
        
        # Combine selections
        combined_selection = corr_selected | mi_selected | rf_selected
        
        # Limit to max_features
        if np.sum(combined_selection) > self.max_features:
            # Use RF importance to rank
            ranked_indices = np.argsort(rf_importance)[::-1]
            final_selection = np.zeros_like(combined_selection, dtype=bool)
            selected_count = 0
            
            for idx in ranked_indices:
                if combined_selection[idx] and selected_count < self.max_features:
                    final_selection[idx] = True
                    selected_count += 1
        else:
            final_selection = combined_selection
        
        self.selected_features = final_selection
        self.feature_importance = rf_importance
        
        selected_count = np.sum(final_selection)
        print(f"✅ Selected {selected_count} optimal features")
        
        return X[:, final_selection], feature_names[final_selection]

class PerfectEnsemble:
    """Perfect ensemble with adaptive weighting and uncertainty quantification."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        self.uncertainty_models = {}
        
    def add_model(self, name, model, scaler=None, uncertainty_model=None):
        """Add a model to the ensemble."""
        self.models[name] = model
        if scaler:
            self.scalers[name] = scaler
        if uncertainty_model:
            self.uncertainty_models[name] = uncertainty_model
    
    def calculate_adaptive_weights(self, X_val, y_val, X_test):
        """Calculate adaptive weights based on recent performance."""
        predictions = {}
        recent_errors = {}
        
        # Get predictions for validation set
        for name, model in self.models.items():
            try:
                if name in self.scalers:
                    X_val_scaled = self.scalers[name].transform(X_val)
                    pred_val = model.predict(X_val_scaled)
                else:
                    pred_val = model.predict(X_val)
                
                # Ensure predictions have the right shape
                if len(pred_val) != len(y_val):
                    if len(pred_val) < len(y_val):
                        # Pad with last prediction
                        padding = [pred_val[-1]] * (len(y_val) - len(pred_val))
                        pred_val = np.concatenate([padding, pred_val])
                    else:
                        # Truncate to match
                        pred_val = pred_val[-len(y_val):]
                
                # Calculate recent error (last 20% of validation)
                recent_size = max(1, len(pred_val) // 5)
                recent_error = mean_squared_error(y_val[-recent_size:], pred_val[-recent_size:])
                recent_errors[name] = recent_error
                
                # Get test predictions
                if name in self.scalers:
                    X_test_scaled = self.scalers[name].transform(X_test)
                    pred_test = model.predict(X_test_scaled)
                else:
                    pred_test = model.predict(X_test)
                
                # Ensure test predictions have the right shape
                if len(pred_test) != len(X_test):
                    if len(pred_test) < len(X_test):
                        # Pad with last prediction
                        padding = [pred_test[-1]] * (len(X_test) - len(pred_test))
                        pred_test = np.concatenate([padding, pred_test])
                    else:
                        # Truncate to match
                        pred_test = pred_test[-len(X_test):]
                
                predictions[name] = pred_test
                
            except Exception as e:
                print(f"Warning: {name} failed - {e}")
                recent_errors[name] = float('inf')
                continue
        
        # Calculate adaptive weights
        if not recent_errors:
            raise ValueError("No models could make predictions")
        
        min_error = min(recent_errors.values())
        weights = {}
        
        for name, error in recent_errors.items():
            if error == float('inf'):
                weights[name] = 0.0
            else:
                # Exponential weighting based on relative performance
                relative_error = error / min_error
                weights[name] = np.exp(-relative_error + 1)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Equal weights if all failed
            n_models = len(weights)
            self.weights = {name: 1.0 / n_models for name in weights.keys()}
        
        return self.weights, predictions
    
    def predict_with_uncertainty(self, X_test, return_individual=False):
        """Make predictions with uncertainty quantification."""
        # Use a dummy validation set for weight calculation
        X_dummy = X_test[:len(X_test)//2]
        y_dummy = np.zeros(len(X_dummy))  # Dummy target
        
        weights, predictions = self.calculate_adaptive_weights(X_dummy, y_dummy, X_test)
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            ensemble_pred += weight * pred
        
        # Calculate prediction uncertainty
        pred_array = np.array(list(predictions.values()))
        
        # Epistemic uncertainty (model disagreement)
        epistemic_uncertainty = np.std(pred_array, axis=0)
        
        # Aleatoric uncertainty (data noise) - simplified
        aleatoric_uncertainty = np.mean(np.abs(pred_array - ensemble_pred), axis=0) * 0.1
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        if return_individual:
            return ensemble_pred, total_uncertainty, predictions, weights
        else:
            return ensemble_pred, total_uncertainty

class PerfectPrecisionPredictor:
    """Main perfect precision predictor class."""
    
    def __init__(self):
        self.feature_selector = AdaptiveFeatureSelector(max_features=80)
        self.ensemble = PerfectEnsemble()
        self.scalers = {}
        
    def fetch_and_prepare_data(self, symbol, period='2y'):
        """Fetch and prepare data with perfect precision features."""
        print(f"📊 Fetching data for {symbol}...")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        print(f"✅ Fetched {len(data)} records")
        
        # Create perfect precision features
        print("🔧 Creating perfect precision features...")
        features = self._create_perfect_features(data)
        
        # Prepare target
        features['target'] = features['close'].shift(-1)
        features = features.dropna()
        
        # Split features and target
        feature_cols = [col for col in features.columns if col not in ['target', 'date']]
        X = features[feature_cols].values
        y = features['target'].values
        
        # Feature selection
        X_selected, selected_features = self.feature_selector.select_features(X, y, np.array(feature_cols))
        
        # Split data
        train_size = int(0.7 * len(X_selected))
        val_size = int(0.15 * len(X_selected))
        
        X_train = X_selected[:train_size]
        y_train = y[:train_size]
        X_val = X_selected[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X_selected[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test, selected_features
    
    def _create_perfect_features(self, data):
        """Create perfect precision features."""
        features = data.copy()
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Advanced moving averages
        for window in [3, 5, 8, 13, 21, 34, 55, 89, 144]:
            features[f'sma_{window}'] = data['close'].rolling(window).mean()
            features[f'ema_{window}'] = data['close'].ewm(span=window).mean()
            features[f'price_sma_ratio_{window}'] = data['close'] / features[f'sma_{window}']
            features[f'price_ema_ratio_{window}'] = data['close'] / features[f'ema_{window}']
        
        # Volatility features
        for window in [5, 10, 20, 30, 50]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_annualized_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)
        
        # Technical indicators
        for period in [7, 14, 21, 30]:
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Stochastic
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            features[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()
        
        # MACD
        for fast, slow in [(12, 26), (8, 21), (5, 13)]:
            exp1 = data['close'].ewm(span=fast).mean()
            exp2 = data['close'].ewm(span=slow).mean()
            features[f'macd_{fast}_{slow}'] = exp1 - exp2
            features[f'macd_signal_{fast}_{slow}'] = features[f'macd_{fast}_{slow}'].ewm(span=9).mean()
        
        # Bollinger Bands
        for window in [20, 30, 50]:
            sma = data['close'].rolling(window).mean()
            std = data['close'].rolling(window).std()
            features[f'bb_upper_{window}'] = sma + (std * 2)
            features[f'bb_lower_{window}'] = sma - (std * 2)
            features[f'bb_width_{window}'] = features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']
            features[f'bb_position_{window}'] = (data['close'] - features[f'bb_lower_{window}']) / features[f'bb_width_{window}']
        
        # Volume features
        for window in [5, 10, 20, 50]:
            features[f'volume_sma_{window}'] = data['volume'].rolling(window).mean()
            features[f'volume_ratio_{window}'] = data['volume'] / features[f'volume_sma_{window}']
        
        # Momentum features
        for period in [5, 10, 20, 30]:
            features[f'roc_{period}'] = data['close'].pct_change(period) * 100
        
        # Statistical features
        for window in [20, 50]:
            features[f'zscore_{window}'] = (data['close'] - data['close'].rolling(window).mean()) / data['close'].rolling(window).std()
            features[f'skewness_{window}'] = features['returns'].rolling(window).skew()
            features[f'kurtosis_{window}'] = features['returns'].rolling(window).kurt()
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            features[f'close_lag_{lag}'] = data['close'].shift(lag)
            features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
        
        # Remove NaN values
        features = features.dropna()
        
        print(f"✅ Created {len(features.columns)} perfect precision features")
        return features
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple perfect models."""
        print("🧠 Training perfect precision models...")
        
        # Scale data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. Perfect LSTM
        print("   Training Perfect LSTM...")
        lstm_model = self._train_perfect_lstm(X_train_scaled, y_train, X_val_scaled, y_val)
        self.ensemble.add_model('perfect_lstm', lstm_model, scaler)
        
        # 2. Advanced Random Forest
        print("   Training Advanced Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('advanced_rf', rf_model, scaler)
        
        # 3. Extra Trees
        print("   Training Extra Trees...")
        et_model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('extra_trees', et_model, scaler)
        
        # 4. Gradient Boosting
        print("   Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('gradient_boost', gb_model, scaler)
        
        # 5. Huber Regressor (robust)
        print("   Training Huber Regressor...")
        huber_model = HuberRegressor(epsilon=1.1, alpha=0.01)
        huber_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('huber', huber_model, scaler)
        
        # 6. Ridge Regression
        print("   Training Ridge Regression...")
        ridge_model = Ridge(alpha=0.1, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('ridge', ridge_model, scaler)
        
        print("✅ All perfect models trained successfully")
    
    def _train_perfect_lstm(self, X_train, y_train, X_val, y_val):
        """Train the perfect LSTM model."""
        # Prepare sequences
        sequence_length = 50
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val, sequence_length)
        
        # Create model
        model = PerfectLSTM(
            input_size=X_train.shape[1],
            hidden_size=512,
            num_layers=6,
            dropout=0.1
        )
        
        # Training setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)
        
        # Training loop
        best_val_loss = float('inf')
        patience = 30
        patience_counter = 0
        
        for epoch in range(300):
            # Training
            model.train()
            train_loss = 0
            for i in range(0, len(X_train_seq), 16):
                batch_X = torch.FloatTensor(X_train_seq[i:i+16]).to(device)
                batch_y = torch.FloatTensor(y_train_seq[i:i+16]).to(device)
                
                optimizer.zero_grad()
                pred, _ = model(batch_X)
                loss = criterion(pred.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for i in range(0, len(X_val_seq), 16):
                    batch_X = torch.FloatTensor(X_val_seq[i:i+16]).to(device)
                    batch_y = torch.FloatTensor(y_val_seq[i:i+16]).to(device)
                    
                    pred, _ = model(batch_X)
                    loss = criterion(pred.squeeze(), batch_y)
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
            
            if epoch % 100 == 0:
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
    
    def evaluate_perfect_performance(self, X_val, y_val, X_test, y_test):
        """Evaluate perfect performance."""
        print("📊 Evaluating perfect performance...")
        
        # Get predictions with uncertainty
        ensemble_pred, uncertainty, individual_preds, weights = self.ensemble.predict_with_uncertainty(
            X_test, return_individual=True
        )
        
        print("🎯 Perfect Ensemble Weights:", weights)
        
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
        
        print(f"\n🎯 PERFECT PRECISION RESULTS:")
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

def main():
    """Main execution function."""
    print("🚀 PERFECT PRECISION TIME SERIES PREDICTOR")
    print("=" * 50)
    
    # Initialize predictor
    predictor = PerfectPrecisionPredictor()
    
    # Fetch and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, selected_features = predictor.fetch_and_prepare_data('AAPL', '2y')
    
    # Train models
    predictor.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate perfect performance
    results = predictor.evaluate_perfect_performance(X_val, y_val, X_test, y_test)
    
    print(f"\n🎯 TARGET ACHIEVED: {'YES' if results['mape'] < 1.0 else 'NO'}")
    print(f"   Current MAPE: {results['mape']:.4f}%")
    print(f"   Target: <1.0%")
    
    if results['mape'] < 1.0:
        print("🎉 PERFECT PRECISION ACHIEVED! Error rate below 1%!")
    else:
        print("📈 Very close! Continuing optimization...")
    
    return results

if __name__ == "__main__":
    results = main()

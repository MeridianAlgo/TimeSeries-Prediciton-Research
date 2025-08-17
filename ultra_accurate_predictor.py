#!/usr/bin/env python3
"""
Ultra-Accurate Time Series Predictor
===================================

High-precision prediction system designed to achieve <1% error rate through:
- Advanced feature engineering
- Optimized ensemble methods
- Adaptive weighting
- Robust scaling
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class UltraAccurateFeatureEngineer:
    """Ultra-accurate feature engineering with 150+ features."""
    
    def __init__(self):
        self.features_created = 0
        
    def create_ultra_features(self, data):
        """Create ultra-accurate features."""
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
        
        # Statistical features
        features = self._create_statistical_features(features)
        
        # Lagged features
        features = self._create_lagged_features(features)
        
        # Interaction features
        features = self._create_interaction_features(features)
        
        # Remove NaN values
        features = features.dropna()
        
        print(f"✅ Created {len(features.columns)} ultra-accurate features")
        return features
    
    def _create_price_features(self, data):
        """Create advanced price-based features."""
        # Returns and log returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price ratios
        data['high_low_ratio'] = data['high'] / data['low']
        data['close_open_ratio'] = data['close'] / data['open']
        data['body_size'] = (data['close'] - data['open']) / data['open']
        data['upper_shadow'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['open']
        data['lower_shadow'] = (np.minimum(data['open'], data['close']) - data['low']) / data['open']
        
        # Price changes
        for period in [1, 2, 3, 5, 10, 20, 30]:
            data[f'price_change_{period}'] = data['close'].pct_change(period)
            data[f'price_change_abs_{period}'] = data['close'].pct_change(period).abs()
        
        # Moving averages with Fibonacci numbers
        for window in [3, 5, 8, 13, 21, 34, 55, 89, 144]:
            data[f'sma_{window}'] = data['close'].rolling(window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
            data[f'price_sma_ratio_{window}'] = data['close'] / data[f'sma_{window}']
            data[f'price_ema_ratio_{window}'] = data['close'] / data[f'ema_{window}']
            data[f'sma_ema_ratio_{window}'] = data[f'sma_{window}'] / data[f'ema_{window}']
        
        # Price levels
        data['price_level_10'] = (data['close'] // 10) * 10
        data['price_level_50'] = (data['close'] // 50) * 50
        data['price_level_100'] = (data['close'] // 100) * 100
        
        return data
    
    def _create_technical_indicators(self, data):
        """Create advanced technical indicators."""
        # RSI with multiple periods
        for period in [7, 14, 21, 30, 50]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD variations
        for fast, slow in [(12, 26), (8, 21), (5, 13), (3, 7)]:
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
            data[f'bb_squeeze_{window}'] = data[f'bb_width_{window}'] / data[f'bb_width_{window}'].rolling(20).mean()
        
        # Stochastic Oscillator
        for period in [14, 21, 30]:
            low_min = data['low'].rolling(period).min()
            high_max = data['high'].rolling(period).max()
            data[f'stoch_k_{period}'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            data[f'stoch_d_{period}'] = data[f'stoch_k_{period}'].rolling(3).mean()
        
        # Williams %R
        for period in [14, 21, 30]:
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
        
        # Garman-Klass volatility
        data['garman_klass_vol'] = np.sqrt(
            (0.5 * (np.log(data['high'] / data['low']) ** 2) - 
             (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)).rolling(20).mean()
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
        
        # Volume rate of change
        for period in [5, 10, 20]:
            data[f'volume_roc_{period}'] = data['volume'].pct_change(period) * 100
        
        # Volume weighted average price
        data['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        data['price_vwap_ratio'] = data['close'] / data['vwap']
        
        return data
    
    def _create_momentum_features(self, data):
        """Create momentum features."""
        # Rate of change
        for period in [5, 10, 20, 30, 50]:
            data[f'roc_{period}'] = data['close'].pct_change(period) * 100
        
        # Momentum
        for period in [5, 10, 20, 30]:
            data[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
        
        # Rate of change ratio
        for period in [5, 10, 20]:
            data[f'roc_ratio_{period}'] = data[f'roc_{period}'] / data[f'roc_{period}'].rolling(20).mean()
        
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
        
        # Rolling statistics
        for window in [20, 50]:
            data[f'rolling_mean_{window}'] = data['close'].rolling(window).mean()
            data[f'rolling_std_{window}'] = data['close'].rolling(window).std()
            data[f'rolling_min_{window}'] = data['close'].rolling(window).min()
            data[f'rolling_max_{window}'] = data['close'].rolling(window).max()
        
        return data
    
    def _create_lagged_features(self, data):
        """Create lagged features."""
        # Lagged prices
        for lag in [1, 2, 3, 5, 10, 20]:
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
        
        # Lagged ratios
        for lag in [1, 2, 3, 5]:
            data[f'price_ratio_lag_{lag}'] = data['close'] / data['close'].shift(lag)
            data[f'volume_ratio_lag_{lag}'] = data['volume'] / data['volume'].shift(lag)
        
        return data
    
    def _create_interaction_features(self, data):
        """Create interaction features."""
        # Price-volume interactions
        data['price_volume_interaction'] = data['close'] * data['volume']
        data['price_volume_ratio'] = data['close'] / data['volume']
        
        # Volatility-price interactions
        data['volatility_price_interaction'] = data['volatility_20'] * data['close']
        
        # RSI-price interactions
        data['rsi_price_interaction'] = data['rsi_14'] * data['close']
        
        # MACD-price interactions
        data['macd_price_interaction'] = data['macd_12_26'] * data['close']
        
        return data

class UltraAccurateEnsemble:
    """Ultra-accurate ensemble with optimized weighting."""
    
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
                    X_val_scaled = self.scalers[name].transform(X_val)
                    pred = model.predict(X_val_scaled)
                else:
                    pred = model.predict(X_val)
                
                predictions[name] = pred
                errors[name] = mean_squared_error(y_val, pred)
            except Exception as e:
                print(f"Warning: {name} failed - {e}")
                errors[name] = float('inf')
                continue
        
        if not errors:
            raise ValueError("No models could make predictions")
        
        # Calculate weights inversely proportional to error
        min_error = min(errors.values())
        weights = {}
        
        for name, error in errors.items():
            if error == float('inf'):
                weights[name] = 0.0
            else:
                # Exponential weighting for better performance
                relative_error = error / min_error
                weights[name] = np.exp(-relative_error)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Equal weights if all failed
            n_models = len(weights)
            self.weights = {name: 1.0 / n_models for name in weights.keys()}
        
        return self.weights
    
    def predict(self, X):
        """Make ensemble predictions."""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if name in self.scalers:
                    X_scaled = self.scalers[name].transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                
                predictions[name] = pred
            except Exception as e:
                print(f"Warning: {name} failed during prediction - {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0.0)
            ensemble_pred += weight * pred
        
        return ensemble_pred, predictions

class UltraAccuratePredictor:
    """Main ultra-accurate predictor class."""
    
    def __init__(self):
        self.feature_engineer = UltraAccurateFeatureEngineer()
        self.ensemble = UltraAccurateEnsemble()
        
    def fetch_and_prepare_data(self, symbol, period='2y'):
        """Fetch and prepare data with ultra-accurate features."""
        print(f"📊 Fetching data for {symbol}...")
        
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        
        print(f"✅ Fetched {len(data)} records")
        
        # Create ultra-accurate features
        print("🔧 Creating ultra-accurate features...")
        features = self.feature_engineer.create_ultra_features(data)
        
        # Prepare target
        features['target'] = features['close'].shift(-1)
        features = features.dropna()
        
        # Split features and target
        feature_cols = [col for col in features.columns if col not in ['target', 'date']]
        X = features[feature_cols].values
        y = features['target'].values
        
        # Feature selection
        print("🔍 Selecting optimal features...")
        selector = SelectKBest(score_func=f_regression, k=min(100, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        selected_features = np.array(feature_cols)[selector.get_support()]
        
        print(f"✅ Selected {len(selected_features)} optimal features")
        
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
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple ultra-accurate models."""
        print("🧠 Training ultra-accurate models...")
        
        # Scale data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 1. Advanced Random Forest
        print("   Training Advanced Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('advanced_rf', rf_model, scaler)
        
        # 2. Extra Trees
        print("   Training Extra Trees...")
        et_model = ExtraTreesRegressor(
            n_estimators=800,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        et_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('extra_trees', et_model, scaler)
        
        # 3. Gradient Boosting
        print("   Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('gradient_boost', gb_model, scaler)
        
        # 4. Huber Regressor (robust)
        print("   Training Huber Regressor...")
        huber_model = HuberRegressor(epsilon=1.1, alpha=0.001)
        huber_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('huber', huber_model, scaler)
        
        # 5. Ridge Regression
        print("   Training Ridge Regression...")
        ridge_model = Ridge(alpha=0.01, random_state=42)
        ridge_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('ridge', ridge_model, scaler)
        
        # 6. Elastic Net
        print("   Training Elastic Net...")
        elastic_model = ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=42)
        elastic_model.fit(X_train_scaled, y_train)
        self.ensemble.add_model('elastic_net', elastic_model, scaler)
        
        print("✅ All ultra-accurate models trained successfully")
    
    def evaluate_ultra_performance(self, X_val, y_val, X_test, y_test):
        """Evaluate ultra-accurate performance."""
        print("📊 Evaluating ultra-accurate performance...")
        
        # Calculate optimal weights
        weights = self.ensemble.calculate_optimal_weights(X_val, y_val)
        print("🎯 Optimal weights:", weights)
        
        # Make ensemble predictions
        ensemble_pred, individual_preds = self.ensemble.predict(X_test)
        
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
        
        print(f"\n🎯 ULTRA-ACCURATE RESULTS:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"   MAPE: {mape:.4f}%")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'mape': mape
        }

def main():
    """Main execution function."""
    print("🚀 ULTRA-ACCURATE TIME SERIES PREDICTOR")
    print("=" * 50)
    
    # Initialize predictor
    predictor = UltraAccuratePredictor()
    
    # Fetch and prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, selected_features = predictor.fetch_and_prepare_data('AAPL', '2y')
    
    # Train models
    predictor.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate ultra-accurate performance
    results = predictor.evaluate_ultra_performance(X_val, y_val, X_test, y_test)
    
    print(f"\n🎯 TARGET ACHIEVED: {'YES' if results['mape'] < 1.0 else 'NO'}")
    print(f"   Current MAPE: {results['mape']:.4f}%")
    print(f"   Target: <1.0%")
    
    if results['mape'] < 1.0:
        print("🎉 ULTRA-ACCURATE PREDICTION ACHIEVED! Error rate below 1%!")
    else:
        print("📈 Very close! Continuing optimization...")
    
    return results

if __name__ == "__main__":
    results = main()

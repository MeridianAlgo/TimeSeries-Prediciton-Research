#!/usr/bin/env python3
"""Maximum accuracy stock predictor with cutting-edge techniques."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))


class AdvancedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Advanced ensemble with dynamic weighting and outlier handling."""
    
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or Ridge(alpha=0.1)
        self.trained_models = {}
        self.meta_trained = False
        
    def fit(self, X, y):
        """Fit base models and meta model."""
        # Train base models
        for name, model in self.base_models.items():
            self.trained_models[name] = model.fit(X, y)
        
        # Create meta features
        meta_features = np.column_stack([
            model.predict(X) for model in self.trained_models.values()
        ])
        
        # Train meta model
        self.meta_model.fit(meta_features, y)
        self.meta_trained = True
        
        return self
    
    def predict(self, X):
        """Make predictions using stacked ensemble."""
        if not self.meta_trained:
            raise ValueError("Model not fitted!")
        
        # Get base predictions
        base_predictions = np.column_stack([
            model.predict(X) for model in self.trained_models.values()
        ])
        
        # Meta prediction
        return self.meta_model.predict(base_predictions)


class MaximumAccuracyPredictor:
    """Maximum accuracy predictor with cutting-edge techniques."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.weights = {}
        self.is_trained = False
        self.feature_importance = {}
        self.outlier_threshold = 3.0
        
    def create_ultimate_features(self, data):
        """Create the most comprehensive feature set possible."""
        df = data.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Advanced price patterns
        df['doji'] = (np.abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
        df['hammer'] = ((df['close'] > df['open']) & 
                       ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) &
                       ((df['high'] - df['close']) < 0.1 * (df['close'] - df['open']))).astype(int)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['price_volume'] = df['close'] * df['volume']
        df['vwap'] = (df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum())
        df['volume_price_trend'] = df['volume'] * np.sign(df['returns'])
        
        # Multi-timeframe moving averages
        ma_periods = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
        for period in ma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'wma_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1))
            )
            
            # MA relationships
            if period > 3:
                prev_period = ma_periods[ma_periods.index(period) - 1]
                df[f'ma_ratio_{period}_{prev_period}'] = df[f'sma_{period}'] / df[f'sma_{prev_period}']
            
            df[f'price_ma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
            df[f'ma_{period}_slope'] = df[f'sma_{period}'].diff(3)
        
        # Bollinger Bands with multiple periods
        for period in [10, 20, 50]:
            for std_mult in [1.5, 2.0, 2.5]:
                sma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                df[f'bb_upper_{period}_{std_mult}'] = sma + (std_mult * std)
                df[f'bb_lower_{period}_{std_mult}'] = sma - (std_mult * std)
                df[f'bb_position_{period}_{std_mult}'] = (df['close'] - df[f'bb_lower_{period}_{std_mult}']) / (df[f'bb_upper_{period}_{std_mult}'] - df[f'bb_lower_{period}_{std_mult}'])
                df[f'bb_width_{period}_{std_mult}'] = (df[f'bb_upper_{period}_{std_mult}'] - df[f'bb_lower_{period}_{std_mult}']) / sma
        
        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # RSI divergence
            df[f'rsi_{period}_slope'] = df[f'rsi_{period}'].diff(3)
            df[f'price_rsi_{period}_divergence'] = np.sign(df['close'].diff(3)) != np.sign(df[f'rsi_{period}'].diff(3))
        
        # MACD variations
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (19, 39, 9)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            df[f'macd_histogram_{fast}_{slow}_{signal}'] = macd - macd_signal
            df[f'macd_slope_{fast}_{slow}'] = macd.diff(3)
        
        # Stochastic variations
        for k_period, d_period in [(14, 3), (21, 3), (14, 5)]:
            low_min = df['low'].rolling(window=k_period).min()
            high_max = df['high'].rolling(window=k_period).max()
            stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
            stoch_d = stoch_k.rolling(d_period).mean()
            df[f'stoch_k_{k_period}'] = stoch_k
            df[f'stoch_d_{k_period}_{d_period}'] = stoch_d
            df[f'stoch_divergence_{k_period}'] = stoch_k - stoch_d
        
        # Williams %R
        for period in [14, 21, 28]:
            high_max = df['high'].rolling(window=period).max()
            low_min = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
        
        # Commodity Channel Index
        for period in [14, 20, 30]:
            tp = (df['high'] + df['low'] + df['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        # Average True Range
        for period in [7, 14, 21, 28]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(period).mean()
            df[f'atr_{period}'] = atr
            df[f'atr_ratio_{period}'] = atr / df['close']
            df[f'atr_normalized_{period}'] = atr / df['close'].rolling(period).mean()
        
        # Momentum and Rate of Change
        for period in [3, 5, 10, 15, 20, 25]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
            df[f'price_acceleration_{period}'] = df[f'roc_{period}'].diff()
        
        # Volatility measures
        for period in [5, 10, 20, 30, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()
            df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df[f'volatility_{period}'].rolling(50).mean()
            df[f'volatility_rank_{period}'] = df[f'volatility_{period}'].rolling(252).rank(pct=True)
        
        # Support and Resistance levels
        for period in [20, 50, 100]:
            df[f'support_{period}'] = df['low'].rolling(period).min()
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_distance_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']
            df[f'resistance_distance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
        
        # Fibonacci retracement levels
        for period in [20, 50]:
            high = df['high'].rolling(period).max()
            low = df['low'].rolling(period).min()
            diff = high - low
            for fib_level in [0.236, 0.382, 0.5, 0.618, 0.786]:
                df[f'fib_{fib_level}_{period}'] = high - (diff * fib_level)
                df[f'fib_distance_{fib_level}_{period}'] = np.abs(df['close'] - df[f'fib_{fib_level}_{period}']) / df['close']
        
        # Market structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_low'] = ((df['low'] > df['low'].shift(1)) & (df['high'] > df['high'].shift(1))).astype(int)
        df['lower_high'] = ((df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1))).astype(int)
        
        # Trend strength
        for period in [10, 20, 50]:
            df[f'trend_strength_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
            )
        
        # Lagged features with different periods
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'high_lag_{lag}'] = df['high'].shift(lag)
            df[f'low_lag_{lag}'] = df['low'].shift(lag)
        
        # Rolling statistics with multiple windows
        for window in [3, 5, 10, 15, 20, 30]:
            df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_max_{window}'] = df['close'].rolling(window).max()
            df[f'close_median_{window}'] = df['close'].rolling(window).median()
            df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
            df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
            df[f'close_range_{window}'] = df[f'close_max_{window}'] - df[f'close_min_{window}']
            df[f'close_position_{window}'] = (df['close'] - df[f'close_min_{window}']) / (df[f'close_range_{window}'] + 1e-10)
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['day_of_month'] = df.index.day
        df['week_of_year'] = df.index.isocalendar().week
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        df['is_year_start'] = df.index.is_year_start.astype(int)
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        # Cyclical encoding of time features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        return df
    
    def create_maximum_ensemble(self):
        """Create the most advanced ensemble possible."""
        # Level 1: Base models with different characteristics
        base_models_l1 = {
            'rf_deep': RandomForestRegressor(
                n_estimators=1000, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                random_state=42, n_jobs=-1
            ),
            'rf_wide': RandomForestRegressor(
                n_estimators=500, max_depth=None, min_samples_split=5,
                min_samples_leaf=2, max_features='log2', bootstrap=True,
                random_state=43, n_jobs=-1
            ),
            'et_aggressive': ExtraTreesRegressor(
                n_estimators=800, max_depth=25, min_samples_split=2,
                min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                random_state=44, n_jobs=-1
            ),
            'gbm_precise': GradientBoostingRegressor(
                n_estimators=1000, learning_rate=0.005, max_depth=6,
                min_samples_split=3, min_samples_leaf=2, subsample=0.8,
                random_state=45
            ),
            'gbm_robust': GradientBoostingRegressor(
                n_estimators=500, learning_rate=0.02, max_depth=10,
                min_samples_split=5, min_samples_leaf=3, subsample=0.9,
                random_state=46
            ),
            'ada_boost': AdaBoostRegressor(
                n_estimators=200, learning_rate=0.1, random_state=47
            )
        }
        
        # Level 2: Linear models with different regularization
        linear_models = {
            'ridge_strong': Ridge(alpha=10.0, random_state=48),
            'ridge_weak': Ridge(alpha=0.1, random_state=49),
            'lasso_strong': Lasso(alpha=1.0, random_state=50),
            'lasso_weak': Lasso(alpha=0.01, random_state=51),
            'elastic_balanced': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=52),
            'elastic_l1': ElasticNet(alpha=0.1, l1_ratio=0.9, random_state=53),
            'bayesian_ridge': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        }
        
        # Level 3: Neural networks with different architectures
        nn_models = {
            'nn_deep': MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 64, 32), activation='relu',
                solver='adam', alpha=0.001, learning_rate='adaptive',
                max_iter=2000, random_state=54
            ),
            'nn_wide': MLPRegressor(
                hidden_layer_sizes=(1024, 512, 256), activation='relu',
                solver='adam', alpha=0.01, learning_rate='adaptive',
                max_iter=2000, random_state=55
            ),
            'nn_tanh': MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), activation='tanh',
                solver='lbfgs', alpha=0.001, max_iter=1000, random_state=56
            )
        }
        
        # Level 4: Support Vector Machines
        svm_models = {
            'svr_rbf': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01),
            'svr_poly': SVR(kernel='poly', degree=3, C=100, gamma='scale', epsilon=0.01),
            'svr_linear': SVR(kernel='linear', C=10, epsilon=0.01)
        }
        
        # Combine all models
        all_models = {**base_models_l1, **linear_models, **nn_models, **svm_models}
        
        # Create advanced ensemble
        self.models = {
            'stacked_ensemble': AdvancedEnsembleRegressor(
                base_models=all_models,
                meta_model=Ridge(alpha=1.0)
            ),
            'voting_ensemble': VotingRegressor([
                ('rf', base_models_l1['rf_deep']),
                ('gbm', base_models_l1['gbm_precise']),
                ('ridge', linear_models['ridge_strong']),
                ('nn', nn_models['nn_deep'])
            ]),
            'bagged_ensemble': BaggingRegressor(
                estimator=GradientBoostingRegressor(n_estimators=100, random_state=57),
                n_estimators=10, random_state=58, n_jobs=-1
            )
        }
        
        # Set up scalers and feature selectors
        for model_name in self.models.keys():
            if 'nn' in model_name or 'svr' in model_name or 'stacked' in model_name:
                self.scalers[model_name] = QuantileTransformer(output_distribution='normal', random_state=59)
            else:
                self.scalers[model_name] = RobustScaler()
            
            # Use different feature selection strategies
            if 'stacked' in model_name:
                self.feature_selectors[model_name] = SelectKBest(f_regression, k=100)
            elif 'voting' in model_name:
                self.feature_selectors[model_name] = SelectKBest(f_regression, k=80)
            else:
                self.feature_selectors[model_name] = SelectKBest(f_regression, k=60)
    
    def remove_outliers(self, X, y):
        """Remove outliers using multiple methods."""
        # Z-score method
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        outlier_mask = z_scores < self.outlier_threshold
        
        # IQR method
        Q1 = np.percentile(y, 25)
        Q3 = np.percentile(y, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_mask = (y >= lower_bound) & (y <= upper_bound)
        
        # Combine masks
        final_mask = outlier_mask & iqr_mask
        
        return X[final_mask], y[final_mask]
    
    def train_maximum_accuracy_models(self, X, y):
        """Train models with maximum accuracy techniques."""
        self.create_maximum_ensemble()
        
        # Remove outliers
        X_clean, y_clean = self.remove_outliers(X, y)
        
        # Advanced time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        model_scores = {}
        
        for model_name, model in self.models.items():
            # Feature selection
            X_selected = self.feature_selectors[model_name].fit_transform(X_clean, y_clean)
            
            # Scaling
            X_scaled = self.scalers[model_name].fit_transform(X_selected)
            
            # Cross-validation
            cv_scores = {'rmse': [], 'mae': [], 'r2': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
                y_train_cv, y_val_cv = y_clean[train_idx], y_clean[val_idx]
                
                # Train model
                model.fit(X_train_cv, y_train_cv)
                
                # Validate
                y_pred_cv = model.predict(X_val_cv)
                
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                mae = mean_absolute_error(y_val_cv, y_pred_cv)
                r2 = r2_score(y_val_cv, y_pred_cv)
                
                cv_scores['rmse'].append(rmse)
                cv_scores['mae'].append(mae)
                cv_scores['r2'].append(r2)
            
            # Final training on all clean data
            model.fit(X_scaled, y_clean)
            
            # Store performance
            avg_rmse = np.mean(cv_scores['rmse'])
            avg_mae = np.mean(cv_scores['mae'])
            avg_r2 = np.mean(cv_scores['r2'])
            
            model_scores[model_name] = {
                'rmse': avg_rmse,
                'mae': avg_mae,
                'r2': avg_r2
            }
            
            pass  # Removed print statement
        
        # Calculate optimal ensemble weights
        total_inverse_rmse = sum(1/scores['rmse'] for scores in model_scores.values())
        
        for model_name, scores in model_scores.items():
            weight = (1/scores['rmse']) / total_inverse_rmse
            self.weights[model_name] = weight
        
        self.is_trained = True
        
        return model_scores
    
    def predict_maximum_accuracy(self, X):
        """Make maximum accuracy ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Models not trained!")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            # Feature selection
            X_selected = self.feature_selectors[model_name].transform(X)
            
            # Scaling
            X_scaled = self.scalers[model_name].transform(X_selected)
            
            # Predict
            pred = model.predict(X_scaled)
            predictions[model_name] = pred
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(predictions[list(predictions.keys())[0]]))
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        return ensemble_pred, predictions


def run_maximum_accuracy_simulation():
    """Run maximum accuracy stock prediction simulation."""
    print("🚀 MAXIMUM ACCURACY Stock Prediction Simulation")
    print("🎯 Target: MAXIMUM Accuracy, MINIMUM Error!")
    print("🧠 Using Cutting-Edge ML Ensemble + Advanced Features")
    print("=" * 70)
    
    try:
        # Load and enhance data
        print("📊 Loading and enhancing data...")
        from stock_predictor.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=3.0)
        
        if raw_data is None or raw_data.empty:
            raise Exception("Could not fetch data")
        
        # Prepare data
        if 'date' in raw_data.columns:
            raw_data['date'] = pd.to_datetime(raw_data['date'])
            raw_data = raw_data.set_index('date')
        
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in raw_data.columns:
                raw_data = raw_data.rename(columns={old_name: new_name})
        
        data = raw_data.sort_index()
        
        # Initialize predictor
        predictor = MaximumAccuracyPredictor()
        
        # Create ultimate features
        enhanced_data = predictor.create_ultimate_features(data)
        
        # Prepare training data - only select numeric columns
        numeric_cols = enhanced_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        # Handle NaN values more carefully
        print(f"🔍 Data shape before cleaning: {enhanced_data.shape}")
        print(f"🔍 NaN values per column: {enhanced_data.isnull().sum().sum()}")
        
        # Fill NaN values with forward fill, then backward fill, then 0
        enhanced_data = enhanced_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"🔍 Data shape after cleaning: {enhanced_data.shape}")
        print(f"🔍 Remaining NaN values: {enhanced_data.isnull().sum().sum()}")
        
        # Split data
        split_idx = int(len(enhanced_data) * 0.8)
        train_data = enhanced_data.iloc[:split_idx]
        test_data = enhanced_data.iloc[split_idx:]
        
        X_train = train_data[feature_cols].values
        y_train = train_data['close'].values
        X_test = test_data[feature_cols].values
        y_test = test_data['close'].values
        
        print(f"📈 Training samples: {len(X_train)}")
        print(f"📊 Test samples: {len(X_test)}")
        print(f"🔧 Features: {len(feature_cols)}")
        
        # Train maximum accuracy models
        model_scores = predictor.train_maximum_accuracy_models(X_train, y_train)
        
        # Make predictions
        print("\n🎯 Making maximum accuracy predictions...")
        ensemble_pred, individual_preds = predictor.predict_maximum_accuracy(X_test)
        
        # Calculate accuracy metrics
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mae = mean_absolute_error(y_test, ensemble_pred)
        r2 = r2_score(y_test, ensemble_pred)
        
        # Calculate percentage accuracy
        percentage_errors = np.abs((y_test - ensemble_pred) / y_test) * 100
        accuracy_0_5pct = np.mean(percentage_errors <= 0.5) * 100
        accuracy_1pct = np.mean(percentage_errors <= 1.0) * 100
        accuracy_1_5pct = np.mean(percentage_errors <= 1.5) * 100
        accuracy_2pct = np.mean(percentage_errors <= 2.0) * 100
        accuracy_3pct = np.mean(percentage_errors <= 3.0) * 100
        
        print(f"\n🎉 MAXIMUM ACCURACY RESULTS:")
        print(f"📊 RMSE: ${rmse:.3f}")
        print(f"📊 MAE: ${mae:.3f}")
        print(f"📊 R² Score: {r2:.4f}")
        print(f"🎯 Accuracy (±0.5%): {accuracy_0_5pct:.1f}%")
        print(f"🎯 Accuracy (±1.0%): {accuracy_1pct:.1f}%")
        print(f"🎯 Accuracy (±1.5%): {accuracy_1_5pct:.1f}%")
        print(f"🎯 Accuracy (±2.0%): {accuracy_2pct:.1f}%")
        print(f"🎯 Accuracy (±3.0%): {accuracy_3pct:.1f}%")
        print(f"📈 Mean Error: {np.mean(percentage_errors):.2f}%")
        print(f"📉 Median Error: {np.median(percentage_errors):.2f}%")
        print(f"📊 Max Error: {np.max(percentage_errors):.2f}%")
        
        # Visualize results
        plt.figure(figsize=(16, 12))
        
        # Main prediction plot
        plt.subplot(2, 3, 1)
        dates = test_data.index
        plt.plot(dates, y_test, label='Actual', color='black', linewidth=2)
        plt.plot(dates, ensemble_pred, label='Maximum Accuracy Prediction', color='red', linewidth=2)
        plt.title('Maximum Accuracy Stock Price Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(2, 3, 2)
        errors = y_test - ensemble_pred
        plt.hist(errors, bins=30, alpha=0.7, color='blue')
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error ($)')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        plt.grid(True, alpha=0.3)
        
        # Percentage error distribution
        plt.subplot(2, 3, 3)
        plt.hist(percentage_errors, bins=30, alpha=0.7, color='green')
        plt.title('Percentage Error Distribution')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.8, label='1% threshold')
        plt.axvline(x=2.0, color='orange', linestyle='--', alpha=0.8, label='2% threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy over time
        plt.subplot(2, 3, 4)
        rolling_accuracy = pd.Series(percentage_errors).rolling(20).apply(lambda x: np.mean(x <= 1.5) * 100)
        plt.plot(dates, rolling_accuracy, color='purple', linewidth=2)
        plt.title('Rolling Accuracy (±1.5%) Over Time')
        plt.xlabel('Date')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        # Model performance comparison
        plt.subplot(2, 3, 5)
        model_names = list(model_scores.keys())
        rmse_scores = [model_scores[name]['rmse'] for name in model_names]
        plt.bar(range(len(model_names)), rmse_scores, alpha=0.7)
        plt.title('Model RMSE Comparison')
        plt.xlabel('Models')
        plt.ylabel('RMSE ($)')
        plt.xticks(range(len(model_names)), model_names, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Scatter plot: Actual vs Predicted
        plt.subplot(2, 3, 6)
        plt.scatter(y_test, ensemble_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Actual vs Predicted (R² = {r2:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return predictor, model_scores, {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy_0_5pct': accuracy_0_5pct,
            'accuracy_1pct': accuracy_1pct,
            'accuracy_1_5pct': accuracy_1_5pct,
            'accuracy_2pct': accuracy_2pct,
            'accuracy_3pct': accuracy_3pct,
            'mean_error': np.mean(percentage_errors),
            'median_error': np.median(percentage_errors),
            'max_error': np.max(percentage_errors)
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    predictor, scores, results = run_maximum_accuracy_simulation()
    
    if results:
        print(f"\n🏆 FINAL MAXIMUM ACCURACY RESULTS:")
        print(f"🎯 Ultra-High Accuracy (±1%): {results['accuracy_1pct']:.1f}%")
        print(f"🎯 High Accuracy (±1.5%): {results['accuracy_1_5pct']:.1f}%")
        print(f"🎯 Good Accuracy (±2%): {results['accuracy_2pct']:.1f}%")
        print(f"📉 Ultra-Low RMSE: ${results['rmse']:.3f}")
        print(f"📈 Ultra-High R²: {results['r2']:.4f}")
        print(f"📊 Mean Error: {results['mean_error']:.2f}%")
        print(f"📊 Median Error: {results['median_error']:.2f}%")
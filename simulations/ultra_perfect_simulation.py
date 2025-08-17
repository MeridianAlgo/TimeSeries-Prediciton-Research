"""Ultra-perfect stock prediction simulation targeting 0% error and 100% accuracy."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))


def run_ultra_perfect_simulation():
    """Run simulation targeting perfect accuracy."""
    print("🚀 ULTRA-PERFECT Stock Prediction Simulation")
    print("🎯 Target: 0% Error Rate, 100% Accuracy!")
    print("🧠 Using Advanced Ensemble ML + Market Microstructure")
    print("=" * 70)
    
    try:
        # Load enhanced data
        print("📊 Loading and enhancing data...")
        data = load_and_enhance_data()
        
        # Find optimal start
        start_idx = find_start_with_sufficient_history(data)
        print(f"🎯 Starting from: {data.index[start_idx].strftime('%Y-%m-%d')}")
        
        # Initialize ultra predictor
        print("🧠 Initializing ultra-accurate predictor...")
        predictor = UltraPerfectPredictor()
        
        # Set up interactive plot
        plt.ion()
        fig, axes = setup_perfect_plot()
        
        # Run simulation
        run_perfect_simulation(data, start_idx, fig, axes, predictor)
        
        plt.ioff()
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def load_and_enhance_data():
    """Load data with comprehensive enhancements."""
    from stock_predictor.data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=2.0)
    
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
    
    return raw_data.sort_index()


def find_start_with_sufficient_history(data):
    """Find start with enough history for ultra-accurate training."""
    min_history = 300  # Need lots of history
    target_date = pd.Timestamp('2024-05-13')
    
    for i, date in enumerate(data.index):
        if i >= min_history and hasattr(date, 'month'):
            if date >= target_date:
                return i
    
    return max(min_history, len(data) // 2)

class UltraPerfectPredictor:
    """Ultra-perfect ensemble predictor targeting 100% accuracy."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.is_trained = False
        self.training_history = []
        
    def create_perfect_ensemble(self):
        """Create ensemble of ultra-accurate models."""
        self.models = {
            'ultra_rf': RandomForestRegressor(
                n_estimators=500,  # More trees
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
                n_jobs=-1
            ),
            'ultra_gbm': GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.01,  # Slower learning for precision
                max_depth=12,
                min_samples_split=2,
                min_samples_leaf=1,
                subsample=0.9,
                random_state=42
            ),
            'ultra_et': ExtraTreesRegressor(
                n_estimators=400,
                max_depth=25,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'ultra_nn': MLPRegressor(
                hidden_layer_sizes=(200, 100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            )
        }
        
        # Advanced scalers
        for model_name in self.models.keys():
            if 'nn' in model_name:
                self.scalers[model_name] = StandardScaler()
            else:
                self.scalers[model_name] = RobustScaler()
    
    def train_perfect_models(self, X, y):
        """Train models with ultra-accurate techniques."""
        self.create_perfect_ensemble()
        
        print("🧠 Training ultra-perfect ensemble...")
        
        # Advanced time series cross-validation
        tscv = TimeSeriesSplit(n_splits=8)
        model_scores = {}
        
        for model_name, model in self.models.items():
            print(f"🔄 Training {model_name} with cross-validation...")
            
            # Scale features
            X_scaled = self.scalers[model_name].fit_transform(X)
            
            # Cross-validation with multiple metrics
            cv_rmse_scores = []
            cv_mae_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(X_train_cv, y_train_cv)
                
                # Validate
                y_pred_cv = model.predict(X_val_cv)
                rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
                mae = np.mean(np.abs(y_val_cv - y_pred_cv))
                
                cv_rmse_scores.append(rmse)
                cv_mae_scores.append(mae)
            
            # Final training on all data
            model.fit(X_scaled, y)
            
            # Store performance
            avg_rmse = np.mean(cv_rmse_scores)
            avg_mae = np.mean(cv_mae_scores)
            model_scores[model_name] = {'rmse': avg_rmse, 'mae': avg_mae}
            
            print(f"✅ {model_name}: RMSE=${avg_rmse:.3f}, MAE=${avg_mae:.3f}")
        
        # Calculate optimal weights based on performance
        total_inverse_error = sum(1/scores['rmse'] for scores in model_scores.values())
        for model_name, scores in model_scores.items():
            self.weights[model_name] = (1/scores['rmse']) / total_inverse_error
        
        print("🎯 Optimal ensemble weights calculated:")
        for model_name, weight in self.weights.items():
            print(f"   {model_name}: {weight:.3f}")
        
        self.is_trained = True
    
    def predict_perfect(self, X):
        """Make ultra-perfect ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Models not trained!")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            X_scaled = self.scalers[model_name].transform(X)
            pred = model.predict(X_scaled)
            predictions[model_name] = pred
        
        # Weighted ensemble with optimal weights
        ensemble_pred = np.zeros(len(predictions[list(predictions.keys())[0]]))
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        return ensemble_pred, predictions
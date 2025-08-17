"""
Predictor for Time Series Prediction
===================================

Main prediction interface for making forecasts with trained models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class Predictor:
    """
    Main prediction interface for time series forecasting.
    
    Provides a unified interface for making predictions with different model types.
    """
    
    def __init__(self, 
                 models: Dict[str, Any] = None,
                 ensemble_weights: Dict[str, float] = None,
                 scaler=None,
                 target_scaler=None):
        """
        Initialize predictor.
        
        Args:
            models: Dictionary of trained models
            ensemble_weights: Weights for ensemble prediction
            scaler: Feature scaler for inverse transformation
            target_scaler: Target scaler for inverse transformation
        """
        self.models = models or {}
        self.ensemble_weights = ensemble_weights or {}
        self.scaler = scaler
        self.target_scaler = target_scaler
        
    def add_model(self, name: str, model: Any):
        """Add a model to the predictor."""
        self.models[name] = model
        
    def set_ensemble_weights(self, weights: Dict[str, float]):
        """Set ensemble weights."""
        self.ensemble_weights = weights
        
    def predict_single(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a single model.
        
        Args:
            model_name: Name of the model to use
            X: Input features
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Handle different model types
        if isinstance(model, nn.Module):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions = model(X_tensor).cpu().numpy().flatten()
        elif hasattr(model, 'predict'):
            # Scikit-learn model
            predictions = model.predict(X).flatten()
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        
        # Inverse transform if target scaler available
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def predict_ensemble(self, X: np.ndarray, return_individual: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            return_individual: Whether to return individual model predictions
            
        Returns:
            Ensemble predictions, optionally with individual predictions
        """
        if not self.models:
            raise ValueError("No models available for prediction")
        
        if not self.ensemble_weights:
            # Equal weights if no weights specified
            n_models = len(self.models)
            self.ensemble_weights = {name: 1.0 / n_models for name in self.models.keys()}
        
        # Get predictions from all models
        individual_predictions = {}
        for model_name in self.models.keys():
            try:
                pred = self.predict_single(model_name, X)
                individual_predictions[model_name] = pred
            except Exception as e:
                print(f"Warning: Failed to get prediction from {model_name}: {e}")
                continue
        
        if not individual_predictions:
            raise ValueError("No models could make predictions")
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(individual_predictions.values())[0])
        total_weight = 0.0
        
        for model_name, pred in individual_predictions.items():
            weight = self.ensemble_weights.get(model_name, 0.0)
            ensemble_pred += weight * pred
            total_weight += weight
        
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
        
        if return_individual:
            return ensemble_pred, individual_predictions
        else:
            return ensemble_pred
    
    def predict_with_confidence(self, X: np.ndarray, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals.
        
        Args:
            X: Input features
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Get individual predictions
        ensemble_pred, individual_preds = self.predict_ensemble(X, return_individual=True)
        
        # Calculate confidence intervals using ensemble variance
        pred_array = np.array(list(individual_preds.values()))
        
        # Calculate mean and standard deviation
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # Calculate confidence interval
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        confidence_lower = mean_pred - z_score * std_pred
        confidence_upper = mean_pred + z_score * std_pred
        
        return {
            'predictions': ensemble_pred,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'prediction_std': std_pred,
            'individual_predictions': individual_preds
        }
    
    def predict_future(self, X_last: np.ndarray, n_steps: int = 5, 
                      feature_updater=None) -> np.ndarray:
        """
        Make multi-step future predictions.
        
        Args:
            X_last: Last known feature values
            n_steps: Number of future steps to predict
            feature_updater: Function to update features for next step
            
        Returns:
            Array of future predictions
        """
        predictions = []
        X_current = X_last.copy()
        
        for step in range(n_steps):
            # Make prediction for current step
            pred = self.predict_ensemble(X_current.reshape(1, -1))
            predictions.append(pred[0])
            
            # Update features for next step if updater provided
            if feature_updater is not None and step < n_steps - 1:
                X_current = feature_updater(X_current, pred[0])
        
        return np.array(predictions)
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prediction accuracy.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculate basic metrics
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(np.diff(y_pred)) == np.sign(np.diff(y_true)))
        directional_accuracy = direction_correct / (len(y_pred) - 1) * 100
        
        # Calculate correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Calculate MAPE
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'mape': mape
        }
        
        return metrics
    
    def get_model_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Get performance comparison of all models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with performance metrics for each model
        """
        results = []
        
        for model_name in self.models.keys():
            try:
                y_pred = self.predict_single(model_name, X_test)
                metrics = self.evaluate_predictions(y_test, y_pred)
                metrics['model_name'] = model_name
                results.append(metrics)
            except Exception as e:
                print(f"Warning: Could not evaluate {model_name}: {e}")
                continue
        
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
    
    def save_predictor(self, filepath: str):
        """Save predictor state."""
        import joblib
        
        # Prepare data for saving
        save_data = {
            'ensemble_weights': self.ensemble_weights,
            'scaler': self.scaler,
            'target_scaler': self.target_scaler
        }
        
        # Note: Models need to be saved separately due to different types
        joblib.dump(save_data, filepath)
        print(f"✅ Predictor state saved to {filepath}")
    
    def load_predictor(self, filepath: str):
        """Load predictor state."""
        import joblib
        
        save_data = joblib.load(filepath)
        self.ensemble_weights = save_data['ensemble_weights']
        self.scaler = save_data['scaler']
        self.target_scaler = save_data['target_scaler']
        
        print(f"✅ Predictor state loaded from {filepath}")
    
    def get_prediction_summary(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Get a summary of predictions.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with prediction summary
        """
        # Get ensemble prediction with confidence
        results = self.predict_with_confidence(X)
        
        summary = {
            'n_predictions': len(results['predictions']),
            'mean_prediction': np.mean(results['predictions']),
            'std_prediction': np.mean(results['prediction_std']),
            'min_prediction': np.min(results['predictions']),
            'max_prediction': np.max(results['predictions']),
            'confidence_range': np.mean(results['confidence_upper'] - results['confidence_lower']),
            'n_models_used': len(results['individual_predictions'])
        }
        
        return summary

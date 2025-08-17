"""
Ensemble Framework for Time Series Prediction
============================================

Provides dynamic ensemble methods combining multiple models with
adaptive weighting and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn


class EnsembleFramework:
    """
    Advanced ensemble framework for time series prediction.
    
    Supports multiple weighting methods and uncertainty quantification.
    """
    
    def __init__(self, 
                 models: Dict[str, Any] = None,
                 weighting_method: str = 'performance_based',
                 uncertainty_method: str = 'ensemble_variance',
                 performance_window: int = 30):
        """
        Initialize ensemble framework.
        
        Args:
            models: Dictionary of model names to model instances
            weighting_method: Method for calculating ensemble weights
            uncertainty_method: Method for uncertainty quantification
            performance_window: Window for performance-based weighting
        """
        self.models = models or {}
        self.weights = {}
        self.weighting_method = weighting_method
        self.uncertainty_method = uncertainty_method
        self.performance_window = performance_window
        self.performance_history = {}
        
    def add_model(self, name: str, model: Any, performance_metrics: Dict[str, float] = None):
        """Add a model to the ensemble."""
        self.models[name] = model
        if performance_metrics:
            self.performance_history[name] = performance_metrics
            
    def calculate_weights(self, performance_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate ensemble weights based on performance.
        
        Args:
            performance_dict: Dictionary of model performance metrics
            
        Returns:
            Dictionary of model weights
        """
        if self.weighting_method == 'inverse_error':
            return self._inverse_error_weighting(performance_dict)
        elif self.weighting_method == 'performance_based':
            return self._performance_based_weighting(performance_dict)
        elif self.weighting_method == 'equal':
            return self._equal_weighting(performance_dict)
        else:
            return self._inverse_error_weighting(performance_dict)
    
    def _inverse_error_weighting(self, performance_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate weights inversely proportional to error."""
        errors = {}
        for model_name, metrics in performance_dict.items():
            # Use RMSE as error metric
            errors[model_name] = metrics.get('rmse', float('inf'))
        
        # Avoid division by zero
        min_error = min(errors.values()) if errors else 1.0
        if min_error == 0:
            min_error = 1e-6
            
        # Calculate inverse weights
        inverse_weights = {name: min_error / max(error, 1e-6) for name, error in errors.items()}
        total_weight = sum(inverse_weights.values())
        
        # Normalize
        weights = {name: weight / total_weight for name, weight in inverse_weights.items()}
        self.weights = weights
        return weights
    
    def _performance_based_weighting(self, performance_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate weights based on R² score."""
        scores = {}
        for model_name, metrics in performance_dict.items():
            r2 = metrics.get('r2_score', -float('inf'))
            # Convert negative R² to small positive weight
            scores[model_name] = max(r2, 0.01)
        
        total_score = sum(scores.values())
        weights = {name: score / total_score for name, score in scores.items()}
        self.weights = weights
        return weights
    
    def _equal_weighting(self, performance_dict: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Equal weighting for all models."""
        n_models = len(performance_dict)
        weights = {name: 1.0 / n_models for name in performance_dict.keys()}
        self.weights = weights
        return weights
    
    def predict_ensemble(self, X: np.ndarray, return_confidence: bool = False) -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with ensemble predictions and optional confidence intervals
        """
        if not self.weights:
            raise ValueError("Ensemble weights not calculated. Call calculate_weights() first.")
        
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                elif hasattr(model, 'forward'):
                    # PyTorch model
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        pred = model(X_tensor).cpu().numpy()
                else:
                    continue
                    
                predictions[model_name] = pred.flatten()
            except Exception as e:
                print(f"Warning: Failed to get prediction from {model_name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        result = {'ensemble_prediction': ensemble_pred}
        
        # Calculate confidence intervals if requested
        if return_confidence and len(predictions) > 1:
            confidence_intervals = self._calculate_confidence_intervals(predictions)
            result.update(confidence_intervals)
        
        return result
    
    def _calculate_confidence_intervals(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals using ensemble variance."""
        pred_array = np.array(list(predictions.values()))
        
        # Calculate mean and standard deviation
        mean_pred = np.mean(pred_array, axis=0)
        std_pred = np.std(pred_array, axis=0)
        
        # 95% confidence interval (1.96 * std)
        confidence_level = 1.96
        confidence_lower = mean_pred - confidence_level * std_pred
        confidence_upper = mean_pred + confidence_level * std_pred
        
        return {
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'prediction_std': std_pred
        }
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 100) -> Dict[str, Any]:
        """
        Train all models in the ensemble.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs for neural networks
            
        Returns:
            Dictionary of training results
        """
        training_results = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'train'):
                    # Custom training method
                    result = model.train(X_train, y_train, X_val, y_val, epochs)
                elif hasattr(model, 'fit'):
                    # Scikit-learn style
                    result = model.fit(X_train, y_train)
                else:
                    print(f"Warning: Model {model_name} has no training method")
                    continue
                    
                training_results[model_name] = result
                print(f"✓ Trained {model_name}")
                
            except Exception as e:
                print(f"✗ Failed to train {model_name}: {e}")
                continue
        
        return training_results
    
    def get_model_performance(self) -> pd.DataFrame:
        """Get performance summary of all models."""
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(self.performance_history, orient='index')
        df['model_name'] = df.index
        df['weight'] = df.index.map(self.weights)
        
        return df.reset_index(drop=True)
    
    def update_performance(self, model_name: str, metrics: Dict[str, float]):
        """Update performance metrics for a model."""
        self.performance_history[model_name] = metrics
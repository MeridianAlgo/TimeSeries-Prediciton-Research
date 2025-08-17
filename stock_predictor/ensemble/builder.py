"""Ensemble builder for combining multiple prediction models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import json
from datetime import datetime

from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import EnsembleError
from stock_predictor.evaluation.evaluator import ModelEvaluator


class EnsembleBuilder:
    """Builds and manages ensemble predictions from multiple models."""
    
    def __init__(self, weighting_method: str = "inverse_error", min_weight: float = 0.1):
        self.logger = get_logger('ensemble.builder')
        self.weighting_method = weighting_method
        self.min_weight = min_weight
        self.models = {}
        self.weights = {}
        self.performance_history = {}
        self.evaluator = ModelEvaluator()
        
        # Configuration
        self.config = {
            'confidence_level': 0.95,
            'dynamic_adjustment': True,
            'adjustment_window': 30,
            'weight_decay': 0.95,  # For exponential decay of old performance
            'bootstrap_samples': 1000  # For confidence interval estimation
        }
    
    def add_model(self, model_name: str, model: Any, 
                  performance_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Name identifier for the model
            model: Trained model object with predict method
            performance_metrics: Optional pre-computed performance metrics
        """
        self.models[model_name] = model
        
        if performance_metrics:
            self.performance_history[model_name] = [performance_metrics]
        
        self.logger.info(f"Added model '{model_name}' to ensemble")
    
    def remove_model(self, model_name: str) -> None:
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.weights:
                del self.weights[model_name]
            if model_name in self.performance_history:
                del self.performance_history[model_name]
            
            self.logger.info(f"Removed model '{model_name}' from ensemble")
    
    def calculate_weights(self, performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate ensemble weights based on performance metrics.
        
        Args:
            performance_metrics: Dict of {model_name: {metric: value}}
            
        Returns:
            Dictionary of normalized weights
        """
        if not performance_metrics:
            raise EnsembleError("No performance metrics provided for weight calculation")
        
        weights = {}
        
        if self.weighting_method == "inverse_error":
            # Use inverse of RMSE for weighting (lower error = higher weight)
            for model_name, metrics in performance_metrics.items():
                rmse = metrics.get('rmse', float('inf'))
                if rmse == 0:
                    weights[model_name] = 1.0
                else:
                    weights[model_name] = 1.0 / rmse
        
        elif self.weighting_method == "inverse_mae":
            # Use inverse of MAE for weighting
            for model_name, metrics in performance_metrics.items():
                mae = metrics.get('mae', float('inf'))
                if mae == 0:
                    weights[model_name] = 1.0
                else:
                    weights[model_name] = 1.0 / mae
        
        elif self.weighting_method == "r2_score":
            # Use R² score for weighting (higher R² = higher weight)
            for model_name, metrics in performance_metrics.items():
                r2 = metrics.get('r2_score', 0.0)
                weights[model_name] = max(0.0, r2)  # Ensure non-negative
        
        elif self.weighting_method == "directional_accuracy":
            # Use directional accuracy for weighting
            for model_name, metrics in performance_metrics.items():
                dir_acc = metrics.get('directional_accuracy', 0.0)
                weights[model_name] = dir_acc / 100.0  # Convert percentage to decimal
        
        elif self.weighting_method == "equal":
            # Equal weights for all models
            for model_name in performance_metrics.keys():
                weights[model_name] = 1.0
        
        else:
            raise EnsembleError(f"Unknown weighting method: {self.weighting_method}")
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            # Fallback to equal weights
            weights = {name: 1.0 for name in performance_metrics.keys()}
            total_weight = len(weights)
        
        normalized_weights = {
            name: weight / total_weight for name, weight in weights.items()
        }
        
        # Apply minimum weight constraint
        normalized_weights = self._apply_min_weight_constraint(normalized_weights)
        
        self.weights = normalized_weights
        
        self.logger.info(f"Calculated weights using {self.weighting_method}: {normalized_weights}")
        
        return normalized_weights
    
    def _apply_min_weight_constraint(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum weight constraint to prevent model exclusion."""
        adjusted_weights = weights.copy()
        n_models = len(adjusted_weights)
        
        # Check if min_weight constraint is feasible
        if self.min_weight * n_models > 1.0:
            # If not feasible, adjust min_weight
            feasible_min_weight = 0.8 / n_models  # Leave some room for variation
            self.logger.warning(f"Adjusting min_weight from {self.min_weight} to {feasible_min_weight}")
            min_weight_to_use = feasible_min_weight
        else:
            min_weight_to_use = self.min_weight
        
        # Apply minimum weight constraint
        for name in adjusted_weights:
            if adjusted_weights[name] < min_weight_to_use:
                adjusted_weights[name] = min_weight_to_use
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                name: weight / total_weight for name, weight in adjusted_weights.items()
            }
        
        return adjusted_weights
    
    def weighted_prediction(self, predictions: Dict[str, np.ndarray], 
                          weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Combine predictions using weighted averaging.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            weights: Optional custom weights (uses self.weights if None)
            
        Returns:
            Weighted ensemble predictions
        """
        if not predictions:
            raise EnsembleError("No predictions provided")
        
        if weights is None:
            weights = self.weights
        
        if not weights:
            raise EnsembleError("No weights available. Calculate weights first.")
        
        # Ensure all models have predictions
        missing_models = set(weights.keys()) - set(predictions.keys())
        if missing_models:
            self.logger.warning(f"Missing predictions for models: {missing_models}")
            # Remove missing models from weights and renormalize
            available_weights = {k: v for k, v in weights.items() if k in predictions}
            total_weight = sum(available_weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in available_weights.items()}
            else:
                raise EnsembleError("No valid model predictions available")
        
        # Calculate weighted average
        ensemble_pred = np.zeros_like(next(iter(predictions.values())))
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                ensemble_pred += weights[model_name] * pred
        
        return ensemble_pred
    
    def calculate_confidence_intervals(self, predictions: Dict[str, np.ndarray], 
                                     weights: Optional[Dict[str, float]] = None,
                                     confidence_level: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for ensemble predictions.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            weights: Optional custom weights
            confidence_level: Confidence level (default from config)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if confidence_level is None:
            confidence_level = self.config['confidence_level']
        
        if weights is None:
            weights = self.weights
        
        # Calculate weighted ensemble prediction
        ensemble_pred = self.weighted_prediction(predictions, weights)
        
        # Calculate prediction variance using weighted variance formula
        pred_arrays = [predictions[name] for name in weights.keys() if name in predictions]
        weight_values = [weights[name] for name in weights.keys() if name in predictions]
        
        if len(pred_arrays) < 2:
            # If only one model, use a default uncertainty
            uncertainty = np.std(ensemble_pred) * 0.1  # 10% of prediction std
            margin = uncertainty * stats.norm.ppf((1 + confidence_level) / 2)
            return ensemble_pred - margin, ensemble_pred + margin
        
        # Calculate weighted variance
        weighted_mean = ensemble_pred
        weighted_variance = np.zeros_like(ensemble_pred)
        
        for pred, weight in zip(pred_arrays, weight_values):
            weighted_variance += weight * (pred - weighted_mean) ** 2
        
        # Calculate confidence intervals
        std_error = np.sqrt(weighted_variance)
        margin = std_error * stats.norm.ppf((1 + confidence_level) / 2)
        
        lower_bounds = ensemble_pred - margin
        upper_bounds = ensemble_pred + margin
        
        return lower_bounds, upper_bounds
    
    def bootstrap_confidence_intervals(self, predictions: Dict[str, np.ndarray],
                                     weights: Optional[Dict[str, float]] = None,
                                     n_bootstrap: int = None,
                                     confidence_level: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals using bootstrap resampling.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
            weights: Optional custom weights
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if n_bootstrap is None:
            n_bootstrap = self.config['bootstrap_samples']
        
        if confidence_level is None:
            confidence_level = self.config['confidence_level']
        
        if weights is None:
            weights = self.weights
        
        n_samples = len(next(iter(predictions.values())))
        bootstrap_predictions = []
        
        # Generate bootstrap samples
        for _ in range(n_bootstrap):
            # Resample indices
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Create bootstrap predictions
            bootstrap_pred_dict = {
                name: pred[bootstrap_indices] for name, pred in predictions.items()
            }
            
            # Calculate ensemble prediction for this bootstrap sample
            ensemble_pred = self.weighted_prediction(bootstrap_pred_dict, weights)
            bootstrap_predictions.append(ensemble_pred)
        
        # Calculate percentiles
        bootstrap_predictions = np.array(bootstrap_predictions)
        alpha = 1 - confidence_level
        
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        return lower_bounds, upper_bounds
    
    def update_weights_dynamically(self, recent_errors: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Update ensemble weights based on recent performance.
        
        Args:
            recent_errors: Dict of {model_name: [recent_error_values]}
            
        Returns:
            Updated weights
        """
        if not recent_errors:
            return self.weights
        
        # Calculate recent performance metrics
        recent_performance = {}
        for model_name, errors in recent_errors.items():
            if errors:
                recent_performance[model_name] = {
                    'rmse': float(np.sqrt(np.mean(np.array(errors) ** 2))),
                    'mae': float(np.mean(np.abs(errors)))
                }
        
        if recent_performance:
            # Update weights based on recent performance
            new_weights = self.calculate_weights(recent_performance)
            
            # Apply exponential smoothing with old weights
            if self.weights:
                decay = self.config.get('weight_decay', 0.95)
                smoothed_weights = {}
                
                for model_name in new_weights:
                    old_weight = self.weights.get(model_name, 0.0)
                    new_weight = new_weights[model_name]
                    smoothed_weights[model_name] = decay * old_weight + (1 - decay) * new_weight
                
                # Renormalize
                total_weight = sum(smoothed_weights.values())
                if total_weight > 0:
                    smoothed_weights = {
                        name: weight / total_weight for name, weight in smoothed_weights.items()
                    }
                
                self.weights = smoothed_weights
            else:
                self.weights = new_weights
            
            self.logger.info(f"Updated weights dynamically: {self.weights}")
        
        return self.weights
    
    def predict_ensemble(self, X_test: np.ndarray, 
                        return_individual: bool = False,
                        return_confidence: bool = True) -> Dict[str, Any]:
        """
        Generate ensemble predictions.
        
        Args:
            X_test: Test features
            return_individual: Whether to return individual model predictions
            return_confidence: Whether to return confidence intervals
            
        Returns:
            Dictionary with ensemble results
        """
        if not self.models:
            raise EnsembleError("No models in ensemble")
        
        # Get predictions from all models
        individual_predictions = {}
        failed_models = []
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_test)
                individual_predictions[model_name] = pred
            except Exception as e:
                self.logger.warning(f"Model {model_name} prediction failed: {str(e)}")
                failed_models.append(model_name)
        
        if not individual_predictions:
            raise EnsembleError("All model predictions failed")
        
        # Calculate ensemble prediction
        ensemble_pred = self.weighted_prediction(individual_predictions)
        
        results = {
            'ensemble_prediction': ensemble_pred,
            'model_weights': self.weights.copy(),
            'failed_models': failed_models
        }
        
        if return_individual:
            results['individual_predictions'] = individual_predictions
        
        if return_confidence:
            try:
                lower_bounds, upper_bounds = self.calculate_confidence_intervals(individual_predictions)
                results['confidence_lower'] = lower_bounds
                results['confidence_upper'] = upper_bounds
            except Exception as e:
                self.logger.warning(f"Confidence interval calculation failed: {str(e)}")
        
        return results
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""
        return {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'weighting_method': self.weighting_method,
            'current_weights': self.weights.copy(),
            'min_weight': self.min_weight,
            'config': self.config.copy(),
            'performance_history_available': bool(self.performance_history)
        }
    
    def save_ensemble_state(self, filepath: str) -> None:
        """Save ensemble state to file."""
        state = {
            'weighting_method': self.weighting_method,
            'min_weight': self.min_weight,
            'weights': self.weights,
            'performance_history': self.performance_history,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Ensemble state saved to {filepath}")
            
        except Exception as e:
            raise EnsembleError(f"Failed to save ensemble state: {str(e)}")
    
    def load_ensemble_state(self, filepath: str) -> None:
        """Load ensemble state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.weighting_method = state.get('weighting_method', 'inverse_error')
            self.min_weight = state.get('min_weight', 0.1)
            self.weights = state.get('weights', {})
            self.performance_history = state.get('performance_history', {})
            self.config.update(state.get('config', {}))
            
            self.logger.info(f"Ensemble state loaded from {filepath}")
            
        except Exception as e:
            raise EnsembleError(f"Failed to load ensemble state: {str(e)}")
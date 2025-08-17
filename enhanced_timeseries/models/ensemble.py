"""
Ensemble prediction system for combining multiple time series models.
Implements performance-based weighting, stacking, blending, and dynamic adaptation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ..core.interfaces import BaseModel, BasePredictor, PredictionResult
import warnings
warnings.filterwarnings('ignore')


class EnsembleWeighting:
    """Different weighting strategies for ensemble models."""
    
    @staticmethod
    def equal_weights(n_models: int) -> np.ndarray:
        """Equal weights for all models."""
        return np.ones(n_models) / n_models
    
    @staticmethod
    def performance_weights(performances: np.ndarray, method: str = 'inverse_error') -> np.ndarray:
        """Performance-based weights."""
        if method == 'inverse_error':
            # Lower error = higher weight
            weights = 1.0 / (performances + 1e-8)
        elif method == 'exponential':
            # Exponential weighting favoring best performers
            weights = np.exp(-performances * 10)
        elif method == 'softmax':
            # Softmax of negative errors
            weights = np.exp(-performances * 5)
            weights = weights / np.sum(weights)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
        
        # Normalize weights
        return weights / np.sum(weights)
    
    @staticmethod
    def volatility_adjusted_weights(performances: np.ndarray, volatilities: np.ndarray, 
                                  risk_aversion: float = 1.0) -> np.ndarray:
        """Risk-adjusted weights considering both performance and volatility."""
        # Sharpe-like ratio: performance / volatility
        risk_adjusted_scores = performances / (volatilities + 1e-8)
        
        # Apply risk aversion
        adjusted_scores = risk_adjusted_scores ** risk_aversion
        
        # Convert to weights (higher score = higher weight)
        weights = adjusted_scores / np.sum(adjusted_scores)
        
        return weights
    
    @staticmethod
    def time_decay_weights(performances: np.ndarray, decay_factor: float = 0.95) -> np.ndarray:
        """Time-decaying weights giving more importance to recent performance."""
        n_periods = len(performances)
        time_weights = np.array([decay_factor ** (n_periods - i - 1) for i in range(n_periods)])
        
        # Weight by time-decayed performance
        weighted_performance = performances * time_weights
        
        # Convert to ensemble weights
        weights = 1.0 / (weighted_performance + 1e-8)
        return weights / np.sum(weights)


class StackingEnsemble(nn.Module):
    """Stacking ensemble using a meta-learner."""
    
    def __init__(self, n_models: int, meta_learner_type: str = 'linear', 
                 hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_models = n_models
        self.meta_learner_type = meta_learner_type
        
        if meta_learner_type == 'linear':
            self.meta_learner = nn.Linear(n_models, 1)
        elif meta_learner_type == 'mlp':
            self.meta_learner = nn.Sequential(
                nn.Linear(n_models, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            raise ValueError(f"Unknown meta-learner type: {meta_learner_type}")
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacking ensemble.
        
        Args:
            predictions: Tensor of shape (batch_size, n_models)
            
        Returns:
            Final prediction of shape (batch_size, 1)
        """
        return self.meta_learner(predictions)


class BlendingEnsemble:
    """Blending ensemble using holdout validation."""
    
    def __init__(self, blend_method: str = 'linear'):
        self.blend_method = blend_method
        self.blender = None
        self.is_fitted = False
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Fit the blending model.
        
        Args:
            predictions: Array of shape (n_samples, n_models)
            targets: Array of shape (n_samples,)
        """
        if self.blend_method == 'linear':
            self.blender = LinearRegression()
        elif self.blend_method == 'ridge':
            self.blender = Ridge(alpha=1.0)
        elif self.blend_method == 'lasso':
            self.blender = Lasso(alpha=0.1)
        elif self.blend_method == 'random_forest':
            self.blender = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown blend method: {self.blend_method}")
        
        self.blender.fit(predictions, targets)
        self.is_fitted = True
    
    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Make blended predictions.
        
        Args:
            predictions: Array of shape (n_samples, n_models)
            
        Returns:
            Blended predictions of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Blender must be fitted before making predictions")
        
        return self.blender.predict(predictions)


class EnsemblePredictionSystem(BasePredictor):
    """
    Comprehensive ensemble prediction system that combines multiple models
    with various weighting and combination strategies.
    """
    
    def __init__(self, models: List[BaseModel], weighting_method: str = 'performance',
                 combination_method: str = 'weighted_average', 
                 adaptation_window: int = 100, min_samples: int = 50):
        super().__init__()
        
        self.models = models
        self.weighting_method = weighting_method
        self.combination_method = combination_method
        self.adaptation_window = adaptation_window
        self.min_samples = min_samples
        
        # Performance tracking
        self.model_performances = {i: [] for i in range(len(models))}
        self.model_predictions_history = {i: [] for i in range(len(models))}
        self.actual_values_history = []
        
        # Current weights
        self.current_weights = EnsembleWeighting.equal_weights(len(models))
        
        # Ensemble methods
        self.stacking_ensemble = None
        self.blending_ensemble = None
        
        # Initialize stacking if needed
        if combination_method == 'stacking':
            self.stacking_ensemble = StackingEnsemble(len(models))
        elif combination_method == 'blending':
            self.blending_ensemble = BlendingEnsemble()
        
        self.is_trained = False
        
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None, 
              **kwargs) -> Dict[str, float]:
        """
        Train all models in the ensemble.
        
        Args:
            data: Training data
            validation_data: Optional validation data for blending
            **kwargs: Additional training arguments
            
        Returns:
            Dictionary of training metrics
        """
        print("🚀 Training Ensemble Models...")
        
        individual_metrics = {}
        
        # Train each model
        for i, model in enumerate(self.models):
            print(f"Training Model {i+1}/{len(self.models)}: {model.__class__.__name__}")
            
            try:
                if hasattr(model, 'train') and callable(model.train):
                    metrics = model.train(data, **kwargs)
                    individual_metrics[f'model_{i}'] = metrics
                else:
                    print(f"Warning: Model {i} does not have a train method")
                    individual_metrics[f'model_{i}'] = {'mae': float('inf')}
                    
            except Exception as e:
                print(f"Error training model {i}: {e}")
                individual_metrics[f'model_{i}'] = {'mae': float('inf')}
        
        # Train meta-learners if validation data is provided
        if validation_data is not None and self.combination_method in ['stacking', 'blending']:
            self._train_meta_learners(validation_data)
        
        self.is_trained = True
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_ensemble_metrics(individual_metrics)
        
        return ensemble_metrics
    
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """
        Make ensemble prediction.
        
        Args:
            data: Input data
            
        Returns:
            Ensemble prediction result
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        # Get predictions from all models
        individual_predictions = []
        individual_uncertainties = []
        model_contributions = {}
        
        for i, model in enumerate(self.models):
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    pred, uncertainty = model.predict_with_uncertainty(data)
                    if isinstance(pred, torch.Tensor):
                        pred = pred.detach().cpu().numpy()
                    if isinstance(uncertainty, torch.Tensor):
                        uncertainty = uncertainty.detach().cpu().numpy()
                elif hasattr(model, 'predict'):
                    pred = model.predict(data)
                    if isinstance(pred, torch.Tensor):
                        pred = pred.detach().cpu().numpy()
                    uncertainty = np.zeros_like(pred)
                else:
                    pred = np.array([0.0])
                    uncertainty = np.array([1.0])
                
                individual_predictions.append(pred.flatten()[0])
                individual_uncertainties.append(uncertainty.flatten()[0])
                model_contributions[f'model_{i}'] = pred.flatten()[0]
                
            except Exception as e:
                print(f"Error getting prediction from model {i}: {e}")
                individual_predictions.append(0.0)
                individual_uncertainties.append(1.0)
                model_contributions[f'model_{i}'] = 0.0
        
        # Combine predictions
        ensemble_prediction, ensemble_uncertainty = self._combine_predictions(
            individual_predictions, individual_uncertainties
        )
        
        # Create prediction result
        result = PredictionResult(
            symbol='ensemble',
            timestamp=pd.Timestamp.now(),
            prediction=ensemble_prediction,
            confidence_lower=ensemble_prediction - 1.96 * np.sqrt(ensemble_uncertainty),
            confidence_upper=ensemble_prediction + 1.96 * np.sqrt(ensemble_uncertainty),
            uncertainty=ensemble_uncertainty,
            model_contributions=model_contributions,
            features_used=['ensemble_features']
        )
        
        return result
    
    def predict_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, PredictionResult]:
        """
        Make batch predictions for multiple assets.
        
        Args:
            data_dict: Dictionary of asset data
            
        Returns:
            Dictionary of prediction results
        """
        results = {}
        
        for symbol, data in data_dict.items():
            try:
                result = self.predict(data)
                result.symbol = symbol
                results[symbol] = result
            except Exception as e:
                print(f"Error predicting for {symbol}: {e}")
                # Create dummy result
                results[symbol] = PredictionResult(
                    symbol=symbol,
                    timestamp=pd.Timestamp.now(),
                    prediction=0.0,
                    confidence_lower=-0.1,
                    confidence_upper=0.1,
                    uncertainty=1.0,
                    model_contributions={},
                    features_used=[]
                )
        
        return results
    
    def update_performance(self, predictions: List[float], actual: float):
        """
        Update model performance tracking.
        
        Args:
            predictions: List of individual model predictions
            actual: Actual value
        """
        self.actual_values_history.append(actual)
        
        # Update individual model performance
        for i, pred in enumerate(predictions):
            self.model_predictions_history[i].append(pred)
            
            # Calculate recent performance
            if len(self.model_predictions_history[i]) >= self.min_samples:
                recent_preds = self.model_predictions_history[i][-self.adaptation_window:]
                recent_actuals = self.actual_values_history[-len(recent_preds):]
                
                mae = mean_absolute_error(recent_actuals, recent_preds)
                self.model_performances[i].append(mae)
        
        # Update ensemble weights
        self._update_weights()
    
    def _combine_predictions(self, predictions: List[float], 
                           uncertainties: List[float]) -> Tuple[float, float]:
        """
        Combine individual predictions into ensemble prediction.
        
        Args:
            predictions: List of individual predictions
            uncertainties: List of individual uncertainties
            
        Returns:
            Tuple of (ensemble_prediction, ensemble_uncertainty)
        """
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        
        if self.combination_method == 'weighted_average':
            # Weighted average
            ensemble_pred = np.sum(self.current_weights * predictions)
            
            # Uncertainty combination (assuming independence)
            weighted_uncertainties = self.current_weights ** 2 * uncertainties
            ensemble_uncertainty = np.sum(weighted_uncertainties)
            
        elif self.combination_method == 'uncertainty_weighted':
            # Weight by inverse uncertainty
            inv_uncertainties = 1.0 / (uncertainties + 1e-8)
            weights = inv_uncertainties / np.sum(inv_uncertainties)
            
            ensemble_pred = np.sum(weights * predictions)
            ensemble_uncertainty = 1.0 / np.sum(inv_uncertainties)
            
        elif self.combination_method == 'stacking':
            if self.stacking_ensemble is not None:
                pred_tensor = torch.FloatTensor(predictions).unsqueeze(0)
                ensemble_pred = self.stacking_ensemble(pred_tensor).item()
                ensemble_uncertainty = np.mean(uncertainties)  # Simple average for uncertainty
            else:
                # Fallback to weighted average
                ensemble_pred = np.sum(self.current_weights * predictions)
                ensemble_uncertainty = np.sum(self.current_weights ** 2 * uncertainties)
                
        elif self.combination_method == 'blending':
            if self.blending_ensemble is not None and self.blending_ensemble.is_fitted:
                ensemble_pred = self.blending_ensemble.predict(predictions.reshape(1, -1))[0]
                ensemble_uncertainty = np.mean(uncertainties)
            else:
                # Fallback to weighted average
                ensemble_pred = np.sum(self.current_weights * predictions)
                ensemble_uncertainty = np.sum(self.current_weights ** 2 * uncertainties)
                
        else:
            # Default: simple average
            ensemble_pred = np.mean(predictions)
            ensemble_uncertainty = np.mean(uncertainties)
        
        return float(ensemble_pred), float(ensemble_uncertainty)
    
    def _update_weights(self):
        """Update ensemble weights based on recent performance."""
        if len(self.model_performances[0]) < self.min_samples:
            return  # Not enough data yet
        
        # Get recent performance for each model
        recent_performances = []
        for i in range(len(self.models)):
            if len(self.model_performances[i]) > 0:
                recent_perf = np.mean(self.model_performances[i][-10:])  # Last 10 evaluations
                recent_performances.append(recent_perf)
            else:
                recent_performances.append(1.0)  # Default high error
        
        recent_performances = np.array(recent_performances)
        
        # Update weights based on method
        if self.weighting_method == 'performance':
            self.current_weights = EnsembleWeighting.performance_weights(recent_performances)
        elif self.weighting_method == 'equal':
            self.current_weights = EnsembleWeighting.equal_weights(len(self.models))
        elif self.weighting_method == 'exponential':
            self.current_weights = EnsembleWeighting.performance_weights(
                recent_performances, method='exponential'
            )
        
        # Ensure weights are valid
        if np.any(np.isnan(self.current_weights)) or np.sum(self.current_weights) == 0:
            self.current_weights = EnsembleWeighting.equal_weights(len(self.models))
    
    def _train_meta_learners(self, validation_data: pd.DataFrame):
        """Train meta-learners for stacking/blending."""
        # Get predictions from all models on validation data
        val_predictions = []
        
        for model in self.models:
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(validation_data)
                    if isinstance(pred, torch.Tensor):
                        pred = pred.detach().cpu().numpy()
                    val_predictions.append(pred.flatten())
                else:
                    val_predictions.append(np.zeros(len(validation_data)))
            except:
                val_predictions.append(np.zeros(len(validation_data)))
        
        val_predictions = np.array(val_predictions).T  # Shape: (n_samples, n_models)
        
        # Create dummy targets (in practice, you'd have real validation targets)
        val_targets = np.random.randn(len(validation_data))  # Placeholder
        
        # Train meta-learners
        if self.combination_method == 'stacking' and self.stacking_ensemble is not None:
            # Train stacking ensemble
            pred_tensor = torch.FloatTensor(val_predictions)
            target_tensor = torch.FloatTensor(val_targets)
            
            optimizer = torch.optim.Adam(self.stacking_ensemble.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            for epoch in range(100):
                optimizer.zero_grad()
                output = self.stacking_ensemble(pred_tensor).squeeze()
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
        
        elif self.combination_method == 'blending' and self.blending_ensemble is not None:
            # Train blending ensemble
            self.blending_ensemble.fit(val_predictions, val_targets)
    
    def _calculate_ensemble_metrics(self, individual_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate ensemble-level metrics."""
        # Extract MAE values
        maes = []
        for metrics in individual_metrics.values():
            if isinstance(metrics, dict) and 'mae' in metrics:
                maes.append(metrics['mae'])
        
        if maes:
            ensemble_mae = np.mean(maes)
            best_mae = np.min(maes)
            worst_mae = np.max(maes)
        else:
            ensemble_mae = float('inf')
            best_mae = float('inf')
            worst_mae = float('inf')
        
        return {
            'ensemble_mae': ensemble_mae,
            'best_individual_mae': best_mae,
            'worst_individual_mae': worst_mae,
            'n_models': len(self.models),
            'diversity': worst_mae - best_mae if worst_mae != float('inf') else 0.0
        }
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return {f'model_{i}': weight for i, weight in enumerate(self.current_weights)}
    
    def get_performance_history(self) -> Dict[str, List[float]]:
        """Get performance history for all models."""
        return self.model_performances.copy()
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get comprehensive ensemble information."""
        return {
            'n_models': len(self.models),
            'weighting_method': self.weighting_method,
            'combination_method': self.combination_method,
            'current_weights': self.get_model_weights(),
            'is_trained': self.is_trained,
            'adaptation_window': self.adaptation_window,
            'model_types': [model.__class__.__name__ for model in self.models]
        }


class EnsembleConfig:
    """Configuration for ensemble system."""
    
    def __init__(self, **kwargs):
        self.weighting_method = kwargs.get('weighting_method', 'performance')
        self.combination_method = kwargs.get('combination_method', 'weighted_average')
        self.adaptation_window = kwargs.get('adaptation_window', 100)
        self.min_samples = kwargs.get('min_samples', 50)
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'weighting_method': self.weighting_method,
            'combination_method': self.combination_method,
            'adaptation_window': self.adaptation_window,
            'min_samples': self.min_samples
        }


def create_ensemble_system(models: List[BaseModel], config: EnsembleConfig) -> EnsemblePredictionSystem:
    """Factory function to create ensemble system."""
    return EnsemblePredictionSystem(models, **config.to_dict())
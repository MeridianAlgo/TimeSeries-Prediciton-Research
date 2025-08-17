"""Base model interface for all prediction models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import ModelTrainingError, ModelPredictionError
from stock_predictor.models.persistence import ModelPersistence, HyperparameterManager


class BaseModel(ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.hyperparameters = {}
        self.training_history = {}
        self.model = None
        self.feature_names = []
        self.training_time = 0.0
        self.last_trained = None
        
        # Initialize utilities
        self.logger = get_logger(f'models.{name}')
        self.persistence = ModelPersistence()
        self.hyperparameter_manager = HyperparameterManager()
        
        # Load default hyperparameters
        self._load_default_hyperparameters()
    
    @abstractmethod
    def _build_model(self) -> Any:
        """Build the underlying model with current hyperparameters."""
        pass
    
    @abstractmethod
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def _predict_model(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        pass
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              feature_names: Optional[list] = None) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            feature_names: Names of features (optional)
        """
        self.logger.info(f"Starting training for {self.name}")
        start_time = datetime.now()
        
        try:
            # Validate inputs
            self._validate_training_data(X_train, y_train, X_val, y_val)
            
            # Store feature names
            if feature_names:
                self.feature_names = feature_names
            else:
                self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            
            # Build model
            self.model = self._build_model()
            
            # Fit model
            self._fit_model(X_train, y_train, X_val, y_val)
            
            # Update training status
            self.is_trained = True
            self.last_trained = datetime.now()
            self.training_time = (self.last_trained - start_time).total_seconds()
            
            self.logger.info(f"Training completed for {self.name} in {self.training_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.name}: {str(e)}")
            raise ModelTrainingError(f"Training failed for {self.name}: {str(e)}")
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on the test data.
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ModelPredictionError(f"Model {self.name} is not trained")
        
        try:
            self._validate_prediction_data(X_test)
            predictions = self._predict_model(X_test)
            
            self.logger.debug(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {self.name}: {str(e)}")
            raise ModelPredictionError(f"Prediction failed for {self.name}: {str(e)}")
    
    def predict_with_uncertainty(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X_test: Test features
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        predictions = self.predict(X_test)
        
        # Default implementation returns zero uncertainty
        # Subclasses can override for model-specific uncertainty estimation
        uncertainties = np.zeros_like(predictions)
        
        return predictions, uncertainties
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()
    
    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Set hyperparameters for the model."""
        self.hyperparameters.update(params)
        self.hyperparameter_manager.save_hyperparameters(self.name, self.hyperparameters)
        self.logger.info(f"Updated hyperparameters for {self.name}")
    
    def save_model(self, filepath: str = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to saved model
        """
        if not self.is_trained:
            raise ModelTrainingError(f"Cannot save untrained model {self.name}")
        
        metadata = {
            'model_name': self.name,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'training_history': self.training_history
        }
        
        if filepath:
            # Custom filepath - save directly
            import joblib
            joblib.dump(self.model, filepath)
            return filepath
        else:
            # Use persistence manager
            return self.persistence.save_model(self.model, self.name, metadata)
    
    def load_model(self, filepath: str = None) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to model file (if None, loads latest)
        """
        try:
            if filepath is None:
                filepath = self.persistence.get_latest_model(self.name)
            
            self.model = self.persistence.load_model(filepath)
            
            # Load metadata
            metadata = self.persistence.load_model_metadata(filepath)
            if metadata:
                self.hyperparameters = metadata.get('hyperparameters', {})
                self.feature_names = metadata.get('feature_names', [])
                self.training_time = metadata.get('training_time', 0.0)
                self.training_history = metadata.get('training_history', {})
                
                last_trained_str = metadata.get('last_trained')
                if last_trained_str:
                    self.last_trained = datetime.fromisoformat(last_trained_str)
            
            self.is_trained = True
            self.logger.info(f"Model {self.name} loaded from {filepath}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model {self.name}: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Default implementation returns empty dict
        # Subclasses should override for model-specific importance
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            'name': self.name,
            'is_trained': self.is_trained,
            'hyperparameters': self.hyperparameters,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'training_time': self.training_time,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'training_history': self.training_history
        }
    
    def _load_default_hyperparameters(self) -> None:
        """Load default hyperparameters from config."""
        default_params = self.hyperparameter_manager.load_hyperparameters(self.name)
        if default_params:
            self.hyperparameters = default_params
    
    def _validate_training_data(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: Optional[np.ndarray], y_val: Optional[np.ndarray]) -> None:
        """Validate training data inputs."""
        if X_train.shape[0] != y_train.shape[0]:
            raise ModelTrainingError("X_train and y_train must have same number of samples")
        
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ModelTrainingError("X_val and y_val must have same number of samples")
            
            if X_train.shape[1] != X_val.shape[1]:
                raise ModelTrainingError("X_train and X_val must have same number of features")
    
    def _validate_prediction_data(self, X_test: np.ndarray) -> None:
        """Validate prediction data inputs."""
        if len(self.feature_names) > 0 and X_test.shape[1] != len(self.feature_names):
            raise ModelPredictionError(
                f"Expected {len(self.feature_names)} features, got {X_test.shape[1]}"
            )
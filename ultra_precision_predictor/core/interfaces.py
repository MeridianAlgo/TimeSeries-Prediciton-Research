"""Abstract interfaces for ultra-precision predictor components."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from .config import PredictionResult, ValidationReport, SystemHealthReport


class FeatureEngineer(ABC):
    """Abstract interface for feature engineering components."""
    
    @abstractmethod
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate engineered features from raw data."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings."""
        pass


class OutlierDetector(ABC):
    """Abstract interface for outlier detection components."""
    
    @abstractmethod
    def detect_outliers(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Detect outliers and return boolean mask."""
        pass
    
    @abstractmethod
    def remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers from data."""
        pass
    
    @abstractmethod
    def get_removal_statistics(self) -> Dict[str, float]:
        """Get outlier removal statistics."""
        pass


class EnsembleModel(ABC):
    """Abstract interface for ensemble model components."""
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the ensemble model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate predictions with confidence scores."""
        pass
    
    @abstractmethod
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        pass
    
    @abstractmethod
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models."""
        pass


class PredictionRefiner(ABC):
    """Abstract interface for prediction refinement components."""
    
    @abstractmethod
    def refine_predictions(self, predictions: np.ndarray, context: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply multi-stage refinement to predictions."""
        pass
    
    @abstractmethod
    def get_refinement_stages(self, predictions: np.ndarray) -> List[np.ndarray]:
        """Get predictions at each refinement stage."""
        pass
    
    @abstractmethod
    def validate_refinement_quality(self, refined: np.ndarray, original: np.ndarray) -> Dict[str, float]:
        """Validate refinement effectiveness."""
        pass


class Validator(ABC):
    """Abstract interface for validation components."""
    
    @abstractmethod
    def validate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> ValidationReport:
        """Comprehensive validation of predictions."""
        pass
    
    @abstractmethod
    def cross_validate(self, X: np.ndarray, y: np.ndarray, model: Any) -> Dict[str, float]:
        """Perform cross-validation."""
        pass
    
    @abstractmethod
    def calculate_accuracy_thresholds(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[float, float]:
        """Calculate accuracy at multiple thresholds."""
        pass


class Monitor(ABC):
    """Abstract interface for monitoring components."""
    
    @abstractmethod
    def monitor_prediction(self, prediction: float, actual: float) -> None:
        """Monitor single prediction quality."""
        pass
    
    @abstractmethod
    def check_system_health(self) -> SystemHealthReport:
        """Comprehensive system health check."""
        pass
    
    @abstractmethod
    def detect_model_drift(self) -> float:
        """Detect model performance drift."""
        pass
    
    @abstractmethod
    def should_retrain(self) -> bool:
        """Determine if model retraining is needed."""
        pass


class DataProcessor(ABC):
    """Abstract interface for data processing components."""
    
    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data."""
        pass
    
    @abstractmethod
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality."""
        pass


class ModelTrainer(ABC):
    """Abstract interface for model training components."""
    
    @abstractmethod
    def train_model(self, X: np.ndarray, y: np.ndarray, model_config: Dict[str, Any]) -> Any:
        """Train individual model."""
        pass
    
    @abstractmethod
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        pass


class FeatureSelector(ABC):
    """Abstract interface for feature selection components."""
    
    @abstractmethod
    def select_features(self, X: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, List[int]]:
        """Select top k features."""
        pass
    
    @abstractmethod
    def get_feature_scores(self) -> np.ndarray:
        """Get feature importance scores."""
        pass


class ErrorAnalyzer(ABC):
    """Abstract interface for error analysis components."""
    
    @abstractmethod
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Analyze prediction errors."""
        pass
    
    @abstractmethod
    def calculate_percentage_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate percentage errors with high precision."""
        pass


class Visualizer(ABC):
    """Abstract interface for visualization components."""
    
    @abstractmethod
    def plot_error_distribution(self, errors: np.ndarray) -> None:
        """Plot error distribution focusing on sub-0.5% range."""
        pass
    
    @abstractmethod
    def plot_accuracy_curves(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot cumulative accuracy curves."""
        pass
    
    @abstractmethod
    def generate_report(self, validation_report: ValidationReport) -> str:
        """Generate comprehensive visual report."""
        pass
"""
Core interfaces and abstract base classes for the enhanced time series prediction system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MarketData:
    """Market data structure for OHLCV data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    
    def to_ohlcv(self) -> np.ndarray:
        """Convert to OHLCV numpy array."""
        return np.array([self.open, self.high, self.low, self.close, self.volume])


@dataclass
class PredictionResult:
    """Prediction result with uncertainty quantification."""
    symbol: str
    timestamp: datetime
    prediction: float
    confidence_lower: float
    confidence_upper: float
    uncertainty: float
    model_contributions: Dict[str, float]
    features_used: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'prediction': self.prediction,
            'confidence_lower': self.confidence_lower,
            'confidence_upper': self.confidence_upper,
            'uncertainty': self.uncertainty,
            'model_contributions': self.model_contributions,
            'features_used': self.features_used
        }


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    mae: float
    rmse: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float
    regime_performance: Dict[str, Dict[str, float]]
    
    def summary_stats(self) -> Dict[str, float]:
        """Calculate summary statistics."""
        return {
            'overall_score': self.calculate_composite_score(),
            'risk_adjusted_return': self.sharpe_ratio,
            'consistency': 1.0 - (self.rmse / self.mae) if self.mae > 0 else 0.0
        }
    
    def calculate_composite_score(self) -> float:
        """Calculate composite performance score."""
        # Weighted combination of key metrics
        accuracy_score = max(0, 1.0 - self.mae)
        direction_score = self.directional_accuracy / 100.0
        risk_score = max(0, min(1.0, self.sharpe_ratio / 2.0))
        
        return (0.4 * accuracy_score + 0.3 * direction_score + 0.3 * risk_score)


class BaseModel(nn.Module, ABC):
    """Abstract base class for all prediction models."""
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.model_name = self.__class__.__name__
        self.is_trained = False
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make prediction with uncertainty estimation."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        return {
            'name': self.model_name,
            'input_dim': self.input_dim,
            'num_parameters': sum(p.numel() for p in self.parameters()),
            'is_trained': self.is_trained
        }


class BaseFeatureEngineer(ABC):
    """Abstract base class for feature engineering."""
    
    def __init__(self, **kwargs):
        self.feature_names = []
        self.is_fitted = False
        
    @abstractmethod
    def create_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create features from market data."""
        pass
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BaseFeatureEngineer':
        """Fit the feature engineer on training data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted feature engineer."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()


class BasePredictor(ABC):
    """Abstract base class for predictors."""
    
    def __init__(self, **kwargs):
        self.is_trained = False
        self.feature_engineer = None
        self.models = []
        
    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Train the predictor on data."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make prediction on new data."""
        pass
    
    @abstractmethod
    def predict_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, PredictionResult]:
        """Make batch predictions for multiple assets."""
        pass
    
    def get_predictor_info(self) -> Dict[str, Any]:
        """Get predictor information."""
        return {
            'is_trained': self.is_trained,
            'num_models': len(self.models),
            'feature_engineer': type(self.feature_engineer).__name__ if self.feature_engineer else None
        }


class BaseBacktester(ABC):
    """Abstract base class for backtesting frameworks."""
    
    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, predictor: BasePredictor, **kwargs) -> Dict[str, Any]:
        """Run backtest on historical data."""
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, predictions: List[float], 
                                    actuals: List[float]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        pass


class BaseMonitor(ABC):
    """Abstract base class for performance monitoring."""
    
    @abstractmethod
    def track_prediction(self, prediction: PredictionResult, actual: Optional[float] = None):
        """Track a prediction and optionally its actual outcome."""
        pass
    
    @abstractmethod
    def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        pass
    
    @abstractmethod
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        pass


class ConfigManager:
    """Configuration management for the system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'models': {
                'transformer': {
                    'd_model': 256,
                    'nhead': 16,
                    'num_layers': 8,
                    'dropout': 0.1,
                    'seq_len': 60
                },
                'lstm': {
                    'hidden_dim': 256,
                    'num_layers': 4,
                    'dropout': 0.2,
                    'bidirectional': True
                },
                'cnn_lstm': {
                    'cnn_channels': [64, 128, 256],
                    'lstm_hidden': 256,
                    'kernel_sizes': [3, 5, 7]
                }
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'early_stopping_patience': 20,
                'weight_decay': 1e-5
            },
            'features': {
                'lookback_periods': [5, 10, 20, 50],
                'technical_indicators': True,
                'microstructure': True,
                'cross_asset': True,
                'num_features_target': 25
            },
            'backtesting': {
                'initial_train_size': 252,
                'retraining_frequency': 21,
                'test_size': 21,
                'min_train_size': 126
            },
            'monitoring': {
                'performance_window': 100,
                'alert_thresholds': {
                    'mae_threshold': 0.02,
                    'directional_accuracy_threshold': 0.55,
                    'max_drawdown_threshold': 0.15
                }
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        import json
        save_path = path or self.config_path
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def load(self, path: str):
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            self.config = json.load(f)
        self.config_path = path


# Global configuration instance
config = ConfigManager()
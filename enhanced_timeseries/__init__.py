"""
Enhanced Time Series Prediction System
=====================================

A comprehensive financial market prediction platform that combines multiple 
advanced neural network architectures, sophisticated feature engineering, 
and robust backtesting capabilities for ultra-high accuracy time series forecasting.

Main Components:
- Advanced Model Architectures (Transformer, LSTM, CNN-LSTM)
- Sophisticated Feature Engineering (75+ technical indicators)
- Ensemble Framework with Dynamic Weighting
- Real-time Monitoring & Alerting
- Uncertainty Quantification
- Multi-Asset Support
- Comprehensive Backtesting
"""

__version__ = "1.0.0"
__author__ = "MeridianLearning Team"

# Core imports
from .core.data_processor import DataProcessor
from .core.model_trainer import ModelTrainer
from .core.predictor import Predictor

# Core imports
from .core.data_processor import DataProcessor
from .core.model_trainer import ModelTrainer
from .core.predictor import Predictor

# Ensemble framework
from .ensemble.ensemble_framework import EnsembleFramework

__all__ = [
    # Core
    'DataProcessor',
    'ModelTrainer', 
    'Predictor',
    
    # Ensemble
    'EnsembleFramework',
]
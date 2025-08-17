"""
Ensemble prediction framework for time series forecasting.
"""

from .ensemble_framework import (
    BaseModel,
    ModelWrapper,
    PerformanceTracker,
    WeightingStrategy,
    PerformanceBasedWeighting,
    TrendAwareWeighting,
    EnsembleMethod,
    WeightedAveraging,
    StackingEnsemble,
    EnsemblePredictor,
    create_ensemble_predictor,
    evaluate_ensemble_performance
)

__all__ = [
    'BaseModel',
    'ModelWrapper',
    'PerformanceTracker',
    'WeightingStrategy',
    'PerformanceBasedWeighting',
    'TrendAwareWeighting',
    'EnsembleMethod',
    'WeightedAveraging',
    'StackingEnsemble',
    'EnsemblePredictor',
    'create_ensemble_predictor',
    'evaluate_ensemble_performance'
]
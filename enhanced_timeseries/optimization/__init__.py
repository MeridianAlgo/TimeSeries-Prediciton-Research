"""
Hyperparameter optimization and configuration management system.
"""

from .bayesian_optimizer import (
    Parameter,
    Trial,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    GaussianProcess,
    BayesianOptimizer,
    MultiObjectiveBayesianOptimizer,
    HyperparameterTuner,
    create_parameter_space,
    optimize_model_hyperparameters
)

from .config_manager import (
    ModelConfiguration,
    PerformanceRecord,
    ConfigurationManager,
    create_config_manager,
    auto_save_model_config
)

__all__ = [
    'Parameter',
    'Trial',
    'ExpectedImprovement',
    'UpperConfidenceBound',
    'ProbabilityOfImprovement',
    'GaussianProcess',
    'BayesianOptimizer',
    'MultiObjectiveBayesianOptimizer',
    'HyperparameterTuner',
    'create_parameter_space',
    'optimize_model_hyperparameters',
    'ModelConfiguration',
    'PerformanceRecord',
    'ConfigurationManager',
    'create_config_manager',
    'auto_save_model_config'
]
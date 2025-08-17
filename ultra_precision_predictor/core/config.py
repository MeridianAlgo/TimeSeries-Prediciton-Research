"""Configuration management for ultra-precision predictor."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import os


@dataclass
class UltraPrecisionConfig:
    """Main configuration for ultra-precision predictor system."""
    
    # Core targets
    target_error_rate: float = 0.5
    feature_count_target: int = 500
    outlier_removal_strictness: float = 0.8
    ensemble_model_count: int = 12
    refinement_stages: int = 3
    validation_folds: int = 10
    
    # Feature engineering parameters
    micro_pattern_lookbacks: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 8, 13])
    fractional_ma_periods: List[float] = field(default_factory=lambda: [2.5, 3.7, 5.2, 7.8, 10.3, 13.6, 18.4, 25.1, 34.2])
    bollinger_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    bollinger_multipliers: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.618, 2.0, 2.618])
    rsi_periods: List[int] = field(default_factory=lambda: [5, 7, 11, 14, 19, 25, 31])
    macd_configs: List[tuple] = field(default_factory=lambda: [(8, 17, 9), (12, 26, 9), (19, 39, 9), (5, 13, 8), (21, 55, 13)])
    volatility_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 30])
    
    # Outlier removal parameters
    z_score_threshold: float = 1.0
    modified_z_threshold: float = 1.5
    iqr_percentiles: tuple = (15, 85)
    iqr_multiplier: float = 0.5
    local_window_size: int = 20
    removal_target_percentage: float = 20.0
    
    # Ensemble parameters
    base_models_config: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'rf_extreme_1': {
            'n_estimators': 1500, 'max_depth': 30, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 0.3, 'random_state': 42
        },
        'rf_extreme_2': {
            'n_estimators': 1200, 'max_depth': 25, 'min_samples_split': 3,
            'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': 43
        },
        'rf_extreme_3': {
            'n_estimators': 1000, 'max_depth': 20, 'min_samples_split': 4,
            'min_samples_leaf': 3, 'max_features': 'log2', 'random_state': 44
        },
        'et_extreme_1': {
            'n_estimators': 1500, 'max_depth': 35, 'min_samples_split': 2,
            'min_samples_leaf': 1, 'max_features': 0.4, 'random_state': 45
        },
        'et_extreme_2': {
            'n_estimators': 1000, 'max_depth': 15, 'min_samples_split': 5,
            'min_samples_leaf': 4, 'max_features': 'sqrt', 'random_state': 46
        },
        'gbm_extreme_1': {
            'n_estimators': 2000, 'learning_rate': 0.003, 'max_depth': 10,
            'min_samples_split': 2, 'min_samples_leaf': 1, 'subsample': 0.95, 'random_state': 47
        },
        'gbm_extreme_2': {
            'n_estimators': 1500, 'learning_rate': 0.005, 'max_depth': 8,
            'min_samples_split': 3, 'min_samples_leaf': 2, 'subsample': 0.9, 'random_state': 48
        },
        'gbm_extreme_3': {
            'n_estimators': 1000, 'learning_rate': 0.01, 'max_depth': 6,
            'min_samples_split': 4, 'min_samples_leaf': 3, 'subsample': 0.85, 'random_state': 49
        }
    })
    
    feature_selection_counts: Dict[str, int] = field(default_factory=lambda: {
        'extreme_1': 150, 'extreme_2': 100, 'extreme_3': 80, 'default': 60
    })
    
    # Refinement parameters
    smoothing_windows: List[int] = field(default_factory=lambda: [3, 2])
    outlier_correction_threshold: float = 2.0
    max_prediction_error_threshold: float = 2.0
    
    # Validation parameters
    accuracy_thresholds: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    sub_half_percent_target: float = 50.0
    cv_consistency_threshold: float = 0.2
    mean_cv_error_target: float = 0.8
    
    # Monitoring parameters
    real_time_alert_threshold: float = 60.0
    drift_detection_window: int = 100
    retraining_trigger_threshold: float = 50.0
    
    # System parameters
    random_seed: int = 42
    n_jobs: int = -1
    verbose: bool = True
    log_level: str = "INFO"
    
    @classmethod
    def from_file(cls, config_path: str) -> 'UltraPrecisionConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.target_error_rate <= 0 or self.target_error_rate > 10:
            raise ValueError("target_error_rate must be between 0 and 10")
        
        if self.feature_count_target < 100:
            raise ValueError("feature_count_target must be at least 100")
        
        if self.outlier_removal_strictness < 0 or self.outlier_removal_strictness > 1:
            raise ValueError("outlier_removal_strictness must be between 0 and 1")
        
        if self.ensemble_model_count < 5:
            raise ValueError("ensemble_model_count must be at least 5")
        
        if self.validation_folds < 3:
            raise ValueError("validation_folds must be at least 3")


@dataclass
class PredictionResult:
    """Result of ultra-precision prediction."""
    prediction: float
    confidence: float
    individual_predictions: Dict[str, float]
    refinement_stages: List[float]
    error_estimate: float
    feature_count: int
    model_weights: Dict[str, float]


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    mean_error: float
    median_error: float
    sub_half_percent_rate: float
    accuracy_thresholds: Dict[float, float]
    statistical_significance: float
    cross_validation_consistency: float
    individual_model_errors: Dict[str, float]
    feature_importance: Dict[str, float]


@dataclass
class SystemHealthReport:
    """System health monitoring report."""
    current_error_rate: float
    sub_half_percent_achievement: float
    model_drift_score: float
    recommendation: str
    requires_retraining: bool
    last_update: str
    performance_trend: List[float]
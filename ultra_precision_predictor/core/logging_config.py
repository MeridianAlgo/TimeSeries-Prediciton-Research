"""Logging configuration for ultra-precision predictor system."""

import logging
import sys
from typing import Optional
from datetime import datetime
import os


class UltraPrecisionLogger:
    """Custom logger for ultra-precision predictor system."""
    
    def __init__(self, name: str = "ultra_precision", level: str = "INFO", log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger."""
        return self.logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging for the ultra-precision predictor system."""
    
    # Create logs directory if log_file is specified
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/ultra_precision_{timestamp}.log"
    
    logger_instance = UltraPrecisionLogger(
        name="ultra_precision",
        level=level,
        log_file=log_file
    )
    
    logger = logger_instance.get_logger()
    logger.info("Ultra-precision predictor logging initialized")
    
    return logger


def log_performance_metrics(logger: logging.Logger, metrics: dict, stage: str = ""):
    """Log performance metrics in a structured format."""
    logger.info(f"=== Performance Metrics {stage} ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.6f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * (25 + len(stage)))


def log_feature_statistics(logger: logging.Logger, feature_stats: dict):
    """Log feature engineering statistics."""
    logger.info("=== Feature Engineering Statistics ===")
    logger.info(f"Total features generated: {feature_stats.get('total_features', 'N/A')}")
    logger.info(f"Feature categories: {feature_stats.get('categories', 'N/A')}")
    logger.info(f"Processing time: {feature_stats.get('processing_time', 'N/A'):.2f}s")
    logger.info("=" * 40)


def log_outlier_removal_stats(logger: logging.Logger, removal_stats: dict):
    """Log outlier removal statistics."""
    logger.info("=== Outlier Removal Statistics ===")
    logger.info(f"Original samples: {removal_stats.get('original_count', 'N/A')}")
    logger.info(f"Removed samples: {removal_stats.get('removed_count', 'N/A')}")
    logger.info(f"Removal percentage: {removal_stats.get('removal_percentage', 'N/A'):.2f}%")
    logger.info(f"Remaining samples: {removal_stats.get('remaining_count', 'N/A')}")
    logger.info("=" * 36)


def log_model_performance(logger: logging.Logger, model_name: str, performance: dict):
    """Log individual model performance."""
    logger.info(f"=== Model Performance: {model_name} ===")
    logger.info(f"CV Error: {performance.get('cv_error', 'N/A'):.4f}%")
    logger.info(f"CV Std: {performance.get('cv_std', 'N/A'):.4f}%")
    logger.info(f"Weight: {performance.get('weight', 'N/A'):.4f}")
    logger.info(f"Features: {performance.get('feature_count', 'N/A')}")
    logger.info("=" * (25 + len(model_name)))


def log_validation_results(logger: logging.Logger, validation_report):
    """Log comprehensive validation results."""
    logger.info("=== Ultra-Precision Validation Results ===")
    logger.info(f"Mean Error: {validation_report.mean_error:.4f}%")
    logger.info(f"Median Error: {validation_report.median_error:.4f}%")
    logger.info(f"Sub-0.5% Rate: {validation_report.sub_half_percent_rate:.2f}%")
    
    logger.info("Accuracy Thresholds:")
    for threshold, accuracy in validation_report.accuracy_thresholds.items():
        logger.info(f"  ±{threshold}%: {accuracy:.2f}%")
    
    logger.info(f"Statistical Significance: {validation_report.statistical_significance:.4f}")
    logger.info(f"CV Consistency: {validation_report.cross_validation_consistency:.4f}")
    logger.info("=" * 43)


def log_system_health(logger: logging.Logger, health_report):
    """Log system health monitoring results."""
    logger.info("=== System Health Report ===")
    logger.info(f"Current Error Rate: {health_report.current_error_rate:.4f}%")
    logger.info(f"Sub-0.5% Achievement: {health_report.sub_half_percent_achievement:.2f}%")
    logger.info(f"Model Drift Score: {health_report.model_drift_score:.4f}")
    logger.info(f"Requires Retraining: {health_report.requires_retraining}")
    logger.info(f"Recommendation: {health_report.recommendation}")
    logger.info("=" * 29)
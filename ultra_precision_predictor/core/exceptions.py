"""Custom exceptions for ultra-precision predictor system."""


class UltraPrecisionError(Exception):
    """Base exception for ultra-precision predictor system."""
    pass


class FeatureEngineeringError(UltraPrecisionError):
    """Exception raised during feature engineering process."""
    pass


class OutlierDetectionError(UltraPrecisionError):
    """Exception raised during outlier detection process."""
    pass


class EnsembleError(UltraPrecisionError):
    """Exception raised during ensemble model operations."""
    pass


class RefinementError(UltraPrecisionError):
    """Exception raised during prediction refinement process."""
    pass


class ValidationError(UltraPrecisionError):
    """Exception raised during validation process."""
    pass


class MonitoringError(UltraPrecisionError):
    """Exception raised during monitoring operations."""
    pass


class ConfigurationError(UltraPrecisionError):
    """Exception raised for configuration-related issues."""
    pass


class DataQualityError(UltraPrecisionError):
    """Exception raised for data quality issues."""
    pass


class ModelTrainingError(UltraPrecisionError):
    """Exception raised during model training."""
    pass


class PredictionError(UltraPrecisionError):
    """Exception raised during prediction generation."""
    pass


class VisualizationError(UltraPrecisionError):
    """Exception raised during visualization generation."""
    pass
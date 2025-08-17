"""Custom exceptions for the stock predictor system."""


class StockPredictorError(Exception):
    """Base exception for stock predictor errors."""
    pass


class DataFetchError(StockPredictorError):
    """Raised when data fetching fails."""
    pass


class DataPreprocessingError(StockPredictorError):
    """Raised when data preprocessing fails."""
    pass


class ModelTrainingError(StockPredictorError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(StockPredictorError):
    """Raised when model prediction fails."""
    pass


class EnsembleError(StockPredictorError):
    """Raised when ensemble operations fail."""
    pass


class ConfigurationError(StockPredictorError):
    """Raised when configuration is invalid."""
    pass
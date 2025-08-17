"""Visualization-specific exceptions for the stock prediction system."""

from stock_predictor.utils.exceptions import StockPredictorError


class VisualizationError(StockPredictorError):
    """Base exception for visualization errors."""
    pass


class ChartCreationError(VisualizationError):
    """Raised when chart creation fails."""
    pass


class DashboardBuildError(VisualizationError):
    """Raised when dashboard building fails."""
    pass


class ReportGenerationError(VisualizationError):
    """Raised when report generation fails."""
    pass


class ThemeError(VisualizationError):
    """Raised when theme operations fail."""
    pass


class ExportError(VisualizationError):
    """Raised when export operations fail."""
    pass


class ConfigurationError(VisualizationError):
    """Raised when configuration operations fail."""
    pass


class DataValidationError(VisualizationError):
    """Raised when input data validation fails."""
    pass


class BackendError(VisualizationError):
    """Raised when visualization backend operations fail."""
    pass


class TemplateError(VisualizationError):
    """Raised when template operations fail."""
    pass
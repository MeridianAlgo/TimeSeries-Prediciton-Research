"""Enhanced visualization system for stock price prediction results."""

from .visualizer import Visualizer
from .manager import VisualizationManager
from .base_chart import InteractiveChart
from .price_chart import PriceChart
from .performance_chart import PerformanceChart
from .theme_manager import ThemeManager, Theme
from .chart_factory import ChartFactory
from .dashboard_builder import DashboardBuilder, Dashboard
from .report_generator import ReportGenerator, Report
from .exceptions import (
    VisualizationError,
    ChartCreationError,
    DashboardBuildError,
    ReportGenerationError,
    ThemeError,
    ExportError,
    ConfigurationError,
    DataValidationError,
    BackendError,
    TemplateError
)

__all__ = [
    'Visualizer',  # Legacy visualizer for backward compatibility
    'VisualizationManager',  # New central manager
    'InteractiveChart',
    'PriceChart',
    'PerformanceChart',
    'ThemeManager',
    'Theme',
    'ChartFactory',
    'DashboardBuilder',
    'Dashboard',
    'ReportGenerator',
    'Report',
    'VisualizationError',
    'ChartCreationError',
    'DashboardBuildError',
    'ReportGenerationError',
    'ThemeError',
    'ExportError',
    'ConfigurationError',
    'DataValidationError',
    'BackendError',
    'TemplateError'
]
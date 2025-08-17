"""Chart factory for creating different types of interactive charts."""

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

from stock_predictor.utils.logging import get_logger
from .exceptions import ChartCreationError, ExportError
from .base_chart import InteractiveChart
from .price_chart import PriceChart
from .performance_chart import PerformanceChart


class ChartFactory:
    """Factory for creating different types of charts with consistent theming."""
    
    def __init__(self, theme_manager: Any):
        """
        Initialize chart factory.
        
        Args:
            theme_manager: ThemeManager instance for consistent styling
        """
        self.theme_manager = theme_manager
        self.logger = get_logger('visualization.chart_factory')
        
        # Registry of available chart types
        self.chart_types = {
            'price': PriceChart,
            'performance': PerformanceChart,
            # Additional chart types can be registered here
        }
        
        self.logger.debug(f"Chart factory initialized with {len(self.chart_types)} chart types")
    
    def create_price_chart(self, 
                          historical_data: pd.DataFrame,
                          predictions: Dict[str, np.ndarray],
                          confidence_intervals: Optional[Dict[str, tuple]] = None,
                          ensemble_data: Optional[Dict[str, Any]] = None,
                          config: Optional[Dict[str, Any]] = None) -> PriceChart:
        """
        Create interactive price chart.
        
        Args:
            historical_data: DataFrame with historical price data
            predictions: Dict of {model_name: predictions_array}
            confidence_intervals: Optional confidence intervals
            ensemble_data: Optional ensemble prediction data
            config: Optional chart configuration
            
        Returns:
            PriceChart instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            chart = PriceChart(theme=theme, config=config)
            
            # Add historical data
            if 'close' in historical_data.columns:
                chart.add_historical_data(
                    dates=historical_data.index if hasattr(historical_data.index, 'to_pydatetime') else None,
                    prices=historical_data['close']
                )
            
            # Add model predictions
            if predictions:
                chart.add_predictions(predictions)
            
            # Add confidence intervals
            if confidence_intervals:
                for model_name, (lower, upper) in confidence_intervals.items():
                    chart.add_confidence_bands(f"{model_name}_confidence", lower, upper)
            
            # Add ensemble prediction
            if ensemble_data:
                ensemble_pred = ensemble_data.get('prediction')
                ensemble_confidence = ensemble_data.get('confidence_intervals')
                
                if ensemble_pred is not None:
                    chart.add_ensemble_prediction(ensemble_pred, ensemble_confidence)
            
            self.logger.debug("Created price chart successfully")
            return chart
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create price chart: {str(e)}")
    
    def create_performance_chart(self,
                               model_metrics: Dict[str, Dict[str, float]],
                               chart_subtype: str = 'comparison',
                               config: Optional[Dict[str, Any]] = None) -> PerformanceChart:
        """
        Create performance comparison chart.
        
        Args:
            model_metrics: Dict of {model_name: {metric: value}}
            chart_subtype: Type of performance chart ('comparison', 'time_series', 'residuals')
            config: Optional chart configuration
            
        Returns:
            PerformanceChart instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            chart = PerformanceChart(theme=theme, config=config, subtype=chart_subtype)
            
            # Add model metrics
            chart.add_model_metrics(model_metrics)
            
            self.logger.debug(f"Created {chart_subtype} performance chart successfully")
            return chart
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create performance chart: {str(e)}")
    
    def create_chart(self, 
                    chart_type: str, 
                    data: Dict[str, Any],
                    config: Optional[Dict[str, Any]] = None) -> InteractiveChart:
        """
        Create chart of specified type with provided data.
        
        Args:
            chart_type: Type of chart to create
            data: Data dictionary for the chart
            config: Optional chart configuration
            
        Returns:
            InteractiveChart instance
        """
        if chart_type not in self.chart_types:
            raise ChartCreationError(f"Unknown chart type: {chart_type}")
        
        try:
            theme = self.theme_manager.get_current_theme()
            chart_class = self.chart_types[chart_type]
            chart = chart_class(theme=theme, config=config)
            
            # Add data based on chart type
            if chart_type == 'price':
                self._populate_price_chart(chart, data)
            elif chart_type == 'performance':
                self._populate_performance_chart(chart, data)
            
            self.logger.debug(f"Created {chart_type} chart successfully")
            return chart
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create {chart_type} chart: {str(e)}")
    
    def register_chart_type(self, name: str, chart_class: type) -> None:
        """
        Register a new chart type.
        
        Args:
            name: Name of the chart type
            chart_class: Chart class that extends InteractiveChart
        """
        if not issubclass(chart_class, InteractiveChart):
            raise ChartCreationError(f"Chart class must extend InteractiveChart")
        
        self.chart_types[name] = chart_class
        self.logger.info(f"Registered new chart type: {name}")
    
    def get_available_chart_types(self) -> list:
        """Get list of available chart types."""
        return list(self.chart_types.keys())
    
    def export_chart(self, 
                    chart: InteractiveChart, 
                    filepath: Union[str, Path], 
                    format: str = 'png',
                    config: Optional[Dict[str, Any]] = None) -> None:
        """
        Export chart to file with factory-level configuration.
        
        Args:
            chart: Chart to export
            filepath: Path to save the exported chart
            format: Export format
            config: Optional export configuration
        """
        try:
            # Apply factory-level export settings
            export_config = config or {}
            
            # Add theme-specific export settings
            theme_export_settings = self.theme_manager.get_current_theme().export_settings
            for key, value in theme_export_settings.items():
                if key not in export_config:
                    export_config[key] = value
            
            # Export the chart
            chart.export(filepath, format, **export_config)
            
            self.logger.info(f"Chart exported to: {filepath}")
            
        except Exception as e:
            raise ExportError(f"Failed to export chart: {str(e)}")
    
    def _populate_price_chart(self, chart: PriceChart, data: Dict[str, Any]) -> None:
        """Populate price chart with data."""
        # Add historical data
        historical_data = data.get('historical_data')
        if historical_data is not None and 'close' in historical_data.columns:
            chart.add_historical_data(
                dates=historical_data.index if hasattr(historical_data.index, 'to_pydatetime') else None,
                prices=historical_data['close']
            )
        
        # Add predictions
        predictions = data.get('predictions', {})
        if predictions:
            chart.add_predictions(predictions)
        
        # Add confidence intervals
        confidence_intervals = data.get('confidence_intervals', {})
        if confidence_intervals:
            for model_name, (lower, upper) in confidence_intervals.items():
                chart.add_confidence_bands(f"{model_name}_confidence", lower, upper)
        
        # Add ensemble data
        ensemble_data = data.get('ensemble_data')
        if ensemble_data:
            ensemble_pred = ensemble_data.get('prediction')
            ensemble_confidence = ensemble_data.get('confidence_intervals')
            
            if ensemble_pred is not None:
                chart.add_ensemble_prediction(ensemble_pred, ensemble_confidence)
    
    def _populate_performance_chart(self, chart: PerformanceChart, data: Dict[str, Any]) -> None:
        """Populate performance chart with data."""
        # Add model metrics
        model_metrics = data.get('model_metrics', {})
        if model_metrics:
            chart.add_model_metrics(model_metrics)
        
        # Add time series performance data if available
        time_series_data = data.get('time_series_performance')
        if time_series_data:
            chart.add_time_series_performance(time_series_data)
        
        # Add residual analysis data if available
        residuals = data.get('residuals')
        if residuals:
            chart.add_residual_analysis(residuals)
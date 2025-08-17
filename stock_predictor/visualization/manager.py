"""Central visualization manager for coordinating all visualization components."""

from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from pathlib import Path

from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import StockPredictorError
from .theme_manager import ThemeManager
from .chart_factory import ChartFactory
from .dashboard_builder import DashboardBuilder
from .report_generator import ReportGenerator
from .exceptions import VisualizationError


class VisualizationManager:
    """Central manager for all visualization components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualization manager.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.logger = get_logger('visualization.manager')
        self.config = config
        
        # Initialize core components
        try:
            self.theme_manager = ThemeManager(config.get('themes', {}))
            self.chart_factory = ChartFactory(self.theme_manager)
            self.dashboard_builder = DashboardBuilder(self.chart_factory)
            self.report_generator = ReportGenerator(self.theme_manager)
            
            self.logger.info("Visualization manager initialized successfully")
            
        except Exception as e:
            raise VisualizationError(f"Failed to initialize visualization manager: {str(e)}")
    
    def create_interactive_price_chart(self, 
                                     historical_data: pd.DataFrame,
                                     predictions: Dict[str, np.ndarray],
                                     confidence_intervals: Optional[Dict[str, tuple]] = None,
                                     ensemble_data: Optional[Dict[str, Any]] = None,
                                     **kwargs) -> Any:
        """
        Create interactive price chart with predictions and confidence intervals.
        
        Args:
            historical_data: DataFrame with historical price data
            predictions: Dict of {model_name: predictions_array}
            confidence_intervals: Optional confidence intervals for predictions
            ensemble_data: Optional ensemble prediction data
            **kwargs: Additional chart configuration options
            
        Returns:
            Interactive chart object
        """
        try:
            chart_config = self.config.get('charts', {}).get('price_chart', {})
            chart_config.update(kwargs)
            
            return self.chart_factory.create_price_chart(
                historical_data=historical_data,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                ensemble_data=ensemble_data,
                config=chart_config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create price chart: {str(e)}")
            raise VisualizationError(f"Price chart creation failed: {str(e)}")
    
    def create_performance_dashboard(self, 
                                   model_metrics: Dict[str, Dict[str, float]],
                                   backtest_results: Optional[Dict[str, Any]] = None,
                                   ensemble_weights: Optional[Dict[str, float]] = None,
                                   **kwargs) -> Any:
        """
        Create comprehensive performance dashboard.
        
        Args:
            model_metrics: Dict of {model_name: {metric: value}}
            backtest_results: Optional backtesting results
            ensemble_weights: Optional ensemble model weights
            **kwargs: Additional dashboard configuration options
            
        Returns:
            Dashboard object
        """
        try:
            dashboard_config = self.config.get('dashboard', {})
            dashboard_config.update(kwargs)
            
            return self.dashboard_builder.create_performance_dashboard(
                model_metrics=model_metrics,
                backtest_results=backtest_results,
                ensemble_weights=ensemble_weights,
                config=dashboard_config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create performance dashboard: {str(e)}")
            raise VisualizationError(f"Performance dashboard creation failed: {str(e)}")
    
    def create_model_analysis_dashboard(self,
                                      model_predictions: Dict[str, np.ndarray],
                                      actual_values: np.ndarray,
                                      feature_importance: Optional[Dict[str, np.ndarray]] = None,
                                      residual_analysis: Optional[Dict[str, np.ndarray]] = None,
                                      **kwargs) -> Any:
        """
        Create model analysis dashboard with detailed model insights.
        
        Args:
            model_predictions: Dict of {model_name: predictions}
            actual_values: Array of actual values
            feature_importance: Optional feature importance data
            residual_analysis: Optional residual analysis data
            **kwargs: Additional dashboard configuration options
            
        Returns:
            Dashboard object
        """
        try:
            dashboard_config = self.config.get('dashboard', {})
            dashboard_config.update(kwargs)
            
            return self.dashboard_builder.create_model_analysis_dashboard(
                model_predictions=model_predictions,
                actual_values=actual_values,
                feature_importance=feature_importance,
                residual_analysis=residual_analysis,
                config=dashboard_config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create model analysis dashboard: {str(e)}")
            raise VisualizationError(f"Model analysis dashboard creation failed: {str(e)}")
    
    def generate_comprehensive_report(self,
                                    prediction_results: Dict[str, Any],
                                    output_path: Union[str, Path],
                                    template: str = 'comprehensive',
                                    **kwargs) -> str:
        """
        Generate comprehensive HTML report with all visualizations.
        
        Args:
            prediction_results: Complete prediction results dictionary
            output_path: Path to save the report
            template: Report template to use
            **kwargs: Additional report configuration options
            
        Returns:
            Path to generated report file
        """
        try:
            report_config = self.config.get('reports', {})
            report_config.update(kwargs)
            
            return self.report_generator.create_comprehensive_report(
                results=prediction_results,
                output_path=output_path,
                template=template,
                config=report_config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {str(e)}")
            raise VisualizationError(f"Report generation failed: {str(e)}")
    
    def set_theme(self, theme_name: str) -> None:
        """
        Set the current theme for all visualizations.
        
        Args:
            theme_name: Name of the theme to apply
        """
        try:
            self.theme_manager.set_current_theme(theme_name)
            self.logger.info(f"Theme changed to: {theme_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to set theme: {str(e)}")
            raise VisualizationError(f"Theme setting failed: {str(e)}")
    
    def get_available_themes(self) -> List[str]:
        """
        Get list of available themes.
        
        Returns:
            List of available theme names
        """
        return self.theme_manager.get_available_themes()
    
    def export_chart(self, 
                    chart: Any, 
                    filepath: Union[str, Path], 
                    format: str = 'png',
                    **kwargs) -> None:
        """
        Export chart to file in specified format.
        
        Args:
            chart: Chart object to export
            filepath: Path to save the exported chart
            format: Export format ('png', 'svg', 'pdf', 'html')
            **kwargs: Additional export options
        """
        try:
            export_config = self.config.get('export', {})
            export_config.update(kwargs)
            
            self.chart_factory.export_chart(
                chart=chart,
                filepath=filepath,
                format=format,
                config=export_config
            )
            
            self.logger.info(f"Chart exported to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export chart: {str(e)}")
            raise VisualizationError(f"Chart export failed: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current visualization configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update visualization configuration.
        
        Args:
            new_config: New configuration to merge with existing
        """
        try:
            self.config.update(new_config)
            
            # Update component configurations
            if 'themes' in new_config:
                self.theme_manager.update_config(new_config['themes'])
            
            self.logger.info("Visualization configuration updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {str(e)}")
            raise VisualizationError(f"Configuration update failed: {str(e)}")
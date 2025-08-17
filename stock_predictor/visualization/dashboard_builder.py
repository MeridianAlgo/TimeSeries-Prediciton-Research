"""Dashboard builder for creating comprehensive visualization dashboards."""

from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from stock_predictor.utils.logging import get_logger
from .exceptions import DashboardBuildError


class DashboardLayout:
    """Dashboard layout configuration and management."""
    
    def __init__(self, layout_type: str = 'grid', **kwargs):
        """
        Initialize dashboard layout.
        
        Args:
            layout_type: Type of layout ('grid', 'tabs', 'accordion', 'flex')
            **kwargs: Additional layout parameters
        """
        self.layout_type = layout_type
        self.config = kwargs
        self.responsive_breakpoints = {
            'mobile': 768,
            'tablet': 1024,
            'desktop': 1440
        }
        
    def get_grid_config(self, num_charts: int) -> Dict[str, Any]:
        """Get optimal grid configuration for number of charts."""
        if num_charts <= 1:
            return {'rows': 1, 'cols': 1}
        elif num_charts <= 2:
            return {'rows': 1, 'cols': 2}
        elif num_charts <= 4:
            return {'rows': 2, 'cols': 2}
        elif num_charts <= 6:
            return {'rows': 2, 'cols': 3}
        elif num_charts <= 9:
            return {'rows': 3, 'cols': 3}
        else:
            # For more charts, use dynamic calculation
            cols = int(np.ceil(np.sqrt(num_charts)))
            rows = int(np.ceil(num_charts / cols))
            return {'rows': rows, 'cols': cols}
    
    def get_responsive_config(self, screen_width: int) -> Dict[str, Any]:
        """Get responsive configuration based on screen width."""
        if screen_width <= self.responsive_breakpoints['mobile']:
            return {'layout': 'accordion', 'cols': 1}
        elif screen_width <= self.responsive_breakpoints['tablet']:
            return {'layout': 'tabs', 'cols': 2}
        else:
            return {'layout': self.layout_type, 'cols': self.config.get('cols', 3)}


class Dashboard:
    """Container for multiple charts with advanced layout management."""
    
    def __init__(self, title: str, layout: Union[str, DashboardLayout] = 'grid', 
                 theme: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dashboard.
        
        Args:
            title: Dashboard title
            layout: Layout configuration (string or DashboardLayout object)
            theme: Theme object for styling
            config: Optional configuration dictionary
        """
        self.title = title
        self.layout = layout if isinstance(layout, DashboardLayout) else DashboardLayout(layout)
        self.theme = theme
        self.config = config or {}
        
        # Dashboard components
        self.charts = []
        self.summary_panels = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'title': title,
            'version': '1.0'
        }
        
        # Layout state
        self.current_screen_width = self.config.get('screen_width', 1440)
        self.is_responsive = self.config.get('responsive', True)
        
        self.logger = get_logger('visualization.dashboard')
        
    def add_chart(self, chart: Any, position: Optional[Tuple[int, int]] = None, 
                  size: Optional[Tuple[int, int]] = None, title: Optional[str] = None) -> None:
        """
        Add chart to dashboard with positioning and sizing.
        
        Args:
            chart: Chart object to add
            position: Optional (row, col) position
            size: Optional (width, height) size in grid units
            title: Optional chart title override
        """
        chart_info = {
            'chart': chart,
            'position': position or (len(self.charts) // self.layout.config.get('cols', 2), 
                                   len(self.charts) % self.layout.config.get('cols', 2)),
            'size': size or (1, 1),
            'title': title or getattr(chart, 'title', f'Chart {len(self.charts) + 1}'),
            'id': f'chart_{len(self.charts)}',
            'visible': True
        }
        
        self.charts.append(chart_info)
        self.logger.debug(f"Added chart '{chart_info['title']}' at position {chart_info['position']}")
    
    def add_summary_panel(self, summary: Dict[str, Any], position: str = 'top') -> None:
        """
        Add summary panel with key metrics.
        
        Args:
            summary: Summary data dictionary
            position: Panel position ('top', 'bottom', 'left', 'right')
        """
        panel_info = {
            'data': summary,
            'position': position,
            'id': f'summary_{len(self.summary_panels)}',
            'visible': True
        }
        
        self.summary_panels.append(panel_info)
        self.logger.debug(f"Added summary panel at position '{position}'")
    
    def set_layout(self, layout_config: Dict[str, Any]) -> None:
        """
        Set layout configuration.
        
        Args:
            layout_config: Layout configuration dictionary
        """
        if isinstance(layout_config, dict):
            layout_type = layout_config.get('type', 'grid')
            self.layout = DashboardLayout(layout_type, **layout_config)
        
        self.metadata['layout_updated'] = datetime.now().isoformat()
        self.logger.debug(f"Updated layout to: {self.layout.layout_type}")
    
    def set_responsive_mode(self, enabled: bool, screen_width: Optional[int] = None) -> None:
        """
        Enable/disable responsive mode and set screen width.
        
        Args:
            enabled: Whether to enable responsive mode
            screen_width: Optional screen width for responsive calculations
        """
        self.is_responsive = enabled
        if screen_width:
            self.current_screen_width = screen_width
        
        if enabled:
            responsive_config = self.layout.get_responsive_config(self.current_screen_width)
            self.layout.config.update(responsive_config)
        
        self.logger.debug(f"Responsive mode: {'enabled' if enabled else 'disabled'}")
    
    def toggle_chart_visibility(self, chart_id: str, visible: bool) -> None:
        """
        Toggle chart visibility.
        
        Args:
            chart_id: Chart ID to toggle
            visible: Whether chart should be visible
        """
        for chart_info in self.charts:
            if chart_info['id'] == chart_id:
                chart_info['visible'] = visible
                self.logger.debug(f"Chart '{chart_id}' visibility set to {visible}")
                break
    
    def get_layout_html(self) -> str:
        """Generate HTML layout for the dashboard."""
        if self.layout.layout_type == 'grid':
            return self._generate_grid_html()
        elif self.layout.layout_type == 'tabs':
            return self._generate_tabs_html()
        elif self.layout.layout_type == 'accordion':
            return self._generate_accordion_html()
        else:
            return self._generate_flex_html()
    
    def _generate_grid_html(self) -> str:
        """Generate grid layout HTML."""
        visible_charts = [c for c in self.charts if c['visible']]
        grid_config = self.layout.get_grid_config(len(visible_charts))
        
        html = f"""
        <div class="dashboard-container" style="display: grid; 
             grid-template-columns: repeat({grid_config['cols']}, 1fr);
             grid-template-rows: repeat({grid_config['rows']}, 1fr);
             gap: 20px; padding: 20px;">
        """
        
        for chart_info in visible_charts:
            html += f"""
            <div class="chart-container" id="{chart_info['id']}" 
                 style="grid-column: span {chart_info['size'][0]}; 
                        grid-row: span {chart_info['size'][1]};">
                <h3>{chart_info['title']}</h3>
                <div class="chart-content">
                    <!-- Chart content will be inserted here -->
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_tabs_html(self) -> str:
        """Generate tabs layout HTML."""
        html = """
        <div class="dashboard-tabs">
            <div class="tab-headers">
        """
        
        for i, chart_info in enumerate([c for c in self.charts if c['visible']]):
            active_class = "active" if i == 0 else ""
            html += f"""
                <button class="tab-header {active_class}" 
                        onclick="showTab('{chart_info['id']}')">{chart_info['title']}</button>
            """
        
        html += "</div><div class='tab-contents'>"
        
        for i, chart_info in enumerate([c for c in self.charts if c['visible']]):
            display_style = "block" if i == 0 else "none"
            html += f"""
            <div class="tab-content" id="{chart_info['id']}" style="display: {display_style};">
                <div class="chart-content">
                    <!-- Chart content will be inserted here -->
                </div>
            </div>
            """
        
        html += "</div></div>"
        return html
    
    def _generate_accordion_html(self) -> str:
        """Generate accordion layout HTML."""
        html = '<div class="dashboard-accordion">'
        
        for i, chart_info in enumerate([c for c in self.charts if c['visible']]):
            expanded = "expanded" if i == 0 else ""
            html += f"""
            <div class="accordion-item {expanded}">
                <div class="accordion-header" onclick="toggleAccordion('{chart_info['id']}')">
                    <h3>{chart_info['title']}</h3>
                    <span class="accordion-icon">▼</span>
                </div>
                <div class="accordion-content" id="{chart_info['id']}">
                    <div class="chart-content">
                        <!-- Chart content will be inserted here -->
                    </div>
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def _generate_flex_html(self) -> str:
        """Generate flexible layout HTML."""
        html = """
        <div class="dashboard-flex" style="display: flex; flex-wrap: wrap; gap: 20px; padding: 20px;">
        """
        
        for chart_info in [c for c in self.charts if c['visible']]:
            flex_basis = f"{100 // self.layout.config.get('cols', 2)}%"
            html += f"""
            <div class="chart-container" id="{chart_info['id']}" 
                 style="flex: 1 1 {flex_basis}; min-width: 300px;">
                <h3>{chart_info['title']}</h3>
                <div class="chart-content">
                    <!-- Chart content will be inserted here -->
                </div>
            </div>
            """
        
        html += "</div>"
        return html
    
    def export_dashboard(self, format: str, path: Union[str, Path]) -> None:
        """
        Export dashboard to file.
        
        Args:
            format: Export format ('html', 'json', 'pdf')
            path: Export file path
        """
        path = Path(path)
        
        try:
            if format.lower() == 'html':
                self._export_html(path)
            elif format.lower() == 'json':
                self._export_json(path)
            elif format.lower() == 'pdf':
                self._export_pdf(path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Dashboard exported to: {path}")
            
        except Exception as e:
            raise DashboardBuildError(f"Failed to export dashboard: {str(e)}")
    
    def _export_html(self, path: Path) -> None:
        """Export dashboard as HTML file."""
        layout_html = self.get_layout_html()
        
        # Generate summary panels HTML
        summary_html = ""
        for panel in self.summary_panels:
            if panel['visible']:
                summary_html += self._generate_summary_panel_html(panel)
        
        # Complete HTML document
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.title}</title>
            <style>
                {self._get_dashboard_css()}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>{self.title}</h1>
                <div class="dashboard-metadata">
                    Created: {self.metadata.get('created_at', 'Unknown')}
                </div>
            </div>
            
            {summary_html}
            
            <div class="dashboard-main">
                {layout_html}
            </div>
            
            <script>
                {self._get_dashboard_js()}
            </script>
        </body>
        </html>
        """
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _export_json(self, path: Path) -> None:
        """Export dashboard configuration as JSON."""
        dashboard_data = {
            'title': self.title,
            'layout': {
                'type': self.layout.layout_type,
                'config': self.layout.config
            },
            'charts': [
                {
                    'id': chart['id'],
                    'title': chart['title'],
                    'position': chart['position'],
                    'size': chart['size'],
                    'visible': chart['visible']
                }
                for chart in self.charts
            ],
            'summary_panels': self.summary_panels,
            'metadata': self.metadata,
            'config': self.config
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
    
    def _export_pdf(self, path: Path) -> None:
        """Export dashboard as PDF (placeholder implementation)."""
        # This would require additional libraries like weasyprint or reportlab
        self.logger.warning("PDF export not yet implemented")
        raise NotImplementedError("PDF export requires additional dependencies")
    
    def _generate_summary_panel_html(self, panel: Dict[str, Any]) -> str:
        """Generate HTML for summary panel."""
        position_class = f"summary-{panel['position']}"
        
        html = f'<div class="summary-panel {position_class}" id="{panel['id']}">'
        html += '<div class="summary-content">'
        
        for key, value in panel['data'].items():
            html += f'<div class="summary-item"><span class="label">{key}:</span> <span class="value">{value}</span></div>'
        
        html += '</div></div>'
        return html
    
    def _get_dashboard_css(self) -> str:
        """Get CSS styles for dashboard."""
        return """
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; }
        .dashboard-header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .dashboard-header h1 { margin: 0; }
        .dashboard-metadata { font-size: 0.9em; opacity: 0.8; margin-top: 5px; }
        .summary-panel { background: white; margin: 10px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary-content { display: flex; flex-wrap: wrap; gap: 20px; }
        .summary-item { display: flex; align-items: center; }
        .summary-item .label { font-weight: bold; margin-right: 5px; }
        .summary-item .value { color: #3498db; }
        .dashboard-main { padding: 20px; }
        .chart-container { background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 15px; }
        .chart-container h3 { margin-top: 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        .tab-headers { display: flex; background: #ecf0f1; border-radius: 5px 5px 0 0; }
        .tab-header { flex: 1; padding: 15px; background: none; border: none; cursor: pointer; font-size: 16px; }
        .tab-header.active { background: #3498db; color: white; }
        .tab-contents { background: white; border-radius: 0 0 5px 5px; }
        .tab-content { padding: 20px; }
        .accordion-item { margin-bottom: 10px; }
        .accordion-header { background: #3498db; color: white; padding: 15px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
        .accordion-content { background: white; padding: 20px; display: none; }
        .accordion-item.expanded .accordion-content { display: block; }
        .accordion-item.expanded .accordion-icon { transform: rotate(180deg); }
        """
    
    def _get_dashboard_js(self) -> str:
        """Get JavaScript for dashboard interactivity."""
        return """
        function showTab(tabId) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.style.display = 'none');
            
            // Remove active class from all headers
            const headers = document.querySelectorAll('.tab-header');
            headers.forEach(header => header.classList.remove('active'));
            
            // Show selected tab and activate header
            document.getElementById(tabId).style.display = 'block';
            event.target.classList.add('active');
        }
        
        function toggleAccordion(itemId) {
            const item = document.getElementById(itemId).parentElement;
            item.classList.toggle('expanded');
        }
        
        // Responsive handling
        window.addEventListener('resize', function() {
            const width = window.innerWidth;
            // Add responsive behavior here
        });
        """


class DashboardBuilder:
    """Builds comprehensive dashboards from data and configuration."""
    
    def __init__(self, chart_factory: Any, theme_manager: Any):
        """
        Initialize dashboard builder.
        
        Args:
            chart_factory: ChartFactory instance for creating charts
            theme_manager: ThemeManager instance for styling
        """
        self.chart_factory = chart_factory
        self.theme_manager = theme_manager
        self.logger = get_logger('visualization.dashboard_builder')
    
    def create_performance_dashboard(self,
                                   model_metrics: Dict[str, Dict[str, float]],
                                   time_series_data: Optional[Dict[str, pd.Series]] = None,
                                   ensemble_weights: Optional[Dict[str, float]] = None,
                                   residuals: Optional[Dict[str, np.ndarray]] = None,
                                   config: Optional[Dict[str, Any]] = None) -> Dashboard:
        """
        Create comprehensive performance dashboard.
        
        Args:
            model_metrics: Model performance metrics
            time_series_data: Optional time series performance data
            ensemble_weights: Optional ensemble weights
            residuals: Optional residual data
            config: Optional configuration
            
        Returns:
            Dashboard instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            dashboard = Dashboard("Model Performance Dashboard", "grid", theme, config)
            
            # Create performance comparison chart
            perf_chart = self.chart_factory.create_performance_chart(
                model_metrics, 'comparison', config
            )
            perf_chart.add_model_metrics(model_metrics)
            dashboard.add_chart(perf_chart, (0, 0), (2, 1), "Performance Comparison")
            
            # Add time series performance if available
            if time_series_data:
                ts_chart = self.chart_factory.create_performance_chart(
                    {}, 'time_series', config
                )
                dashboard.add_chart(ts_chart, (0, 1), (2, 1), "Performance Over Time")
            
            # Add residual analysis if available
            if residuals:
                residual_chart = self.chart_factory.create_performance_chart(
                    {}, 'residuals', config
                )
                dashboard.add_chart(residual_chart, (1, 0), (1, 2), "Residual Analysis")
            
            # Create correlation matrix if multiple models
            if len(model_metrics) > 1:
                corr_chart = self.chart_factory.create_performance_chart(
                    {}, 'comparison', config
                )
                dashboard.add_chart(corr_chart, (2, 0), (1, 1), "Model Correlations")
            
            # Add radar chart for multi-dimensional comparison
            radar_chart = self.chart_factory.create_performance_chart(
                model_metrics, 'comparison', config
            )
            dashboard.add_chart(radar_chart, (2, 1), (1, 1), "Performance Radar")
            
            # Add summary panel
            if model_metrics:
                best_model = min(model_metrics.items(), 
                               key=lambda x: x[1].get('rmse', float('inf')))
                worst_model = max(model_metrics.items(), 
                                key=lambda x: x[1].get('rmse', 0))
                
                avg_rmse = np.mean([m.get('rmse', 0) for m in model_metrics.values()])
                avg_r2 = np.mean([m.get('r2', 0) for m in model_metrics.values()])
                
                summary = {
                    'Total Models': len(model_metrics),
                    'Best Model': best_model[0],
                    'Best RMSE': f"${best_model[1].get('rmse', 0):.3f}",
                    'Worst RMSE': f"${worst_model[1].get('rmse', 0):.3f}",
                    'Average RMSE': f"${avg_rmse:.3f}",
                    'Average R²': f"{avg_r2:.3f}",
                    'Performance Spread': f"${worst_model[1].get('rmse', 0) - best_model[1].get('rmse', 0):.3f}"
                }
                dashboard.add_summary_panel(summary, 'top')
            
            self.logger.debug("Created comprehensive performance dashboard")
            return dashboard
            
        except Exception as e:
            raise DashboardBuildError(f"Failed to create performance dashboard: {str(e)}")
    
    def create_prediction_dashboard(self,
                                  historical_data: pd.DataFrame,
                                  predictions: Dict[str, np.ndarray],
                                  ensemble_prediction: Optional[np.ndarray] = None,
                                  confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                  technical_indicators: Optional[Dict[str, np.ndarray]] = None,
                                  config: Optional[Dict[str, Any]] = None) -> Dashboard:
        """
        Create prediction dashboard with price charts and analysis.
        
        Args:
            historical_data: Historical price data
            predictions: Model predictions
            ensemble_prediction: Optional ensemble prediction
            confidence_intervals: Optional confidence intervals
            technical_indicators: Optional technical indicators
            config: Optional configuration
            
        Returns:
            Dashboard instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            dashboard = Dashboard("Stock Prediction Dashboard", "grid", theme, config)
            
            # Main price chart
            price_chart = self.chart_factory.create_price_chart(
                historical_data=historical_data,
                predictions=predictions,
                confidence_intervals={'ensemble': confidence_intervals} if confidence_intervals else None,
                ensemble_data={'prediction': ensemble_prediction, 'confidence_intervals': confidence_intervals} if ensemble_prediction is not None else None,
                config=config
            )
            
            # Add technical indicators if available
            if technical_indicators:
                price_chart.add_technical_indicators(technical_indicators)
            
            dashboard.add_chart(price_chart, (0, 0), (2, 2), "Price Predictions")
            
            # Add individual model performance comparison
            if len(predictions) > 1:
                # Calculate simple metrics for each model (assuming we have actual values)
                if 'close' in historical_data.columns:
                    actual_values = historical_data['close'].values[-len(list(predictions.values())[0]):]
                    model_metrics = {}
                    
                    for model_name, pred in predictions.items():
                        if len(pred) == len(actual_values):
                            rmse = np.sqrt(np.mean((actual_values - pred) ** 2))
                            mae = np.mean(np.abs(actual_values - pred))
                            model_metrics[model_name] = {'rmse': rmse, 'mae': mae}
                    
                    if model_metrics:
                        perf_chart = self.chart_factory.create_performance_chart(
                            model_metrics, 'comparison', config
                        )
                        dashboard.add_chart(perf_chart, (0, 2), (1, 1), "Model Performance")
            
            # Add prediction accuracy summary
            if ensemble_prediction is not None and 'close' in historical_data.columns:
                actual_values = historical_data['close'].values[-len(ensemble_prediction):]
                if len(actual_values) == len(ensemble_prediction):
                    errors = np.abs((actual_values - ensemble_prediction) / actual_values) * 100
                    
                    summary = {
                        'Predictions Made': len(ensemble_prediction),
                        'Mean Error': f"{np.mean(errors):.2f}%",
                        'Median Error': f"{np.median(errors):.2f}%",
                        'Max Error': f"{np.max(errors):.2f}%",
                        'Accuracy (±1%)': f"{np.mean(errors <= 1.0) * 100:.1f}%",
                        'Accuracy (±2%)': f"{np.mean(errors <= 2.0) * 100:.1f}%",
                        'Accuracy (±3%)': f"{np.mean(errors <= 3.0) * 100:.1f}%"
                    }
                    dashboard.add_summary_panel(summary, 'top')
            
            self.logger.debug("Created prediction dashboard")
            return dashboard
            
        except Exception as e:
            raise DashboardBuildError(f"Failed to create prediction dashboard: {str(e)}")
    
    def create_model_analysis_dashboard(self,
                                      model_predictions: Dict[str, np.ndarray],
                                      actual_values: np.ndarray,
                                      feature_importance: Optional[Dict[str, Dict[str, float]]] = None,
                                      residual_analysis: Optional[Dict[str, np.ndarray]] = None,
                                      config: Optional[Dict[str, Any]] = None) -> Dashboard:
        """
        Create comprehensive model analysis dashboard.
        
        Args:
            model_predictions: Model predictions
            actual_values: Actual values
            feature_importance: Optional feature importance data
            residual_analysis: Optional residual analysis
            config: Optional configuration
            
        Returns:
            Dashboard instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            dashboard = Dashboard("Model Analysis Dashboard", "tabs", theme, config)
            
            # Calculate model metrics
            model_metrics = {}
            for model_name, predictions in model_predictions.items():
                if len(predictions) == len(actual_values):
                    rmse = np.sqrt(np.mean((actual_values - predictions) ** 2))
                    mae = np.mean(np.abs(actual_values - predictions))
                    r2 = 1 - (np.sum((actual_values - predictions) ** 2) / 
                             np.sum((actual_values - np.mean(actual_values)) ** 2))
                    
                    model_metrics[model_name] = {
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    }
            
            # Model performance comparison
            if model_metrics:
                perf_chart = self.chart_factory.create_performance_chart(
                    model_metrics, 'comparison', config
                )
                dashboard.add_chart(perf_chart, title="Performance Comparison")
            
            # Residual analysis
            if residual_analysis:
                residual_chart = self.chart_factory.create_performance_chart(
                    {}, 'residuals', config
                )
                dashboard.add_chart(residual_chart, title="Residual Analysis")
            
            # Prediction error distribution
            errors = {name: actual_values - pred for name, pred in model_predictions.items()
                     if len(pred) == len(actual_values)}
            if errors:
                error_chart = self.chart_factory.create_performance_chart(
                    {}, 'comparison', config
                )
                dashboard.add_chart(error_chart, title="Error Distribution")
            
            # Q-Q plots for normality testing
            if residual_analysis:
                qq_chart = self.chart_factory.create_performance_chart(
                    {}, 'comparison', config
                )
                dashboard.add_chart(qq_chart, title="Q-Q Plots")
            
            # Correlation matrix
            if len(model_predictions) > 1:
                corr_chart = self.chart_factory.create_performance_chart(
                    {}, 'comparison', config
                )
                dashboard.add_chart(corr_chart, title="Model Correlations")
            
            # Feature importance (if available)
            if feature_importance:
                # Create feature importance chart (simplified)
                # In a full implementation, this would be a dedicated chart type
                dashboard.metadata['feature_importance'] = feature_importance
            
            # Add summary panel
            if model_metrics:
                best_model = min(model_metrics.items(), key=lambda x: x[1]['rmse'])
                summary = {
                    'Models Analyzed': len(model_metrics),
                    'Best Model': best_model[0],
                    'Best RMSE': f"${best_model[1]['rmse']:.3f}",
                    'Best R²': f"{best_model[1]['r2']:.3f}",
                    'Data Points': len(actual_values)
                }
                dashboard.add_summary_panel(summary, 'top')
            
            self.logger.debug("Created model analysis dashboard")
            return dashboard
            
        except Exception as e:
            raise DashboardBuildError(f"Failed to create model analysis dashboard: {str(e)}")
    
    def create_ensemble_dashboard(self,
                                ensemble_weights: Dict[str, float],
                                weight_evolution: Optional[Dict[str, pd.Series]] = None,
                                ensemble_performance: Optional[Dict[str, float]] = None,
                                individual_performance: Optional[Dict[str, Dict[str, float]]] = None,
                                config: Optional[Dict[str, Any]] = None) -> Dashboard:
        """
        Create ensemble analysis dashboard.
        
        Args:
            ensemble_weights: Current ensemble weights
            weight_evolution: Optional weight evolution over time
            ensemble_performance: Optional ensemble performance metrics
            individual_performance: Optional individual model performance
            config: Optional configuration
            
        Returns:
            Dashboard instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            dashboard = Dashboard("Ensemble Analysis Dashboard", "grid", theme, config)
            
            # Weight allocation pie chart (placeholder - would need actual pie chart implementation)
            # For now, we'll use a performance chart to show weights
            if ensemble_weights:
                weight_data = {model: {'weight': weight} for model, weight in ensemble_weights.items()}
                weight_chart = self.chart_factory.create_performance_chart(
                    weight_data, 'comparison', config
                )
                dashboard.add_chart(weight_chart, (0, 0), (1, 1), "Ensemble Weights")
            
            # Weight evolution over time
            if weight_evolution:
                evolution_chart = self.chart_factory.create_performance_chart(
                    {}, 'time_series', config
                )
                dashboard.add_chart(evolution_chart, (0, 1), (1, 1), "Weight Evolution")
            
            # Performance comparison
            if individual_performance:
                perf_chart = self.chart_factory.create_performance_chart(
                    individual_performance, 'comparison', config
                )
                dashboard.add_chart(perf_chart, (1, 0), (1, 2), "Individual vs Ensemble Performance")
            
            # Add summary panel
            summary = {
                'Ensemble Models': len(ensemble_weights),
                'Dominant Model': max(ensemble_weights.items(), key=lambda x: x[1])[0],
                'Max Weight': f"{max(ensemble_weights.values()):.3f}",
                'Min Weight': f"{min(ensemble_weights.values()):.3f}",
                'Weight Diversity': f"{np.std(list(ensemble_weights.values())):.3f}"
            }
            
            if ensemble_performance:
                summary.update({
                    'Ensemble RMSE': f"${ensemble_performance.get('rmse', 0):.3f}",
                    'Ensemble R²': f"{ensemble_performance.get('r2', 0):.3f}"
                })
            
            dashboard.add_summary_panel(summary, 'top')
            
            self.logger.debug("Created ensemble analysis dashboard")
            return dashboard
            
        except Exception as e:
            raise DashboardBuildError(f"Failed to create ensemble dashboard: {str(e)}")
    
    def create_custom_dashboard(self,
                              title: str,
                              charts: List[Dict[str, Any]],
                              layout: str = 'grid',
                              summary_data: Optional[Dict[str, Any]] = None,
                              config: Optional[Dict[str, Any]] = None) -> Dashboard:
        """
        Create custom dashboard from chart specifications.
        
        Args:
            title: Dashboard title
            charts: List of chart specifications
            layout: Layout type
            summary_data: Optional summary panel data
            config: Optional configuration
            
        Returns:
            Dashboard instance
        """
        try:
            theme = self.theme_manager.get_current_theme()
            dashboard = Dashboard(title, layout, theme, config)
            
            # Add charts from specifications
            for chart_spec in charts:
                chart_type = chart_spec.get('type', 'performance')
                chart_data = chart_spec.get('data', {})
                chart_config = chart_spec.get('config', {})
                chart_title = chart_spec.get('title', 'Chart')
                position = chart_spec.get('position')
                size = chart_spec.get('size')
                
                # Create chart based on type
                if chart_type == 'performance':
                    metrics = chart_data.get('metrics', {})
                    chart = self.chart_factory.create_performance_chart(
                        metrics, 'comparison', chart_config
                    )
                elif chart_type == 'price':
                    # Create a minimal DataFrame for historical data
                    historical_data = pd.DataFrame()
                    if 'historical_data' in chart_data:
                        historical_data = pd.DataFrame({'close': chart_data['historical_data']})
                    
                    predictions = chart_data.get('predictions', {})
                    chart = self.chart_factory.create_price_chart(
                        historical_data=historical_data,
                        predictions=predictions,
                        config=chart_config
                    )
                else:
                    # Default to performance chart
                    chart = self.chart_factory.create_performance_chart(
                        {}, 'comparison', chart_config
                    )
                
                dashboard.add_chart(chart, position, size, chart_title)
            
            # Add summary panel if provided
            if summary_data:
                dashboard.add_summary_panel(summary_data, 'top')
            
            self.logger.debug(f"Created custom dashboard: {title}")
            return dashboard
            
        except Exception as e:
            raise DashboardBuildError(f"Failed to create custom dashboard: {str(e)}")
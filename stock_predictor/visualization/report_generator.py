"""Report generator for creating comprehensive HTML reports."""

from typing import Dict, Any, Union, List, Optional
from pathlib import Path
import json
from datetime import datetime

from stock_predictor.utils.logging import get_logger
from .exceptions import ReportGenerationError, TemplateError


class Report:
    """Professional report with embedded visualizations."""
    
    def __init__(self, title: str, template: str = 'default'):
        """
        Initialize report.
        
        Args:
            title: Report title
            template: Template name to use
        """
        self.title = title
        self.template = template
        self.sections = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'title': title,
            'template': template
        }
    
    def add_executive_summary(self, summary: Dict[str, Any]) -> None:
        """Add executive summary section."""
        self.sections.append({
            'type': 'executive_summary',
            'content': summary
        })
    
    def add_dashboard(self, dashboard: Any) -> None:
        """Add dashboard section."""
        self.sections.append({
            'type': 'dashboard',
            'content': dashboard
        })
    
    def add_analysis_section(self, title: str, content: str, charts: List[Any]) -> None:
        """Add analysis section with charts."""
        self.sections.append({
            'type': 'analysis',
            'title': title,
            'content': content,
            'charts': charts
        })
    
    def add_appendix(self, data: Dict[str, Any]) -> None:
        """Add appendix section."""
        self.sections.append({
            'type': 'appendix',
            'content': data
        })
    
    def export(self, format: str, path: str) -> str:
        """Export report to file."""
        # Implementation would depend on format
        return path


class ReportGenerator:
    """Generates comprehensive reports with multiple output formats."""
    
    def __init__(self, theme_manager: Any):
        """
        Initialize report generator.
        
        Args:
            theme_manager: ThemeManager instance for styling
        """
        self.theme_manager = theme_manager
        self.logger = get_logger('visualization.report_generator')
        
        # Report templates
        self.templates = {
            'comprehensive': self._get_comprehensive_template(),
            'summary': self._get_summary_template(),
            'technical': self._get_technical_template()
        }
    
    def create_comprehensive_report(self,
                                  results: Dict[str, Any],
                                  output_path: Union[str, Path],
                                  template: str = 'comprehensive',
                                  config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create comprehensive report.
        
        Args:
            results: Complete prediction results
            output_path: Path to save report
            template: Template to use
            config: Optional configuration
            
        Returns:
            Path to generated report
        """
        try:
            report = Report("Stock Price Prediction Report", template)
            
            # Add executive summary
            if 'ensemble_metrics' in results:
                summary = {
                    'symbol': results.get('symbol', 'Unknown'),
                    'ensemble_rmse': results['ensemble_metrics'].get('rmse', 0),
                    'ensemble_accuracy': results['ensemble_metrics'].get('directional_accuracy', 0),
                    'models_trained': len(results.get('model_metrics', {}))
                }
                report.add_executive_summary(summary)
            
            # Generate HTML report
            html_content = self._generate_html_report(report, results, config)
            
            # Save to file
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Report generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            raise ReportGenerationError(f"Failed to create comprehensive report: {str(e)}")
    
    def _generate_html_report(self, report: Report, results: Dict[str, Any], 
                            config: Optional[Dict[str, Any]]) -> str:
        """Generate HTML content for report."""
        theme = self.theme_manager.get_current_theme()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: {theme.get_font_config('axis')['family']};
                    margin: 40px;
                    background-color: {theme.get_color('background')};
                    color: {theme.get_color('text')};
                }}
                .header {{
                    text-align: center;
                    color: {theme.get_color('primary')};
                    border-bottom: 2px solid {theme.get_color('primary')};
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    border-left: 4px solid {theme.get_color('secondary')};
                    background-color: rgba(128, 128, 128, 0.1);
                }}
                .metrics-table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid {theme.get_color('grid')};
                    padding: 12px;
                    text-align: left;
                }}
                .metrics-table th {{
                    background-color: {theme.get_color('primary')};
                    color: white;
                }}
                .summary-box {{
                    background-color: rgba(31, 119, 180, 0.1);
                    border: 1px solid {theme.get_color('primary')};
                    border-radius: 5px;
                    padding: 20px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report.title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            {self._generate_executive_summary_html(results)}
            {self._generate_model_performance_html(results)}
            {self._generate_predictions_html(results)}
            
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_executive_summary_html(self, results: Dict[str, Any]) -> str:
        """Generate executive summary HTML."""
        if 'ensemble_metrics' not in results:
            return ""
        
        metrics = results['ensemble_metrics']
        symbol = results.get('symbol', 'Unknown')
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-box">
                <h3>Key Results for {symbol}</h3>
                <ul>
                    <li><strong>Ensemble RMSE:</strong> {metrics.get('rmse', 0):.4f}</li>
                    <li><strong>Directional Accuracy:</strong> {metrics.get('directional_accuracy', 0):.2f}%</li>
                    <li><strong>Models Trained:</strong> {len(results.get('model_metrics', {}))}</li>
                    <li><strong>Test Samples:</strong> {metrics.get('n_samples', 0)}</li>
                </ul>
            </div>
        </div>
        """
    
    def _generate_model_performance_html(self, results: Dict[str, Any]) -> str:
        """Generate model performance HTML."""
        if 'model_metrics' not in results:
            return ""
        
        model_metrics = results['model_metrics']
        
        table_rows = ""
        for model_name, metrics in model_metrics.items():
            table_rows += f"""
            <tr>
                <td>{model_name}</td>
                <td>{metrics.get('rmse', 0):.4f}</td>
                <td>{metrics.get('mae', 0):.4f}</td>
                <td>{metrics.get('directional_accuracy', 0):.2f}%</td>
                <td>{metrics.get('r2_score', 0):.4f}</td>
            </tr>
            """
        
        return f"""
        <div class="section">
            <h2>Model Performance Comparison</h2>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>Directional Accuracy</th>
                        <th>R² Score</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """
    
    def _generate_predictions_html(self, results: Dict[str, Any]) -> str:
        """Generate predictions HTML."""
        if 'future_predictions' not in results:
            return ""
        
        predictions = results['future_predictions']
        
        return f"""
        <div class="section">
            <h2>Future Predictions</h2>
            <p>Generated {len(predictions)} future predictions with confidence intervals.</p>
            <p><em>Note: Charts would be embedded here in a full implementation.</em></p>
        </div>
        """
    
    def _get_comprehensive_template(self) -> Dict[str, Any]:
        """Get comprehensive report template configuration."""
        return {
            'sections': ['executive_summary', 'model_performance', 'predictions', 'analysis'],
            'include_charts': True,
            'include_raw_data': False
        }
    
    def _get_summary_template(self) -> Dict[str, Any]:
        """Get summary report template configuration."""
        return {
            'sections': ['executive_summary', 'model_performance'],
            'include_charts': False,
            'include_raw_data': False
        }
    
    def _get_technical_template(self) -> Dict[str, Any]:
        """Get technical report template configuration."""
        return {
            'sections': ['executive_summary', 'model_performance', 'predictions', 'analysis', 'appendix'],
            'include_charts': True,
            'include_raw_data': True
        }
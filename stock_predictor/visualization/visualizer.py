"""Visualization utilities for stock price prediction results."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime

from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import StockPredictorError


class Visualizer:
    """Creates comprehensive charts and reports for stock prediction results."""
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        self.logger = get_logger('visualization.visualizer')
        self.style = style
        self.figsize = figsize
        
        # Try to import plotting libraries
        self.matplotlib_available = self._check_matplotlib()
        self.plotly_available = self._check_plotly()
        
        if not self.matplotlib_available and not self.plotly_available:
            self.logger.warning("No plotting libraries available. Install matplotlib or plotly for visualizations.")
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            return True
        except ImportError:
            return False
    
    def _check_plotly(self) -> bool:
        """Check if plotly is available."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            return True
        except ImportError:
            return False
    
    def plot_historical_vs_predicted(self, historical: pd.DataFrame, 
                                   predictions: Dict[str, np.ndarray],
                                   actual_column: str = 'close',
                                   date_column: str = 'date',
                                   backend: str = 'matplotlib') -> Any:
        """
        Plot historical data vs predictions.
        
        Args:
            historical: DataFrame with historical data
            predictions: Dict of {model_name: predictions_array}
            actual_column: Column name for actual values
            date_column: Column name for dates
            backend: Plotting backend ('matplotlib' or 'plotly')
            
        Returns:
            Figure object
        """
        if backend == 'matplotlib' and self.matplotlib_available:
            return self._plot_historical_vs_predicted_mpl(
                historical, predictions, actual_column, date_column
            )
        elif backend == 'plotly' and self.plotly_available:
            return self._plot_historical_vs_predicted_plotly(
                historical, predictions, actual_column, date_column
            )
        else:
            self.logger.error(f"Backend {backend} not available")
            return None
    
    def _plot_historical_vs_predicted_mpl(self, historical: pd.DataFrame,
                                        predictions: Dict[str, np.ndarray],
                                        actual_column: str, date_column: str):
        """Create matplotlib plot of historical vs predicted."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        if date_column in historical.columns:
            dates = historical[date_column]
        else:
            dates = range(len(historical))
        
        ax.plot(dates, historical[actual_column], 
               label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions)))
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            # Align predictions with dates (assuming they correspond to the last part of historical data)
            pred_dates = dates[-len(pred):] if len(pred) <= len(dates) else dates
            ax.plot(pred_dates, pred, 
                   label=f'{model_name} Prediction', 
                   color=colors[i], linewidth=1.5, alpha=0.7)
        
        ax.set_title('Stock Price: Actual vs Predicted', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date' if date_column in historical.columns else 'Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if date_column in historical.columns:
            fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def _plot_historical_vs_predicted_plotly(self, historical: pd.DataFrame,
                                           predictions: Dict[str, np.ndarray],
                                           actual_column: str, date_column: str):
        """Create plotly plot of historical vs predicted."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = go.Figure()
        
        # Plot historical data
        if date_column in historical.columns:
            x_data = historical[date_column]
        else:
            x_data = list(range(len(historical)))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=historical[actual_column],
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Plot predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            pred_x = x_data[-len(pred):] if len(pred) <= len(x_data) else x_data
            
            fig.add_trace(go.Scatter(
                x=pred_x,
                y=pred,
                mode='lines',
                name=f'{model_name} Prediction',
                line=dict(color=colors[i % len(colors)], width=1.5)
            ))
        
        fig.update_layout(
            title='Stock Price: Actual vs Predicted',
            xaxis_title='Date' if date_column in historical.columns else 'Time',
            yaxis_title='Price',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        return fig
    
    def plot_model_comparison(self, model_predictions: Dict[str, np.ndarray],
                            actual_values: np.ndarray,
                            model_metrics: Dict[str, Dict[str, float]] = None,
                            backend: str = 'matplotlib') -> Any:
        """
        Plot comparison of different model predictions.
        
        Args:
            model_predictions: Dict of {model_name: predictions}
            actual_values: Array of actual values
            model_metrics: Dict of {model_name: {metric: value}}
            backend: Plotting backend
            
        Returns:
            Figure object
        """
        if backend == 'matplotlib' and self.matplotlib_available:
            return self._plot_model_comparison_mpl(model_predictions, actual_values, model_metrics)
        elif backend == 'plotly' and self.plotly_available:
            return self._plot_model_comparison_plotly(model_predictions, actual_values, model_metrics)
        else:
            self.logger.error(f"Backend {backend} not available")
            return None
    
    def _plot_model_comparison_mpl(self, model_predictions: Dict[str, np.ndarray],
                                 actual_values: np.ndarray,
                                 model_metrics: Dict[str, Dict[str, float]] = None):
        """Create matplotlib model comparison plot."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use(self.style)
        
        n_models = len(model_predictions)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: All predictions vs actual
        ax = axes[0]
        ax.plot(actual_values, label='Actual', color='black', linewidth=2, alpha=0.8)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_models))
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            ax.plot(pred, label=model_name, color=colors[i], alpha=0.7)
        
        ax.set_title('Model Predictions Comparison')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot - Predicted vs Actual
        ax = axes[1]
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            ax.scatter(actual_values, pred, alpha=0.6, label=model_name, color=colors[i])
        
        # Add perfect prediction line
        min_val, max_val = min(actual_values.min(), min(pred.min() for pred in model_predictions.values())), \
                          max(actual_values.max(), max(pred.max() for pred in model_predictions.values()))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_title('Predicted vs Actual')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        ax = axes[2]
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            residuals = actual_values - pred
            ax.plot(residuals, alpha=0.7, label=f'{model_name} Residuals', color=colors[i])
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Model Residuals')
        ax.set_xlabel('Time')
        ax.set_ylabel('Residual (Actual - Predicted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics (if provided)
        ax = axes[3]
        if model_metrics:
            metrics = ['rmse', 'mae', 'directional_accuracy']
            x_pos = np.arange(len(model_predictions))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [model_metrics.get(model, {}).get(metric, 0) for model in model_predictions.keys()]
                ax.bar(x_pos + i * width, values, width, label=metric.upper(), alpha=0.7)
            
            ax.set_title('Model Performance Metrics')
            ax.set_xlabel('Models')
            ax.set_ylabel('Metric Value')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(list(model_predictions.keys()), rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metrics provided', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance Metrics')
        
        plt.tight_layout()
        return fig
    
    def _plot_model_comparison_plotly(self, model_predictions: Dict[str, np.ndarray],
                                    actual_values: np.ndarray,
                                    model_metrics: Dict[str, Dict[str, float]] = None):
        """Create plotly model comparison plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Predictions Comparison', 'Predicted vs Actual', 
                          'Residuals', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # Plot 1: All predictions vs actual
        fig.add_trace(go.Scatter(
            y=actual_values, mode='lines', name='Actual',
            line=dict(color='black', width=2)
        ), row=1, col=1)
        
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            fig.add_trace(go.Scatter(
                y=pred, mode='lines', name=model_name,
                line=dict(color=colors[i % len(colors)])
            ), row=1, col=1)
        
        # Plot 2: Scatter plot
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            fig.add_trace(go.Scatter(
                x=actual_values, y=pred, mode='markers', name=f'{model_name} Scatter',
                marker=dict(color=colors[i % len(colors)], opacity=0.6),
                showlegend=False
            ), row=1, col=2)
        
        # Plot 3: Residuals
        for i, (model_name, pred) in enumerate(model_predictions.items()):
            residuals = actual_values - pred
            fig.add_trace(go.Scatter(
                y=residuals, mode='lines', name=f'{model_name} Residuals',
                line=dict(color=colors[i % len(colors)]),
                showlegend=False
            ), row=2, col=1)
        
        # Plot 4: Performance metrics
        if model_metrics:
            metrics = ['rmse', 'mae', 'directional_accuracy']
            for metric in metrics:
                values = [model_metrics.get(model, {}).get(metric, 0) for model in model_predictions.keys()]
                fig.add_trace(go.Bar(
                    x=list(model_predictions.keys()), y=values, name=metric.upper(),
                    showlegend=False
                ), row=2, col=2)
        
        fig.update_layout(height=800, width=1200, title_text="Model Comparison Dashboard")
        return fig
    
    def plot_ensemble_predictions(self, predictions: Dict[str, Any],
                                confidence_intervals: Tuple[np.ndarray, np.ndarray] = None,
                                actual_values: np.ndarray = None,
                                backend: str = 'matplotlib') -> Any:
        """
        Plot ensemble predictions with confidence intervals.
        
        Args:
            predictions: Dict with ensemble results
            confidence_intervals: Tuple of (lower_bounds, upper_bounds)
            actual_values: Array of actual values for comparison
            backend: Plotting backend
            
        Returns:
            Figure object
        """
        if backend == 'matplotlib' and self.matplotlib_available:
            return self._plot_ensemble_predictions_mpl(predictions, confidence_intervals, actual_values)
        elif backend == 'plotly' and self.plotly_available:
            return self._plot_ensemble_predictions_plotly(predictions, confidence_intervals, actual_values)
        else:
            self.logger.error(f"Backend {backend} not available")
            return None
    
    def _plot_ensemble_predictions_mpl(self, predictions: Dict[str, Any],
                                     confidence_intervals: Tuple[np.ndarray, np.ndarray] = None,
                                     actual_values: np.ndarray = None):
        """Create matplotlib ensemble predictions plot."""
        import matplotlib.pyplot as plt
        
        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ensemble_pred = predictions['ensemble_prediction']
        x_data = range(len(ensemble_pred))
        
        # Plot actual values if provided
        if actual_values is not None:
            ax.plot(x_data, actual_values, label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot ensemble prediction
        ax.plot(x_data, ensemble_pred, label='Ensemble Prediction', 
               color='red', linewidth=2, alpha=0.8)
        
        # Plot confidence intervals
        if confidence_intervals is not None:
            lower_bounds, upper_bounds = confidence_intervals
            ax.fill_between(x_data, lower_bounds, upper_bounds, 
                          alpha=0.3, color='red', label='Confidence Interval')
        
        # Plot individual model predictions if available
        if 'individual_predictions' in predictions:
            colors = plt.cm.Set3(np.linspace(0, 1, len(predictions['individual_predictions'])))
            for i, (model_name, pred) in enumerate(predictions['individual_predictions'].items()):
                ax.plot(x_data, pred, label=f'{model_name}', 
                       color=colors[i], alpha=0.5, linestyle='--')
        
        ax.set_title('Ensemble Predictions with Confidence Intervals', fontsize=16, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_ensemble_predictions_plotly(self, predictions: Dict[str, Any],
                                        confidence_intervals: Tuple[np.ndarray, np.ndarray] = None,
                                        actual_values: np.ndarray = None):
        """Create plotly ensemble predictions plot."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        ensemble_pred = predictions['ensemble_prediction']
        x_data = list(range(len(ensemble_pred)))
        
        # Plot actual values if provided
        if actual_values is not None:
            fig.add_trace(go.Scatter(
                x=x_data, y=actual_values, mode='lines', name='Actual',
                line=dict(color='black', width=2)
            ))
        
        # Plot confidence intervals first (so they appear behind the prediction line)
        if confidence_intervals is not None:
            lower_bounds, upper_bounds = confidence_intervals
            
            # Create filled area for confidence interval
            fig.add_trace(go.Scatter(
                x=x_data + x_data[::-1],
                y=list(upper_bounds) + list(lower_bounds[::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        # Plot ensemble prediction
        fig.add_trace(go.Scatter(
            x=x_data, y=ensemble_pred, mode='lines', name='Ensemble Prediction',
            line=dict(color='red', width=2)
        ))
        
        # Plot individual model predictions if available
        if 'individual_predictions' in predictions:
            colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
            for i, (model_name, pred) in enumerate(predictions['individual_predictions'].items()):
                fig.add_trace(go.Scatter(
                    x=x_data, y=pred, mode='lines', name=model_name,
                    line=dict(color=colors[i % len(colors)], dash='dash', width=1),
                    opacity=0.7
                ))
        
        fig.update_layout(
            title='Ensemble Predictions with Confidence Intervals',
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        return fig
    
    def generate_performance_dashboard(self, metrics: Dict[str, Any],
                                     backtest_results: Dict[str, Any] = None,
                                     backend: str = 'matplotlib') -> Any:
        """
        Generate comprehensive performance dashboard.
        
        Args:
            metrics: Dictionary with performance metrics
            backtest_results: Dictionary with backtest results
            backend: Plotting backend
            
        Returns:
            Figure object
        """
        if backend == 'matplotlib' and self.matplotlib_available:
            return self._generate_performance_dashboard_mpl(metrics, backtest_results)
        elif backend == 'plotly' and self.plotly_available:
            return self._generate_performance_dashboard_plotly(metrics, backtest_results)
        else:
            self.logger.error(f"Backend {backend} not available")
            return None
    
    def _generate_performance_dashboard_mpl(self, metrics: Dict[str, Any],
                                          backtest_results: Dict[str, Any] = None):
        """Create matplotlib performance dashboard."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use(self.style)
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Model Performance Comparison (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if isinstance(metrics, dict) and len(metrics) > 1:
            models = list(metrics.keys())
            rmse_values = [metrics[model].get('rmse', 0) for model in models]
            mae_values = [metrics[model].get('mae', 0) for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax1.bar(x - width/2, rmse_values, width, label='RMSE', alpha=0.8)
            ax1.bar(x + width/2, mae_values, width, label='MAE', alpha=0.8)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Error')
            ax1.set_title('Model Performance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Directional Accuracy (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        if isinstance(metrics, dict):
            models = list(metrics.keys())
            dir_acc = [metrics[model].get('directional_accuracy', 0) for model in models]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
            ax2.pie(dir_acc, labels=models, autopct='%1.1f%%', colors=colors)
            ax2.set_title('Directional Accuracy')
        
        # Plot 3: R² Score Comparison (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if isinstance(metrics, dict):
            models = list(metrics.keys())
            r2_values = [metrics[model].get('r2_score', 0) for model in models]
            
            ax3.barh(models, r2_values, alpha=0.8)
            ax3.set_xlabel('R² Score')
            ax3.set_title('Model R² Scores')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correlation Analysis (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if isinstance(metrics, dict):
            models = list(metrics.keys())
            correlations = [metrics[model].get('correlation', 0) for model in models]
            
            ax4.bar(models, correlations, alpha=0.8, color='green')
            ax4.set_ylabel('Correlation')
            ax4.set_title('Prediction Correlation')
            ax4.set_xticklabels(models, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Backtest Results (if available)
        ax5 = fig.add_subplot(gs[1, 2])
        if backtest_results:
            # Plot fold performance variability
            fold_rmse = []
            fold_labels = []
            for model_name, results in backtest_results.items():
                if 'fold_results' in results:
                    rmse_values = [fold['rmse'] for fold in results['fold_results']]
                    fold_rmse.extend(rmse_values)
                    fold_labels.extend([model_name] * len(rmse_values))
            
            if fold_rmse:
                ax5.boxplot([fold_rmse], labels=['All Models'])
                ax5.set_ylabel('RMSE')
                ax5.set_title('Backtest RMSE Distribution')
                ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary Statistics (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create summary text
        summary_text = "Performance Summary\n" + "="*50 + "\n"
        
        if isinstance(metrics, dict):
            best_rmse_model = min(metrics.items(), key=lambda x: x[1].get('rmse', float('inf')))
            best_dir_acc_model = max(metrics.items(), key=lambda x: x[1].get('directional_accuracy', 0))
            
            summary_text += f"Best RMSE: {best_rmse_model[0]} ({best_rmse_model[1].get('rmse', 0):.4f})\n"
            summary_text += f"Best Directional Accuracy: {best_dir_acc_model[0]} ({best_dir_acc_model[1].get('directional_accuracy', 0):.2f}%)\n"
            
            if backtest_results:
                summary_text += f"\nBacktest Results:\n"
                for model_name, results in backtest_results.items():
                    summary_text += f"  {model_name}: {results.get('n_successful_folds', 0)}/{results.get('n_splits', 0)} folds successful\n"
        
        ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', fontfamily='monospace')
        
        plt.suptitle('Stock Prediction Performance Dashboard', fontsize=16, fontweight='bold')
        return fig
    
    def _generate_performance_dashboard_plotly(self, metrics: Dict[str, Any],
                                             backtest_results: Dict[str, Any] = None):
        """Create plotly performance dashboard."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=('Performance Comparison', 'Directional Accuracy', 'R² Scores',
                          'Correlation', 'Backtest Results', 'Model Weights',
                          'Summary Statistics', '', ''),
            specs=[[{"colspan": 2}, None, {}],
                   [{}, {}, {}],
                   [{"colspan": 3}, None, None]]
        )
        
        if isinstance(metrics, dict) and len(metrics) > 1:
            models = list(metrics.keys())
            
            # Performance comparison
            rmse_values = [metrics[model].get('rmse', 0) for model in models]
            mae_values = [metrics[model].get('mae', 0) for model in models]
            
            fig.add_trace(go.Bar(x=models, y=rmse_values, name='RMSE'), row=1, col=1)
            fig.add_trace(go.Bar(x=models, y=mae_values, name='MAE'), row=1, col=1)
            
            # Directional accuracy pie chart
            dir_acc = [metrics[model].get('directional_accuracy', 0) for model in models]
            fig.add_trace(go.Pie(labels=models, values=dir_acc, name="Directional Accuracy"), row=1, col=3)
            
            # R² scores
            r2_values = [metrics[model].get('r2_score', 0) for model in models]
            fig.add_trace(go.Bar(x=models, y=r2_values, name='R² Score', showlegend=False), row=2, col=1)
            
            # Correlation
            correlations = [metrics[model].get('correlation', 0) for model in models]
            fig.add_trace(go.Bar(x=models, y=correlations, name='Correlation', showlegend=False), row=2, col=2)
        
        fig.update_layout(height=1000, width=1400, title_text="Performance Dashboard")
        return fig
    
    def save_figure(self, fig, filepath: str, format: str = 'png', **kwargs) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Figure object
            filepath: Path to save file
            format: File format ('png', 'pdf', 'html', 'svg')
            **kwargs: Additional arguments for saving
        """
        try:
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                fig.savefig(filepath, format=format, dpi=300, bbox_inches='tight', **kwargs)
            elif hasattr(fig, 'write_html'):  # Plotly figure
                if format.lower() == 'html':
                    fig.write_html(filepath, **kwargs)
                elif format.lower() in ['png', 'pdf', 'svg']:
                    # Requires kaleido package
                    if format.lower() == 'png':
                        fig.write_image(filepath, **kwargs)
                    elif format.lower() == 'pdf':
                        fig.write_image(filepath, **kwargs)
                    elif format.lower() == 'svg':
                        fig.write_image(filepath, **kwargs)
            
            self.logger.info(f"Figure saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save figure: {str(e)}")
    
    def create_report_html(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Create comprehensive HTML report.
        
        Args:
            results: Dictionary with all results
            output_path: Path to save HTML report
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Price Prediction Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e7f3ff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Stock Price Prediction Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents the results of stock price prediction using ensemble machine learning methods.</p>
            </div>
            
            <div class="section">
                <h2>Model Performance</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>Directional Accuracy (%)</th>
                        <th>R² Score</th>
                    </tr>
        """
        
        # Add model performance rows
        if 'model_comparison' in results:
            for _, row in results['model_comparison'].iterrows():
                html_content += f"""
                    <tr>
                        <td>{row['model_name']}</td>
                        <td>{row['rmse']:.4f}</td>
                        <td>{row['mae']:.4f}</td>
                        <td>{row['directional_accuracy']:.2f}</td>
                        <td>{row['r2_score']:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Ensemble Configuration</h2>
        """
        
        if 'ensemble_weights' in results:
            html_content += "<ul>"
            for model, weight in results['ensemble_weights'].items():
                html_content += f"<li>{model}: {weight:.4f} ({weight*100:.1f}%)</li>"
            html_content += "</ul>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Key Findings</h2>
                <ul>
                    <li>Ensemble approach combines multiple model strengths</li>
                    <li>Dynamic weighting adjusts based on model performance</li>
                    <li>Confidence intervals provide uncertainty estimates</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create HTML report: {str(e)}")
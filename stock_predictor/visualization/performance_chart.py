"""Performance chart implementation for model comparison and analysis."""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from stock_predictor.utils.logging import get_logger
from .base_chart import InteractiveChart
from .exceptions import ChartCreationError


class PerformanceChart(InteractiveChart):
    """Interactive performance chart for model comparison and analysis."""
    
    def __init__(self, theme: Any, config: Optional[Dict[str, Any]] = None, subtype: str = 'comparison'):
        """
        Initialize performance chart.
        
        Args:
            theme: Theme object for styling
            config: Optional configuration dictionary
            subtype: Chart subtype ('comparison', 'time_series', 'residuals')
        """
        super().__init__('performance', theme, config)
        
        self.subtype = subtype
        self.model_metrics = {}
        self.time_series_data = {}
        self.residuals_data = {}
        
        # Create the figure
        self.figure = self._create_figure()
        
        self.logger.debug(f"Performance chart initialized with subtype: {subtype}")
    
    def _create_figure(self) -> Any:
        """Create the base figure object."""
        if self.backend == 'plotly':
            return self._create_plotly_figure()
        else:
            return self._create_matplotlib_figure()
    
    def _create_plotly_figure(self) -> Any:
        """Create Plotly figure for performance chart."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if self.subtype == 'comparison':
            fig = go.Figure()
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Metric Value',
                showlegend=True
            )
        else:
            # For more complex subtypes, create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('RMSE Comparison', 'MAE Comparison', 
                              'Directional Accuracy', 'R² Score'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
        
        return fig
    
    def _create_matplotlib_figure(self) -> Any:
        """Create Matplotlib figure for performance chart."""
        import matplotlib.pyplot as plt
        
        if self.subtype == 'comparison':
            fig, ax = plt.subplots(figsize=self.config.get('figsize', (10, 6)))
            ax.set_title('Model Performance Comparison')
            ax.set_xlabel('Models')
            ax.set_ylabel('Metric Value')
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.config.get('figsize', (12, 10)))
            fig.suptitle('Model Performance Analysis')
        
        return fig
    
    def add_model_metrics(self, model_metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Add model performance metrics to the chart.
        
        Args:
            model_metrics: Dict of {model_name: {metric: value}}
        """
        try:
            self.model_metrics.update(model_metrics)
            
            if self.backend == 'plotly':
                self._add_model_metrics_plotly(model_metrics)
            else:
                self._add_model_metrics_matplotlib(model_metrics)
            
            self.logger.debug(f"Added metrics for {len(model_metrics)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add model metrics: {str(e)}")
    
    def add_time_series_performance(self, time_series_data: Dict[str, pd.Series]) -> None:
        """
        Add time series performance data.
        
        Args:
            time_series_data: Dict of {model_name: performance_series}
        """
        try:
            self.time_series_data.update(time_series_data)
            
            if self.backend == 'plotly':
                self._add_time_series_plotly(time_series_data)
            else:
                self._add_time_series_matplotlib(time_series_data)
            
            self.logger.debug(f"Added time series data for {len(time_series_data)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add time series performance: {str(e)}")
    
    def add_rolling_metrics(self, rolling_data: Dict[str, Dict[str, pd.Series]], window: int = 30) -> None:
        """
        Add rolling performance metrics over time.
        
        Args:
            rolling_data: Dict of {model_name: {metric: rolling_series}}
            window: Rolling window size
        """
        try:
            colors = self.theme.get_color_palette(len(rolling_data))
            
            if self.backend == 'plotly':
                import plotly.graph_objects as go
                
                for i, (model_name, metrics) in enumerate(rolling_data.items()):
                    for metric_name, series in metrics.items():
                        self.figure.add_trace(go.Scatter(
                            x=series.index,
                            y=series.values,
                            mode='lines',
                            name=f'{model_name} {metric_name}',
                            line=dict(color=colors[i % len(colors)], width=2),
                            hovertemplate=f'{model_name} {metric_name}<br>Date: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
                        ))
                
                self.figure.update_layout(
                    title=f'Rolling Performance Metrics (Window: {window})',
                    xaxis_title='Date',
                    yaxis_title='Metric Value'
                )
            else:
                ax = self.figure.gca()
                
                for i, (model_name, metrics) in enumerate(rolling_data.items()):
                    for metric_name, series in metrics.items():
                        ax.plot(series.index, series.values, 
                               label=f'{model_name} {metric_name}',
                               color=colors[i % len(colors)], linewidth=2)
                
                ax.set_title(f'Rolling Performance Metrics (Window: {window})')
                ax.set_xlabel('Date')
                ax.set_ylabel('Metric Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            self.logger.debug(f"Added rolling metrics for {len(rolling_data)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add rolling metrics: {str(e)}")
    
    def add_residual_analysis(self, residuals: Dict[str, np.ndarray]) -> None:
        """
        Add residual analysis data.
        
        Args:
            residuals: Dict of {model_name: residuals_array}
        """
        try:
            self.residuals_data.update(residuals)
            
            if self.backend == 'plotly':
                self._add_residuals_plotly(residuals)
            else:
                self._add_residuals_matplotlib(residuals)
            
            self.logger.debug(f"Added residual data for {len(residuals)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add residual analysis: {str(e)}")
    
    def create_prediction_error_distribution(self, errors: Dict[str, np.ndarray]) -> None:
        """
        Create prediction error distribution charts.
        
        Args:
            errors: Dict of {model_name: error_array}
        """
        try:
            colors = self.theme.get_color_palette(len(errors))
            
            if self.backend == 'plotly':
                import plotly.graph_objects as go
                import plotly.figure_factory as ff
                
                # Create distribution plots
                hist_data = list(errors.values())
                group_labels = list(errors.keys())
                
                fig = ff.create_distplot(hist_data, group_labels, 
                                       colors=colors[:len(errors)],
                                       show_hist=True, show_rug=False)
                
                fig.update_layout(
                    title='Prediction Error Distribution',
                    xaxis_title='Prediction Error',
                    yaxis_title='Density'
                )
                
                # Replace current figure
                self.figure = fig
                
            else:
                ax = self.figure.gca()
                
                for i, (model_name, error_data) in enumerate(errors.items()):
                    ax.hist(error_data, bins=30, alpha=0.7, 
                           label=model_name, color=colors[i % len(colors)])
                
                ax.set_title('Prediction Error Distribution')
                ax.set_xlabel('Prediction Error')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            self.logger.debug(f"Created error distribution for {len(errors)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create error distribution: {str(e)}")
    
    def create_qq_plots(self, residuals: Dict[str, np.ndarray]) -> None:
        """
        Create Q-Q plots for residual analysis.
        
        Args:
            residuals: Dict of {model_name: residuals_array}
        """
        try:
            import scipy.stats as stats
            
            if self.backend == 'plotly':
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                n_models = len(residuals)
                cols = min(3, n_models)
                rows = (n_models + cols - 1) // cols
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=list(residuals.keys())
                )
                
                colors = self.theme.get_color_palette(n_models)
                
                for i, (model_name, resid_data) in enumerate(residuals.items()):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    # Calculate theoretical quantiles
                    sorted_residuals = np.sort(resid_data)
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
                    
                    # Add Q-Q plot
                    fig.add_trace(go.Scatter(
                        x=theoretical_quantiles,
                        y=sorted_residuals,
                        mode='markers',
                        name=model_name,
                        marker=dict(color=colors[i % len(colors)]),
                        showlegend=False
                    ), row=row, col=col)
                    
                    # Add reference line
                    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
                    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
                    fig.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=False
                    ), row=row, col=col)
                
                fig.update_layout(title='Q-Q Plots for Residual Analysis')
                self.figure = fig
                
            else:
                import matplotlib.pyplot as plt
                
                n_models = len(residuals)
                cols = min(3, n_models)
                rows = (n_models + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
                if n_models == 1:
                    axes = [axes]
                elif rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                colors = self.theme.get_color_palette(n_models)
                
                for i, (model_name, resid_data) in enumerate(residuals.items()):
                    ax = axes[i]
                    stats.probplot(resid_data, dist="norm", plot=ax)
                    ax.set_title(f'{model_name} Q-Q Plot')
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(n_models, len(axes)):
                    axes[i].set_visible(False)
                
                fig.suptitle('Q-Q Plots for Residual Analysis')
                fig.tight_layout()
                self.figure = fig
            
            self.logger.debug(f"Created Q-Q plots for {len(residuals)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create Q-Q plots: {str(e)}")
    
    def create_correlation_matrix(self, predictions: Dict[str, np.ndarray]) -> None:
        """
        Create correlation matrix between model predictions.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
        """
        try:
            import pandas as pd
            
            # Create DataFrame from predictions
            pred_df = pd.DataFrame(predictions)
            correlation_matrix = pred_df.corr()
            
            if self.backend == 'plotly':
                import plotly.graph_objects as go
                
                fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix.round(3).values,
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title='Model Prediction Correlation Matrix',
                    xaxis_title='Models',
                    yaxis_title='Models'
                )
                
                self.figure = fig
                
            else:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create heatmap
                sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                           square=True, ax=ax, cbar_kws={'label': 'Correlation'})
                
                ax.set_title('Model Prediction Correlation Matrix')
                self.figure = fig
            
            self.logger.debug(f"Created correlation matrix for {len(predictions)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create correlation matrix: {str(e)}")
    
    def create_performance_radar_chart(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Create radar chart for multi-dimensional performance comparison.
        
        Args:
            metrics: Dict of {model_name: {metric: value}}
        """
        try:
            if self.backend == 'plotly':
                import plotly.graph_objects as go
                
                # Normalize metrics to 0-1 scale for radar chart
                all_metrics = set()
                for model_metrics in metrics.values():
                    all_metrics.update(model_metrics.keys())
                
                all_metrics = list(all_metrics)
                colors = self.theme.get_color_palette(len(metrics))
                
                fig = go.Figure()
                
                for i, (model_name, model_metrics) in enumerate(metrics.items()):
                    # Normalize values (assuming higher is better for most metrics)
                    values = []
                    for metric in all_metrics:
                        value = model_metrics.get(metric, 0)
                        # Special handling for error metrics (lower is better)
                        if metric.lower() in ['rmse', 'mae', 'mse']:
                            # Invert error metrics (1 - normalized_error)
                            max_val = max([m.get(metric, 0) for m in metrics.values()])
                            normalized = 1 - (value / max_val) if max_val > 0 else 0
                        else:
                            # For accuracy metrics (higher is better)
                            max_val = max([m.get(metric, 0) for m in metrics.values()])
                            normalized = value / max_val if max_val > 0 else 0
                        values.append(normalized)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=all_metrics,
                        fill='toself',
                        name=model_name,
                        line_color=colors[i % len(colors)]
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title='Model Performance Radar Chart',
                    showlegend=True
                )
                
                self.figure = fig
                
            else:
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Matplotlib radar chart implementation
                all_metrics = set()
                for model_metrics in metrics.values():
                    all_metrics.update(model_metrics.keys())
                
                all_metrics = list(all_metrics)
                N = len(all_metrics)
                
                # Compute angle for each axis
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Complete the circle
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                colors = self.theme.get_color_palette(len(metrics))
                
                for i, (model_name, model_metrics) in enumerate(metrics.items()):
                    # Normalize values
                    values = []
                    for metric in all_metrics:
                        value = model_metrics.get(metric, 0)
                        if metric.lower() in ['rmse', 'mae', 'mse']:
                            max_val = max([m.get(metric, 0) for m in metrics.values()])
                            normalized = 1 - (value / max_val) if max_val > 0 else 0
                        else:
                            max_val = max([m.get(metric, 0) for m in metrics.values()])
                            normalized = value / max_val if max_val > 0 else 0
                        values.append(normalized)
                    
                    values += values[:1]  # Complete the circle
                    
                    ax.plot(angles, values, 'o-', linewidth=2, 
                           label=model_name, color=colors[i % len(colors)])
                    ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(all_metrics)
                ax.set_ylim(0, 1)
                ax.set_title('Model Performance Radar Chart', pad=20)
                ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
                ax.grid(True)
                
                self.figure = fig
            
            self.logger.debug(f"Created radar chart for {len(metrics)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create radar chart: {str(e)}")
    
    def create_performance_summary_table(self, metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create a summary table of all performance metrics.
        
        Args:
            metrics: Dict of {model_name: {metric: value}}
            
        Returns:
            DataFrame with performance summary
        """
        try:
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(metrics).T
            
            # Round values for better display
            df = df.round(4)
            
            # Add ranking columns
            for col in df.columns:
                if col.lower() in ['rmse', 'mae', 'mse']:
                    # Lower is better for error metrics
                    df[f'{col}_rank'] = df[col].rank(ascending=True)
                else:
                    # Higher is better for accuracy metrics
                    df[f'{col}_rank'] = df[col].rank(ascending=False)
            
            # Add overall score (simple average of normalized ranks)
            rank_cols = [col for col in df.columns if col.endswith('_rank')]
            df['overall_score'] = df[rank_cols].mean(axis=1)
            df['overall_rank'] = df['overall_score'].rank(ascending=True)
            
            self.logger.debug(f"Created performance summary table for {len(metrics)} models")
            
            return df
            
        except Exception as e:
            raise ChartCreationError(f"Failed to create summary table: {str(e)}")
    
    def _add_model_metrics_plotly(self, model_metrics: Dict[str, Dict[str, float]]) -> None:
        """Add model metrics to Plotly figure."""
        import plotly.graph_objects as go
        
        if self.subtype == 'comparison':
            # Simple bar chart comparison
            models = list(model_metrics.keys())
            metrics = ['rmse', 'mae', 'directional_accuracy', 'r2_score']
            colors = self.theme.get_color_palette(len(metrics))
            
            for i, metric in enumerate(metrics):
                values = [model_metrics.get(model, {}).get(metric, 0) for model in models]
                
                self.figure.add_trace(go.Bar(
                    x=models,
                    y=values,
                    name=metric.upper(),
                    marker_color=colors[i % len(colors)]
                ))
        else:
            # Multi-subplot layout
            models = list(model_metrics.keys())
            
            # RMSE comparison
            rmse_values = [model_metrics.get(model, {}).get('rmse', 0) for model in models]
            self.figure.add_trace(go.Bar(x=models, y=rmse_values, name='RMSE'), row=1, col=1)
            
            # MAE comparison
            mae_values = [model_metrics.get(model, {}).get('mae', 0) for model in models]
            self.figure.add_trace(go.Bar(x=models, y=mae_values, name='MAE'), row=1, col=2)
            
            # Directional accuracy
            dir_acc = [model_metrics.get(model, {}).get('directional_accuracy', 0) for model in models]
            self.figure.add_trace(go.Bar(x=models, y=dir_acc, name='Dir. Acc.'), row=2, col=1)
            
            # R² score
            r2_values = [model_metrics.get(model, {}).get('r2_score', 0) for model in models]
            self.figure.add_trace(go.Bar(x=models, y=r2_values, name='R²'), row=2, col=2)
    
    def _add_model_metrics_matplotlib(self, model_metrics: Dict[str, Dict[str, float]]) -> None:
        """Add model metrics to Matplotlib figure."""
        import numpy as np
        
        models = list(model_metrics.keys())
        
        if self.subtype == 'comparison':
            ax = self.figure.gca()
            
            # Create grouped bar chart
            metrics = ['rmse', 'mae', 'directional_accuracy']
            x = np.arange(len(models))
            width = 0.25
            colors = self.theme.get_color_palette(len(metrics))
            
            for i, metric in enumerate(metrics):
                values = [model_metrics.get(model, {}).get(metric, 0) for model in models]
                ax.bar(x + i * width, values, width, label=metric.upper(), 
                      color=colors[i % len(colors)], alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Metric Value')
            ax.set_xticks(x + width)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            # Multi-subplot layout
            axes = self.figure.get_axes()
            
            if len(axes) >= 4:
                # RMSE
                rmse_values = [model_metrics.get(model, {}).get('rmse', 0) for model in models]
                axes[0].bar(models, rmse_values, alpha=0.8)
                axes[0].set_title('RMSE Comparison')
                axes[0].set_ylabel('RMSE')
                
                # MAE
                mae_values = [model_metrics.get(model, {}).get('mae', 0) for model in models]
                axes[1].bar(models, mae_values, alpha=0.8)
                axes[1].set_title('MAE Comparison')
                axes[1].set_ylabel('MAE')
                
                # Directional Accuracy
                dir_acc = [model_metrics.get(model, {}).get('directional_accuracy', 0) for model in models]
                axes[2].bar(models, dir_acc, alpha=0.8)
                axes[2].set_title('Directional Accuracy')
                axes[2].set_ylabel('Accuracy (%)')
                
                # R² Score
                r2_values = [model_metrics.get(model, {}).get('r2_score', 0) for model in models]
                axes[3].bar(models, r2_values, alpha=0.8)
                axes[3].set_title('R² Score')
                axes[3].set_ylabel('R²')
                
                # Adjust layout
                self.figure.tight_layout()
    
    def _add_time_series_plotly(self, time_series_data: Dict[str, pd.Series]) -> None:
        """Add time series performance data to Plotly figure."""
        import plotly.graph_objects as go
        
        colors = self.theme.get_color_palette(len(time_series_data))
        
        for i, (model_name, series) in enumerate(time_series_data.items()):
            self.figure.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'{model_name}<br>Date: %{{x}}<br>Performance: %{{y:.4f}}<extra></extra>'
            ))
        
        self.figure.update_layout(
            title='Model Performance Over Time',
            xaxis_title='Date',
            yaxis_title='Performance Metric'
        )
    
    def _add_time_series_matplotlib(self, time_series_data: Dict[str, pd.Series]) -> None:
        """Add time series performance data to Matplotlib figure."""
        ax = self.figure.gca()
        colors = self.theme.get_color_palette(len(time_series_data))
        
        for i, (model_name, series) in enumerate(time_series_data.items()):
            ax.plot(series.index, series.values, 
                   label=model_name, color=colors[i % len(colors)], 
                   linewidth=2, marker='o', markersize=3)
        
        ax.set_title('Model Performance Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Performance Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _add_residuals_plotly(self, residuals: Dict[str, np.ndarray]) -> None:
        """Add residual analysis to Plotly figure."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        n_models = len(residuals)
        fig = make_subplots(
            rows=2, cols=n_models,
            subplot_titles=[f'{name} Residuals' for name in residuals.keys()] + 
                          [f'{name} vs Fitted' for name in residuals.keys()],
            vertical_spacing=0.1
        )
        
        colors = self.theme.get_color_palette(n_models)
        
        for i, (model_name, resid_data) in enumerate(residuals.items()):
            col = i + 1
            
            # Residuals plot
            fig.add_trace(go.Scatter(
                y=resid_data,
                mode='markers',
                name=f'{model_name} Residuals',
                marker=dict(color=colors[i % len(colors)]),
                showlegend=False
            ), row=1, col=col)
            
            # Residuals vs fitted (assuming fitted values are indices)
            fitted_values = np.arange(len(resid_data))
            fig.add_trace(go.Scatter(
                x=fitted_values,
                y=resid_data,
                mode='markers',
                name=f'{model_name} vs Fitted',
                marker=dict(color=colors[i % len(colors)]),
                showlegend=False
            ), row=2, col=col)
        
        fig.update_layout(title='Residual Analysis')
        self.figure = fig
    
    def _add_residuals_matplotlib(self, residuals: Dict[str, np.ndarray]) -> None:
        """Add residual analysis to Matplotlib figure."""
        import matplotlib.pyplot as plt
        
        n_models = len(residuals)
        fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        colors = self.theme.get_color_palette(n_models)
        
        for i, (model_name, resid_data) in enumerate(residuals.items()):
            # Residuals plot
            axes[0, i].scatter(range(len(resid_data)), resid_data, 
                             color=colors[i % len(colors)], alpha=0.6)
            axes[0, i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[0, i].set_title(f'{model_name} Residuals')
            axes[0, i].set_ylabel('Residuals')
            axes[0, i].grid(True, alpha=0.3)
            
            # Residuals vs fitted
            fitted_values = np.arange(len(resid_data))
            axes[1, i].scatter(fitted_values, resid_data, 
                             color=colors[i % len(colors)], alpha=0.6)
            axes[1, i].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, i].set_title(f'{model_name} vs Fitted')
            axes[1, i].set_xlabel('Fitted Values')
            axes[1, i].set_ylabel('Residuals')
            axes[1, i].grid(True, alpha=0.3)
        
        fig.suptitle('Residual Analysis')
        fig.tight_layout()
        self.figure = fig
    
    # Required abstract method implementations
    def _add_data_series_impl(self, name: str, data: Any, style: Dict[str, Any]) -> None:
        """Add data series implementation."""
        if self.backend == 'plotly':
            import plotly.graph_objects as go
            self.figure.add_trace(go.Scatter(
                y=data,
                mode='lines',
                name=name,
                line=dict(color=style.get('color', 'blue'), width=style.get('width', 1))
            ))
        else:
            ax = self.figure.gca()
            ax.plot(data, label=name, color=style.get('color', 'blue'), 
                   linewidth=style.get('width', 1), alpha=style.get('alpha', 0.8))
            ax.legend()
    
    def _add_annotation_impl(self, annotation: Dict[str, Any]) -> None:
        """Add annotation implementation."""
        # Implementation would depend on annotation type and backend
        pass
    
    def _apply_interactivity_settings(self) -> None:
        """Apply interactivity settings."""
        if self.backend == 'plotly':
            # Plotly interactivity is built-in
            pass
        else:
            # Matplotlib interactivity would require additional setup
            pass
    
    def _update_series_visibility(self, series_name: str, visible: bool) -> None:
        """Update series visibility."""
        # Implementation would depend on backend
        pass
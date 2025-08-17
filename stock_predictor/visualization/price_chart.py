"""Interactive price chart implementation."""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from stock_predictor.utils.logging import get_logger
from .base_chart import InteractiveChart
from .exceptions import ChartCreationError, DataValidationError


class PriceChart(InteractiveChart):
    """Interactive price chart with predictions and confidence intervals."""
    
    def __init__(self, theme: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize price chart.
        
        Args:
            theme: Theme object for styling
            config: Optional configuration dictionary
        """
        super().__init__('price', theme, config)
        
        # Price chart specific state
        self.historical_dates = None
        self.historical_prices = None
        self.model_predictions = {}
        self.ensemble_prediction = None
        self.technical_indicators = {}
        
        # Create the figure
        self.figure = self._create_figure()
        
        self.logger.debug("Price chart initialized")
    
    def _create_figure(self) -> Any:
        """Create the base figure object."""
        if self.backend == 'plotly':
            return self._create_plotly_figure()
        else:
            return self._create_matplotlib_figure()
    
    def _create_plotly_figure(self) -> Any:
        """Create Plotly figure for price chart."""
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Configure layout
        fig.update_layout(
            title='Stock Price Predictions',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def _create_matplotlib_figure(self) -> Any:
        """Create Matplotlib figure for price chart."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=self.config.get('figsize', (12, 8)))
        
        # Configure axes
        ax.set_title('Stock Price Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def add_historical_data(self, dates: Optional[pd.Index], prices: pd.Series) -> None:
        """
        Add historical price data to the chart.
        
        Args:
            dates: Optional date index
            prices: Price series
        """
        try:
            self.historical_dates = dates
            self.historical_prices = prices.values if hasattr(prices, 'values') else prices
            
            # Add to figure
            if self.backend == 'plotly':
                self._add_historical_data_plotly()
            else:
                self._add_historical_data_matplotlib()
            
            self.logger.debug(f"Added historical data with {len(self.historical_prices)} points")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add historical data: {str(e)}")
    
    def add_predictions(self, predictions: Dict[str, np.ndarray]) -> None:
        """
        Add model predictions to the chart.
        
        Args:
            predictions: Dict of {model_name: predictions_array}
        """
        try:
            self.model_predictions.update(predictions)
            
            # Add to figure
            if self.backend == 'plotly':
                self._add_predictions_plotly(predictions)
            else:
                self._add_predictions_matplotlib(predictions)
            
            self.logger.debug(f"Added predictions for {len(predictions)} models")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add predictions: {str(e)}")
    
    def add_ensemble_prediction(self, prediction: np.ndarray, 
                              confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> None:
        """
        Add ensemble prediction to the chart.
        
        Args:
            prediction: Ensemble prediction array
            confidence_intervals: Optional (lower, upper) confidence bounds
        """
        try:
            self.ensemble_prediction = prediction
            
            # Add to figure
            if self.backend == 'plotly':
                self._add_ensemble_prediction_plotly(prediction, confidence_intervals)
            else:
                self._add_ensemble_prediction_matplotlib(prediction, confidence_intervals)
            
            self.logger.debug("Added ensemble prediction")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add ensemble prediction: {str(e)}")
    
    def add_technical_indicators(self, indicators: Dict[str, np.ndarray]) -> None:
        """
        Add technical indicators to the chart.
        
        Args:
            indicators: Dict of {indicator_name: values_array}
        """
        try:
            self.technical_indicators.update(indicators)
            
            # Add to figure with specific styling for different indicator types
            if self.backend == 'plotly':
                self._add_technical_indicators_plotly(indicators)
            else:
                self._add_technical_indicators_matplotlib(indicators)
            
            self.logger.debug(f"Added {len(indicators)} technical indicators")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add technical indicators: {str(e)}")
    
    def add_moving_averages(self, ma_data: Dict[str, np.ndarray]) -> None:
        """
        Add moving averages with specific styling.
        
        Args:
            ma_data: Dict of {period: ma_values} e.g., {'MA20': values, 'MA50': values}
        """
        try:
            colors = ['orange', 'purple', 'green', 'brown']
            
            for i, (name, values) in enumerate(ma_data.items()):
                color = colors[i % len(colors)]
                
                if self.backend == 'plotly':
                    import plotly.graph_objects as go
                    x_data = self.historical_dates if self.historical_dates is not None else list(range(len(values)))
                    
                    self.figure.add_trace(go.Scatter(
                        x=x_data,
                        y=values,
                        mode='lines',
                        name=name,
                        line=dict(color=color, width=1.5, dash='dash'),
                        hovertemplate=f'{name}<br>Date: %{{x}}<br>Value: $%{{y:.2f}}<extra></extra>'
                    ))
                else:
                    ax = self.figure.gca()
                    x_data = self.historical_dates if self.historical_dates is not None else range(len(values))
                    ax.plot(x_data, values, label=name, color=color, linewidth=1.5, linestyle='--', alpha=0.8)
                    ax.legend()
            
            self.logger.debug(f"Added {len(ma_data)} moving averages")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add moving averages: {str(e)}")
    
    def add_bollinger_bands(self, middle: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> None:
        """
        Add Bollinger Bands to the chart.
        
        Args:
            middle: Middle band (usually 20-day MA)
            upper: Upper band
            lower: Lower band
        """
        try:
            if self.backend == 'plotly':
                import plotly.graph_objects as go
                x_data = self.historical_dates if self.historical_dates is not None else list(range(len(middle)))
                
                # Add bands as filled area
                self.figure.add_trace(go.Scatter(
                    x=list(x_data) + list(x_data[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill='toself',
                    fillcolor='rgba(128,128,128,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Bollinger Bands',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add middle line
                self.figure.add_trace(go.Scatter(
                    x=x_data,
                    y=middle,
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='gray', width=1),
                    hovertemplate='BB Middle<br>Date: %{x}<br>Value: $%{y:.2f}<extra></extra>'
                ))
                
            else:
                ax = self.figure.gca()
                x_data = self.historical_dates if self.historical_dates is not None else range(len(middle))
                
                ax.fill_between(x_data, lower, upper, alpha=0.2, color='gray', label='Bollinger Bands')
                ax.plot(x_data, middle, label='BB Middle', color='gray', linewidth=1)
                ax.legend()
            
            self.logger.debug("Added Bollinger Bands")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add Bollinger Bands: {str(e)}")
    
    def add_rsi_subplot(self, rsi_values: np.ndarray) -> None:
        """
        Add RSI as a subplot below the main chart.
        
        Args:
            rsi_values: RSI values array
        """
        try:
            if self.backend == 'plotly':
                # For Plotly, we'd need to create subplots - simplified version here
                import plotly.graph_objects as go
                x_data = self.historical_dates if self.historical_dates is not None else list(range(len(rsi_values)))
                
                # Add RSI as a secondary y-axis (simplified)
                self.figure.add_trace(go.Scatter(
                    x=x_data,
                    y=rsi_values,
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1),
                    yaxis='y2',
                    hovertemplate='RSI<br>Date: %{x}<br>Value: %{y:.1f}<extra></extra>'
                ))
                
                # Update layout for secondary y-axis
                self.figure.update_layout(
                    yaxis2=dict(
                        title='RSI',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    )
                )
                
            else:
                # For matplotlib, create a subplot
                import matplotlib.pyplot as plt
                
                # This would require restructuring the figure - simplified version
                ax = self.figure.gca()
                ax2 = ax.twinx()
                
                x_data = self.historical_dates if self.historical_dates is not None else range(len(rsi_values))
                ax2.plot(x_data, rsi_values, label='RSI', color='purple', linewidth=1)
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax2.legend(loc='upper right')
            
            self.logger.debug("Added RSI indicator")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add RSI: {str(e)}")
    
    def enable_model_toggle(self, models: list) -> None:
        """
        Enable toggling of individual model predictions.
        
        Args:
            models: List of model names to enable toggling for
        """
        # This would be implemented with interactive widgets in a full implementation
        self.logger.debug(f"Model toggle enabled for: {models}")
    
    def set_date_range(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
        """
        Set the visible date range for the chart.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        try:
            if self.backend == 'plotly':
                self.figure.update_layout(
                    xaxis=dict(
                        range=[start_date, end_date] if start_date and end_date else None
                    )
                )
            else:
                ax = self.figure.gca()
                if start_date and end_date:
                    ax.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))
            
            self.logger.debug(f"Set date range: {start_date} to {end_date}")
            
        except Exception as e:
            self.logger.error(f"Failed to set date range: {str(e)}")
    
    def add_price_annotations(self, annotations: list) -> None:
        """
        Add price annotations (e.g., buy/sell signals, events).
        
        Args:
            annotations: List of annotation dicts with 'date', 'price', 'text', 'type'
        """
        try:
            for annotation in annotations:
                date = annotation.get('date')
                price = annotation.get('price')
                text = annotation.get('text', '')
                annotation_type = annotation.get('type', 'info')
                
                # Choose color based on type
                color_map = {
                    'buy': 'green',
                    'sell': 'red',
                    'info': 'blue',
                    'warning': 'orange'
                }
                color = color_map.get(annotation_type, 'blue')
                
                if self.backend == 'plotly':
                    self.figure.add_annotation(
                        x=date,
                        y=price,
                        text=text,
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor=color,
                        font=dict(color=color)
                    )
                else:
                    ax = self.figure.gca()
                    ax.annotate(text, xy=(date, price), xytext=(10, 10),
                               textcoords='offset points', 
                               arrowprops=dict(arrowstyle='->', color=color),
                               color=color)
            
            self.logger.debug(f"Added {len(annotations)} price annotations")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add annotations: {str(e)}")
    
    def add_volume_subplot(self, volume_data: np.ndarray) -> None:
        """
        Add volume data as a subplot.
        
        Args:
            volume_data: Volume data array
        """
        try:
            if self.backend == 'plotly':
                # For Plotly subplots, we'd need to restructure - simplified version
                import plotly.graph_objects as go
                x_data = self.historical_dates if self.historical_dates is not None else list(range(len(volume_data)))
                
                self.figure.add_trace(go.Bar(
                    x=x_data,
                    y=volume_data,
                    name='Volume',
                    yaxis='y2',
                    opacity=0.3,
                    marker_color='gray'
                ))
                
                # Update layout for secondary y-axis
                self.figure.update_layout(
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right'
                    )
                )
                
            else:
                # For matplotlib, use twin axis
                ax = self.figure.gca()
                ax2 = ax.twinx()
                
                x_data = self.historical_dates if self.historical_dates is not None else range(len(volume_data))
                ax2.bar(x_data, volume_data, alpha=0.3, color='gray', label='Volume')
                ax2.set_ylabel('Volume')
                ax2.legend(loc='upper left')
            
            self.logger.debug("Added volume subplot")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add volume: {str(e)}")
    
    def enable_crossfilter(self, enable: bool = True) -> None:
        """
        Enable crossfilter functionality for linked charts.
        
        Args:
            enable: Whether to enable crossfilter
        """
        if self.backend == 'plotly':
            # Plotly supports crossfilter through dash or custom JS
            self.config['crossfilter'] = enable
        
        self.logger.debug(f"Crossfilter {'enabled' if enable else 'disabled'}")
    
    def get_chart_data(self) -> Dict[str, Any]:
        """
        Get all chart data for export or analysis.
        
        Returns:
            Dictionary containing all chart data
        """
        return {
            'historical_dates': self.historical_dates,
            'historical_prices': self.historical_prices,
            'model_predictions': self.model_predictions,
            'ensemble_prediction': self.ensemble_prediction,
            'technical_indicators': self.technical_indicators,
            'config': self.config
        }
    
    # Backend-specific implementation methods
    def _add_historical_data_plotly(self) -> None:
        """Add historical data to Plotly figure."""
        import plotly.graph_objects as go
        
        x_data = self.historical_dates if self.historical_dates is not None else list(range(len(self.historical_prices)))
        
        self.figure.add_trace(go.Scatter(
            x=x_data,
            y=self.historical_prices,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    def _add_historical_data_matplotlib(self) -> None:
        """Add historical data to Matplotlib figure."""
        ax = self.figure.gca()
        x_data = self.historical_dates if self.historical_dates is not None else range(len(self.historical_prices))
        
        ax.plot(x_data, self.historical_prices, 
               label='Actual', color='black', linewidth=2)
        ax.legend()
    
    def _add_predictions_plotly(self, predictions: Dict[str, np.ndarray]) -> None:
        """Add predictions to Plotly figure."""
        import plotly.graph_objects as go
        
        colors = self.theme.get_color_palette(len(predictions))
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            # Create x-axis data (assuming predictions align with end of historical data)
            if self.historical_dates is not None:
                x_data = self.historical_dates[-len(pred):] if len(pred) <= len(self.historical_dates) else self.historical_dates
            else:
                x_data = list(range(len(pred)))
            
            self.figure.add_trace(go.Scatter(
                x=x_data,
                y=pred,
                mode='lines',
                name=f'{model_name} Prediction',
                line=dict(color=colors[i % len(colors)], width=1.5),
                hovertemplate=f'{model_name}<br>Date: %{{x}}<br>Prediction: $%{{y:.2f}}<extra></extra>'
            ))
    
    def _add_predictions_matplotlib(self, predictions: Dict[str, np.ndarray]) -> None:
        """Add predictions to Matplotlib figure."""
        ax = self.figure.gca()
        colors = self.theme.get_color_palette(len(predictions))
        
        for i, (model_name, pred) in enumerate(predictions.items()):
            if self.historical_dates is not None:
                x_data = self.historical_dates[-len(pred):] if len(pred) <= len(self.historical_dates) else self.historical_dates
            else:
                x_data = range(len(pred))
            
            ax.plot(x_data, pred, 
                   label=f'{model_name} Prediction', 
                   color=colors[i % len(colors)], 
                   linewidth=1.5, alpha=0.8)
        
        ax.legend()
    
    def _add_ensemble_prediction_plotly(self, prediction: np.ndarray, 
                                      confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Add ensemble prediction to Plotly figure."""
        import plotly.graph_objects as go
        
        if self.historical_dates is not None:
            x_data = self.historical_dates[-len(prediction):] if len(prediction) <= len(self.historical_dates) else self.historical_dates
        else:
            x_data = list(range(len(prediction)))
        
        # Add confidence intervals first (if provided)
        if confidence_intervals:
            lower, upper = confidence_intervals
            self.figure.add_trace(go.Scatter(
                x=list(x_data) + list(x_data[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Ensemble Confidence',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Add ensemble prediction
        self.figure.add_trace(go.Scatter(
            x=x_data,
            y=prediction,
            mode='lines',
            name='Ensemble Prediction',
            line=dict(color='red', width=3),
            hovertemplate='Ensemble<br>Date: %{x}<br>Prediction: $%{y:.2f}<extra></extra>'
        ))
    
    def _add_ensemble_prediction_matplotlib(self, prediction: np.ndarray, 
                                          confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Add ensemble prediction to Matplotlib figure."""
        ax = self.figure.gca()
        
        if self.historical_dates is not None:
            x_data = self.historical_dates[-len(prediction):] if len(prediction) <= len(self.historical_dates) else self.historical_dates
        else:
            x_data = range(len(prediction))
        
        # Add confidence intervals first (if provided)
        if confidence_intervals:
            lower, upper = confidence_intervals
            ax.fill_between(x_data, lower, upper, alpha=0.3, color='red', label='Ensemble Confidence')
        
        # Add ensemble prediction
        ax.plot(x_data, prediction, 
               label='Ensemble Prediction', color='red', linewidth=3)
        ax.legend()
    
    def _add_technical_indicators_plotly(self, indicators: Dict[str, np.ndarray]) -> None:
        """Add technical indicators to Plotly figure."""
        import plotly.graph_objects as go
        
        indicator_colors = {
            'sma': 'orange',
            'ema': 'purple', 
            'rsi': 'blue',
            'macd': 'green',
            'volume': 'gray'
        }
        
        for name, values in indicators.items():
            # Determine color based on indicator type
            color = 'blue'  # default
            for indicator_type, type_color in indicator_colors.items():
                if indicator_type.lower() in name.lower():
                    color = type_color
                    break
            
            x_data = self.historical_dates if self.historical_dates is not None else list(range(len(values)))
            
            self.figure.add_trace(go.Scatter(
                x=x_data,
                y=values,
                mode='lines',
                name=name,
                line=dict(color=color, width=1.5, dash='dot'),
                hovertemplate=f'{name}<br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
            ))
    
    def _add_technical_indicators_matplotlib(self, indicators: Dict[str, np.ndarray]) -> None:
        """Add technical indicators to Matplotlib figure."""
        ax = self.figure.gca()
        
        indicator_colors = {
            'sma': 'orange',
            'ema': 'purple', 
            'rsi': 'blue',
            'macd': 'green',
            'volume': 'gray'
        }
        
        for name, values in indicators.items():
            # Determine color based on indicator type
            color = 'blue'  # default
            for indicator_type, type_color in indicator_colors.items():
                if indicator_type.lower() in name.lower():
                    color = type_color
                    break
            
            x_data = self.historical_dates if self.historical_dates is not None else range(len(values))
            ax.plot(x_data, values, label=name, color=color, linewidth=1.5, linestyle=':', alpha=0.7)
        
        ax.legend()
    
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
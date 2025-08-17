"""Base interactive chart class with common functionality."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from stock_predictor.utils.logging import get_logger
from .exceptions import ChartCreationError, ExportError, DataValidationError


class InteractiveChart(ABC):
    """Base class for interactive charts with common functionality."""
    
    def __init__(self, chart_type: str, theme: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize interactive chart.
        
        Args:
            chart_type: Type of chart (e.g., 'price', 'performance', 'analysis')
            theme: Theme object for styling
            config: Optional configuration dictionary
        """
        self.chart_type = chart_type
        self.theme = theme
        self.config = config or {}
        self.logger = get_logger(f'visualization.{chart_type}_chart')
        
        # Chart state
        self.figure = None
        self.data_series = {}
        self.annotations = []
        self.callbacks = {}
        self.metadata = {}
        
        # Interactivity settings
        self.interactivity = {
            'zoom': self.config.get('enable_zoom', True),
            'pan': self.config.get('enable_pan', True),
            'hover': self.config.get('enable_hover', True),
            'select': self.config.get('enable_select', False)
        }
        
        # Backend detection
        self.backend = self._detect_backend()
        
        self.logger.debug(f"Initialized {chart_type} chart with {self.backend} backend")
    
    def _detect_backend(self) -> str:
        """Detect available visualization backend."""
        backend_preference = self.config.get('backend', 'plotly')
        
        if backend_preference == 'plotly':
            try:
                import plotly.graph_objects as go
                return 'plotly'
            except ImportError:
                self.logger.warning("Plotly not available, falling back to matplotlib")
        
        try:
            import matplotlib.pyplot as plt
            return 'matplotlib'
        except ImportError:
            raise ChartCreationError("No visualization backend available. Install matplotlib or plotly.")
    
    @abstractmethod
    def _create_figure(self) -> Any:
        """Create the base figure object. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _add_data_series_impl(self, name: str, data: Any, style: Dict[str, Any]) -> None:
        """Add data series to chart. Must be implemented by subclasses."""
        pass
    
    def add_data_series(self, name: str, data: Union[np.ndarray, pd.Series], 
                       style: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a data series to the chart.
        
        Args:
            name: Name of the data series
            data: Data array or series
            style: Optional styling configuration
        """
        try:
            # Validate data
            if data is None or len(data) == 0:
                raise DataValidationError(f"Empty data provided for series '{name}'")
            
            # Convert to numpy array if needed
            if isinstance(data, pd.Series):
                data = data.values
            elif not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Apply default styling
            default_style = self.theme.get_series_style(name, self.chart_type) if hasattr(self.theme, 'get_series_style') else {'color': 'blue', 'width': 2}
            if style:
                default_style.update(style)
            
            # Store series data
            self.data_series[name] = {
                'data': data,
                'style': default_style,
                'visible': True
            }
            
            # Add to figure
            self._add_data_series_impl(name, data, default_style)
            
            self.logger.debug(f"Added data series '{name}' with {len(data)} points")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add data series '{name}': {str(e)}")
    
    def add_confidence_bands(self, name: str, 
                           lower_bounds: np.ndarray, 
                           upper_bounds: np.ndarray,
                           style: Optional[Dict[str, Any]] = None) -> None:
        """
        Add confidence bands to the chart.
        
        Args:
            name: Name for the confidence bands
            lower_bounds: Lower confidence bounds
            upper_bounds: Upper confidence bounds
            style: Optional styling configuration
        """
        try:
            # Validate data
            if len(lower_bounds) != len(upper_bounds):
                raise DataValidationError("Lower and upper bounds must have same length")
            
            # Apply default styling for confidence bands
            default_style = self.theme.get_confidence_band_style(name) if hasattr(self.theme, 'get_confidence_band_style') else {'alpha': 0.3, 'color': 'gray'}
            if style:
                default_style.update(style)
            
            # Add confidence bands based on backend
            if self.backend == 'plotly':
                self._add_plotly_confidence_bands(name, lower_bounds, upper_bounds, default_style)
            else:
                self._add_matplotlib_confidence_bands(name, lower_bounds, upper_bounds, default_style)
            
            self.logger.debug(f"Added confidence bands '{name}'")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add confidence bands '{name}': {str(e)}")
    
    def add_annotations(self, annotations: List[Dict[str, Any]]) -> None:
        """
        Add annotations to the chart.
        
        Args:
            annotations: List of annotation dictionaries
        """
        try:
            for annotation in annotations:
                self._add_annotation_impl(annotation)
                self.annotations.append(annotation)
            
            self.logger.debug(f"Added {len(annotations)} annotations")
            
        except Exception as e:
            raise ChartCreationError(f"Failed to add annotations: {str(e)}")
    
    def set_interactivity(self, zoom: bool = True, pan: bool = True, 
                         hover: bool = True, select: bool = False) -> None:
        """
        Configure chart interactivity settings.
        
        Args:
            zoom: Enable zoom functionality
            pan: Enable pan functionality
            hover: Enable hover tooltips
            select: Enable data selection
        """
        self.interactivity.update({
            'zoom': zoom,
            'pan': pan,
            'hover': hover,
            'select': select
        })
        
        # Apply interactivity settings to figure
        if self.figure:
            self._apply_interactivity_settings()
        
        self.logger.debug(f"Updated interactivity settings: {self.interactivity}")
    
    def toggle_series_visibility(self, series_name: str, visible: Optional[bool] = None) -> None:
        """
        Toggle or set visibility of a data series.
        
        Args:
            series_name: Name of the series to toggle
            visible: Optional explicit visibility setting
        """
        if series_name not in self.data_series:
            raise ChartCreationError(f"Series '{series_name}' not found")
        
        if visible is None:
            visible = not self.data_series[series_name]['visible']
        
        self.data_series[series_name]['visible'] = visible
        self._update_series_visibility(series_name, visible)
        
        self.logger.debug(f"Set series '{series_name}' visibility to {visible}")
    
    def export(self, filepath: Union[str, Path], format: str = 'png', **kwargs) -> None:
        """
        Export chart to file.
        
        Args:
            filepath: Path to save the exported chart
            format: Export format ('png', 'svg', 'pdf', 'html')
            **kwargs: Additional export options
        """
        try:
            filepath = Path(filepath)
            
            # Ensure directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Export based on backend and format
            if self.backend == 'plotly':
                self._export_plotly(filepath, format, **kwargs)
            else:
                self._export_matplotlib(filepath, format, **kwargs)
            
            self.logger.info(f"Chart exported to: {filepath}")
            
        except Exception as e:
            raise ExportError(f"Failed to export chart: {str(e)}")
    
    def get_figure(self) -> Any:
        """Get the underlying figure object."""
        return self.figure
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of chart data."""
        return {
            'chart_type': self.chart_type,
            'backend': self.backend,
            'series_count': len(self.data_series),
            'series_names': list(self.data_series.keys()),
            'annotations_count': len(self.annotations),
            'interactivity': self.interactivity.copy()
        }
    
    # Abstract methods that subclasses may need to implement
    @abstractmethod
    def _add_annotation_impl(self, annotation: Dict[str, Any]) -> None:
        """Add annotation implementation. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _apply_interactivity_settings(self) -> None:
        """Apply interactivity settings. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _update_series_visibility(self, series_name: str, visible: bool) -> None:
        """Update series visibility. Must be implemented by subclasses."""
        pass
    
    # Backend-specific helper methods
    def _add_plotly_confidence_bands(self, name: str, lower: np.ndarray, 
                                   upper: np.ndarray, style: Dict[str, Any]) -> None:
        """Add confidence bands for Plotly backend."""
        if self.backend != 'plotly':
            return
        
        import plotly.graph_objects as go
        
        x_data = list(range(len(lower)))
        
        # Create filled area
        self.figure.add_trace(go.Scatter(
            x=x_data + x_data[::-1],
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor=style.get('fillcolor', 'rgba(0,100,80,0.2)'),
            line=dict(color='rgba(255,255,255,0)'),
            name=name,
            showlegend=style.get('showlegend', True),
            hoverinfo='skip'
        ))
    
    def _add_matplotlib_confidence_bands(self, name: str, lower: np.ndarray, 
                                       upper: np.ndarray, style: Dict[str, Any]) -> None:
        """Add confidence bands for Matplotlib backend."""
        if self.backend != 'matplotlib':
            return
        
        import matplotlib.pyplot as plt
        
        x_data = range(len(lower))
        
        # Add filled area
        plt.fill_between(
            x_data, lower, upper,
            alpha=style.get('alpha', 0.3),
            color=style.get('color', 'gray'),
            label=name
        )
    
    def _export_plotly(self, filepath: Path, format: str, **kwargs) -> None:
        """Export Plotly figure."""
        if format.lower() == 'html':
            self.figure.write_html(str(filepath), **kwargs)
        elif format.lower() in ['png', 'pdf', 'svg']:
            self.figure.write_image(str(filepath), format=format, **kwargs)
        else:
            raise ExportError(f"Unsupported export format for Plotly: {format}")
    
    def _export_matplotlib(self, filepath: Path, format: str, **kwargs) -> None:
        """Export Matplotlib figure."""
        if self.figure:
            dpi = kwargs.get('dpi', 300)
            bbox_inches = kwargs.get('bbox_inches', 'tight')
            self.figure.savefig(str(filepath), format=format, dpi=dpi, 
                              bbox_inches=bbox_inches, **kwargs)
        else:
            raise ExportError("No figure available for export")
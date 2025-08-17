"""Theme management system for consistent visualization styling."""

from typing import Dict, Any, List, Optional
import json
import yaml
from pathlib import Path

from stock_predictor.utils.logging import get_logger
from .exceptions import ThemeError, ConfigurationError


class Theme:
    """Theme configuration for consistent styling."""
    
    def __init__(self, name: str, config: Dict[str, Any], parent_theme: Optional['Theme'] = None):
        """
        Initialize theme.
        
        Args:
            name: Theme name
            config: Theme configuration dictionary
            parent_theme: Parent theme for inheritance
        """
        self.name = name
        self.parent_theme = parent_theme
        
        # Merge with parent theme if provided
        if parent_theme:
            self.config = self._merge_configs(parent_theme.config, config)
        else:
            self.config = config.copy()
        
        # Core theme components
        self.colors = self.config.get('colors', {})
        self.fonts = self.config.get('fonts', {})
        self.layout = self.config.get('layout', {})
        self.export_settings = self.config.get('export', {})
        
        # Validate theme configuration
        self._validate_config()
    
    def _merge_configs(self, parent_config: Dict[str, Any], child_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge child configuration with parent configuration.
        
        Args:
            parent_config: Parent theme configuration
            child_config: Child theme configuration
            
        Returns:
            Merged configuration
        """
        merged = parent_config.copy()
        
        for key, value in child_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override with child value
                merged[key] = value
        
        return merged
    
    def _validate_config(self) -> None:
        """Validate theme configuration."""
        required_colors = ['primary', 'secondary', 'background', 'text']
        for color in required_colors:
            if color not in self.colors:
                raise ThemeError(f"Theme '{self.name}' missing required color: {color}")
        
        # Validate color format (hex colors)
        for color_name, color_value in self.colors.items():
            if isinstance(color_value, str) and color_value.startswith('#'):
                if not self._is_valid_hex_color(color_value):
                    raise ThemeError(f"Theme '{self.name}' has invalid hex color '{color_name}': {color_value}")
    
    def _is_valid_hex_color(self, color: str) -> bool:
        """Validate hex color format."""
        if not color.startswith('#'):
            return False
        
        hex_part = color[1:]
        if len(hex_part) not in [3, 6]:
            return False
        
        try:
            int(hex_part, 16)
            return True
        except ValueError:
            return False
    
    def get_color(self, color_name: str, default: str = '#000000') -> str:
        """Get color by name with fallback."""
        return self.colors.get(color_name, default)
    
    def get_color_palette(self, n_colors: int, palette_type: str = 'default') -> List[str]:
        """
        Generate color palette with specified number of colors.
        
        Args:
            n_colors: Number of colors needed
            palette_type: Type of palette ('default', 'sequential', 'diverging', 'qualitative')
            
        Returns:
            List of color strings
        """
        if palette_type == 'sequential':
            return self._generate_sequential_palette(n_colors)
        elif palette_type == 'diverging':
            return self._generate_diverging_palette(n_colors)
        elif palette_type == 'qualitative':
            return self._generate_qualitative_palette(n_colors)
        else:
            # Default palette
            base_colors = [
                self.colors.get('primary', '#1f77b4'),
                self.colors.get('secondary', '#ff7f0e'),
                self.colors.get('accent1', '#2ca02c'),
                self.colors.get('accent2', '#d62728'),
                self.colors.get('accent3', '#9467bd'),
                self.colors.get('accent4', '#8c564b'),
                self.colors.get('accent5', '#e377c2'),
                self.colors.get('accent6', '#7f7f7f')
            ]
            
            # Repeat colors if needed
            while len(base_colors) < n_colors:
                base_colors.extend(base_colors)
            
            return base_colors[:n_colors]
    
    def _generate_sequential_palette(self, n_colors: int) -> List[str]:
        """Generate sequential color palette."""
        primary = self.colors.get('primary', '#1f77b4')
        return self._interpolate_colors(primary, '#ffffff', n_colors)
    
    def _generate_diverging_palette(self, n_colors: int) -> List[str]:
        """Generate diverging color palette."""
        color1 = self.colors.get('accent2', '#d62728')  # Red
        color2 = self.colors.get('accent1', '#2ca02c')  # Green
        mid_color = '#ffffff'
        
        if n_colors % 2 == 1:
            # Odd number - include middle color
            half = n_colors // 2
            left_colors = self._interpolate_colors(color1, mid_color, half + 1)[:-1]
            right_colors = self._interpolate_colors(mid_color, color2, half + 1)[1:]
            return left_colors + [mid_color] + right_colors
        else:
            # Even number
            half = n_colors // 2
            left_colors = self._interpolate_colors(color1, mid_color, half + 1)[:-1]
            right_colors = self._interpolate_colors(mid_color, color2, half + 1)[1:]
            return left_colors + right_colors
    
    def _generate_qualitative_palette(self, n_colors: int) -> List[str]:
        """Generate qualitative color palette with maximum contrast."""
        # Use HSV color space for better distribution
        import colorsys
        
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            saturation = 0.7
            value = 0.8
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgb[0] * 255),
                int(rgb[1] * 255),
                int(rgb[2] * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    def _interpolate_colors(self, color1: str, color2: str, n_steps: int) -> List[str]:
        """Interpolate between two colors."""
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        rgb1 = hex_to_rgb(color1)
        rgb2 = hex_to_rgb(color2)
        
        colors = []
        for i in range(n_steps):
            ratio = i / (n_steps - 1) if n_steps > 1 else 0
            rgb = tuple(
                rgb1[j] + ratio * (rgb2[j] - rgb1[j])
                for j in range(3)
            )
            colors.append(rgb_to_hex(rgb))
        
        return colors
    
    def get_font_config(self, element: str) -> Dict[str, Any]:
        """Get font configuration for specific element."""
        return self.fonts.get(element, {
            'family': 'Arial, sans-serif',
            'size': 12,
            'weight': 'normal'
        })
    
    def get_series_style(self, series_name: str, chart_type: str) -> Dict[str, Any]:
        """Get styling for data series."""
        # Default series styling
        default_style = {
            'color': self.get_color('primary'),
            'width': 2,
            'alpha': 0.8,
            'linestyle': '-'
        }
        
        # Chart-specific overrides
        chart_styles = self.config.get('chart_styles', {}).get(chart_type, {})
        series_styles = chart_styles.get('series', {}).get(series_name, {})
        
        default_style.update(series_styles)
        return default_style
    
    def get_confidence_band_style(self, name: str) -> Dict[str, Any]:
        """Get styling for confidence bands."""
        return {
            'fillcolor': f"rgba({self._hex_to_rgb(self.get_color('primary'))}, 0.2)",
            'color': self.get_color('primary'),
            'alpha': 0.3,
            'showlegend': True
        }
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f"{r},{g},{b}"
        return "0,0,0"


class ThemeManager:
    """Manages themes and styling across all visualizations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize theme manager.
        
        Args:
            config: Theme configuration dictionary
        """
        self.logger = get_logger('visualization.theme_manager')
        self.config = config
        self.themes = {}
        self.current_theme = None
        
        # Load built-in themes
        self._load_builtin_themes()
        
        # Load themes from themes directory
        themes_dir = Path(__file__).parent / 'themes'
        self.load_themes_from_directory(themes_dir)
        
        # Load custom themes from config
        self._load_custom_themes(config)
        
        # Set default theme
        default_theme_name = config.get('default', 'light')
        if default_theme_name in self.themes:
            self.current_theme = self.themes[default_theme_name]
        else:
            self.current_theme = self.themes['light']  # Fallback
        
        self.logger.info(f"Theme manager initialized with {len(self.themes)} themes")
    
    def _load_builtin_themes(self) -> None:
        """Load built-in themes."""
        # Light theme
        light_theme_config = {
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent1': '#2ca02c',
                'accent2': '#d62728',
                'accent3': '#9467bd',
                'accent4': '#8c564b',
                'background': '#ffffff',
                'text': '#000000',
                'grid': '#e0e0e0'
            },
            'fonts': {
                'title': {'family': 'Arial, sans-serif', 'size': 16, 'weight': 'bold'},
                'axis': {'family': 'Arial, sans-serif', 'size': 12, 'weight': 'normal'},
                'legend': {'family': 'Arial, sans-serif', 'size': 10, 'weight': 'normal'}
            },
            'layout': {
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
                'grid': True,
                'grid_alpha': 0.3
            }
        }
        
        # Dark theme
        dark_theme_config = {
            'colors': {
                'primary': '#8dd3c7',
                'secondary': '#ffd92f',
                'accent1': '#b3de69',
                'accent2': '#fccde5',
                'accent3': '#d9d9d9',
                'accent4': '#bc80bd',
                'background': '#2f2f2f',
                'text': '#ffffff',
                'grid': '#555555'
            },
            'fonts': {
                'title': {'family': 'Arial, sans-serif', 'size': 16, 'weight': 'bold'},
                'axis': {'family': 'Arial, sans-serif', 'size': 12, 'weight': 'normal'},
                'legend': {'family': 'Arial, sans-serif', 'size': 10, 'weight': 'normal'}
            },
            'layout': {
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
                'grid': True,
                'grid_alpha': 0.3
            }
        }
        
        self.themes['light'] = Theme('light', light_theme_config)
        self.themes['dark'] = Theme('dark', dark_theme_config)
    
    def _load_custom_themes(self, config: Dict[str, Any]) -> None:
        """Load custom themes from configuration."""
        custom_themes = config.get('custom_themes', {})
        
        for theme_name, theme_config in custom_themes.items():
            try:
                self.themes[theme_name] = Theme(theme_name, theme_config)
                self.logger.debug(f"Loaded custom theme: {theme_name}")
            except Exception as e:
                self.logger.error(f"Failed to load custom theme '{theme_name}': {str(e)}")
    
    def get_current_theme(self) -> Theme:
        """Get the current active theme."""
        return self.current_theme
    
    def set_current_theme(self, theme_name: str) -> None:
        """
        Set the current active theme.
        
        Args:
            theme_name: Name of the theme to activate
        """
        if theme_name not in self.themes:
            raise ThemeError(f"Theme '{theme_name}' not found")
        
        self.current_theme = self.themes[theme_name]
        self.logger.info(f"Active theme changed to: {theme_name}")
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self.themes.keys())
    
    def create_custom_theme(self, name: str, config: Dict[str, Any], parent_theme_name: Optional[str] = None) -> Theme:
        """
        Create a new custom theme with optional inheritance.
        
        Args:
            name: Theme name
            config: Theme configuration
            parent_theme_name: Name of parent theme for inheritance
            
        Returns:
            Created theme object
        """
        try:
            parent_theme = None
            if parent_theme_name:
                if parent_theme_name not in self.themes:
                    raise ThemeError(f"Parent theme '{parent_theme_name}' not found")
                parent_theme = self.themes[parent_theme_name]
            
            theme = Theme(name, config, parent_theme)
            self.themes[name] = theme
            self.logger.info(f"Created custom theme: {name}" + 
                           (f" (inherits from {parent_theme_name})" if parent_theme_name else ""))
            return theme
            
        except Exception as e:
            raise ThemeError(f"Failed to create custom theme '{name}': {str(e)}")
    
    def save_theme(self, theme_name: str, filepath: Path, format: str = 'auto') -> None:
        """
        Save theme configuration to file.
        
        Args:
            theme_name: Name of theme to save
            filepath: Path to save theme file
            format: File format ('json', 'yaml', or 'auto' to detect from extension)
        """
        if theme_name not in self.themes:
            raise ThemeError(f"Theme '{theme_name}' not found")
        
        try:
            theme_config = self.themes[theme_name].config
            
            # Determine format
            if format == 'auto':
                format = 'yaml' if filepath.suffix.lower() in ['.yml', '.yaml'] else 'json'
            
            with open(filepath, 'w') as f:
                if format == 'yaml':
                    yaml.dump(theme_config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(theme_config, f, indent=2)
            
            self.logger.info(f"Theme '{theme_name}' saved to: {filepath} (format: {format})")
            
        except Exception as e:
            raise ThemeError(f"Failed to save theme: {str(e)}")
    
    def load_theme_from_file(self, name: str, filepath: Path) -> Theme:
        """
        Load theme from file (supports JSON and YAML).
        
        Args:
            name: Name for the loaded theme
            filepath: Path to theme file
            
        Returns:
            Loaded theme object
        """
        try:
            with open(filepath, 'r') as f:
                # Auto-detect format based on file extension
                if filepath.suffix.lower() in ['.yml', '.yaml']:
                    theme_config = yaml.safe_load(f)
                else:
                    theme_config = json.load(f)
            
            theme = Theme(name, theme_config)
            self.themes[name] = theme
            
            self.logger.info(f"Theme '{name}' loaded from: {filepath}")
            return theme
            
        except Exception as e:
            raise ThemeError(f"Failed to load theme from file: {str(e)}")
    
    def load_themes_from_directory(self, themes_dir: Path) -> int:
        """
        Load all theme files from a directory.
        
        Args:
            themes_dir: Directory containing theme files
            
        Returns:
            Number of themes loaded
        """
        if not themes_dir.exists() or not themes_dir.is_dir():
            self.logger.warning(f"Themes directory not found: {themes_dir}")
            return 0
        
        loaded_count = 0
        
        # Look for theme files
        for theme_file in themes_dir.glob('*.yaml'):
            try:
                theme_name = theme_file.stem
                self.load_theme_from_file(theme_name, theme_file)
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"Failed to load theme from {theme_file}: {str(e)}")
        
        for theme_file in themes_dir.glob('*.yml'):
            try:
                theme_name = theme_file.stem
                self.load_theme_from_file(theme_name, theme_file)
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"Failed to load theme from {theme_file}: {str(e)}")
        
        for theme_file in themes_dir.glob('*.json'):
            try:
                theme_name = theme_file.stem
                self.load_theme_from_file(theme_name, theme_file)
                loaded_count += 1
            except Exception as e:
                self.logger.error(f"Failed to load theme from {theme_file}: {str(e)}")
        
        self.logger.info(f"Loaded {loaded_count} themes from directory: {themes_dir}")
        return loaded_count
    
    def apply_theme_to_figure(self, figure: Any, chart_type: str) -> None:
        """
        Apply current theme to a figure object.
        
        Args:
            figure: Figure object to style
            chart_type: Type of chart for specific styling
        """
        try:
            if hasattr(figure, 'update_layout'):  # Plotly figure
                self._apply_plotly_theme(figure, chart_type)
            elif hasattr(figure, 'patch'):  # Matplotlib figure
                self._apply_matplotlib_theme(figure, chart_type)
            
        except Exception as e:
            self.logger.error(f"Failed to apply theme to figure: {str(e)}")
    
    def _apply_plotly_theme(self, figure: Any, chart_type: str) -> None:
        """Apply theme to Plotly figure."""
        theme = self.current_theme
        
        # Update layout with theme colors and fonts
        figure.update_layout(
            plot_bgcolor=theme.get_color('background'),
            paper_bgcolor=theme.get_color('background'),
            font=dict(
                family=theme.get_font_config('axis')['family'],
                size=theme.get_font_config('axis')['size'],
                color=theme.get_color('text')
            ),
            title_font=dict(
                family=theme.get_font_config('title')['family'],
                size=theme.get_font_config('title')['size'],
                color=theme.get_color('text')
            ),
            legend=dict(
                font=dict(
                    family=theme.get_font_config('legend')['family'],
                    size=theme.get_font_config('legend')['size'],
                    color=theme.get_color('text')
                )
            ),
            xaxis=dict(
                gridcolor=theme.get_color('grid'),
                color=theme.get_color('text')
            ),
            yaxis=dict(
                gridcolor=theme.get_color('grid'),
                color=theme.get_color('text')
            )
        )
    
    def _apply_matplotlib_theme(self, figure: Any, chart_type: str) -> None:
        """Apply theme to Matplotlib figure."""
        import matplotlib.pyplot as plt
        
        theme = self.current_theme
        
        # Set figure background
        figure.patch.set_facecolor(theme.get_color('background'))
        
        # Apply to all axes
        for ax in figure.get_axes():
            ax.set_facecolor(theme.get_color('background'))
            ax.tick_params(colors=theme.get_color('text'))
            ax.xaxis.label.set_color(theme.get_color('text'))
            ax.yaxis.label.set_color(theme.get_color('text'))
            ax.title.set_color(theme.get_color('text'))
            
            # Grid styling
            if theme.layout.get('grid', True):
                ax.grid(True, alpha=theme.layout.get('grid_alpha', 0.3),
                       color=theme.get_color('grid'))
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """Get color palette from current theme."""
        return self.current_theme.get_color_palette(n_colors)
    
    def get_font_config(self, element: str) -> Dict[str, Any]:
        """Get font configuration from current theme."""
        return self.current_theme.get_font_config(element)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update theme manager configuration.
        
        Args:
            new_config: New configuration to merge
        """
        try:
            self.config.update(new_config)
            
            # Reload custom themes if provided
            if 'custom_themes' in new_config:
                self._load_custom_themes(new_config)
            
            # Update default theme if specified
            if 'default' in new_config and new_config['default'] in self.themes:
                self.current_theme = self.themes[new_config['default']]
            
            self.logger.info("Theme manager configuration updated")
            
        except Exception as e:
            raise ConfigurationError(f"Failed to update theme configuration: {str(e)}")
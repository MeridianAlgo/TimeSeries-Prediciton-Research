#!/usr/bin/env python3
"""Test script for enhanced theme manager functionality."""

import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor.visualization.theme_manager import ThemeManager, Theme


def test_theme_manager():
    """Test the enhanced theme manager functionality."""
    print("🎨 Testing Enhanced Theme Manager")
    print("=" * 50)
    
    # Initialize theme manager
    config = {
        'default': 'light',
        'custom_themes': {}
    }
    
    try:
        theme_manager = ThemeManager(config)
        print(f"✅ Theme manager initialized successfully")
        
        # Test available themes
        themes = theme_manager.get_available_themes()
        print(f"📋 Available themes: {', '.join(themes)}")
        
        # Test current theme
        current = theme_manager.get_current_theme()
        print(f"🎯 Current theme: {current.name}")
        
        # Test color palette generation
        print("\n🎨 Testing color palette generation:")
        for palette_type in ['default', 'sequential', 'diverging', 'qualitative']:
            colors = current.get_color_palette(5, palette_type)
            print(f"  {palette_type}: {colors}")
        
        # Test theme switching
        print("\n🔄 Testing theme switching:")
        for theme_name in themes[:3]:  # Test first 3 themes
            theme_manager.set_current_theme(theme_name)
            current = theme_manager.get_current_theme()
            print(f"  Switched to: {current.name}")
            print(f"    Primary color: {current.get_color('primary')}")
            print(f"    Background: {current.get_color('background')}")
        
        # Test custom theme creation
        print("\n🛠️ Testing custom theme creation:")
        custom_config = {
            'colors': {
                'primary': '#FF5733',
                'secondary': '#33FF57',
                'background': '#F0F0F0',
                'text': '#333333'
            },
            'fonts': {
                'title': {'family': 'Arial', 'size': 16, 'weight': 'bold'}
            }
        }
        
        custom_theme = theme_manager.create_custom_theme('test_custom', custom_config)
        print(f"  Created custom theme: {custom_theme.name}")
        print(f"    Primary color: {custom_theme.get_color('primary')}")
        
        # Test theme inheritance
        print("\n🔗 Testing theme inheritance:")
        inherited_config = {
            'colors': {
                'primary': '#9933FF'  # Override only primary color
            }
        }
        
        inherited_theme = theme_manager.create_custom_theme(
            'inherited_test', 
            inherited_config, 
            parent_theme_name='light'
        )
        print(f"  Created inherited theme: {inherited_theme.name}")
        print(f"    Primary color (overridden): {inherited_theme.get_color('primary')}")
        print(f"    Background (inherited): {inherited_theme.get_color('background')}")
        
        # Test theme saving and loading
        print("\n💾 Testing theme save/load:")
        save_path = Path('test_theme.yaml')
        theme_manager.save_theme('test_custom', save_path, 'yaml')
        print(f"  Saved theme to: {save_path}")
        
        # Load it back with a different name
        loaded_theme = theme_manager.load_theme_from_file('loaded_test', save_path)
        print(f"  Loaded theme: {loaded_theme.name}")
        print(f"    Primary color: {loaded_theme.get_color('primary')}")
        
        # Clean up
        save_path.unlink()
        
        print("\n✅ All theme manager tests passed!")
        
    except Exception as e:
        print(f"❌ Error testing theme manager: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_theme_manager()
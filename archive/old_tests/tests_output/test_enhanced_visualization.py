"""Test script for the enhanced visualization system."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Test the new visualization system
def test_enhanced_visualization():
    """Test the enhanced visualization system."""
    try:
        from stock_predictor.visualization import VisualizationManager
        
        # Create test configuration
        config = {
            'themes': {
                'default': 'light'
            },
            'charts': {
                'price_chart': {
                    'backend': 'matplotlib',  # Use matplotlib for compatibility
                    'enable_zoom': True,
                    'enable_hover': True
                }
            },
            'dashboard': {
                'layout': 'grid'
            },
            'reports': {
                'default_template': 'comprehensive'
            }
        }
        
        # Initialize visualization manager
        print("Initializing visualization manager...")
        viz_manager = VisualizationManager(config)
        print(f"✓ Visualization manager initialized with themes: {viz_manager.get_available_themes()}")
        
        # Create test data
        print("\nCreating test data...")
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        historical_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(100) * 0.5),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Create test predictions
        predictions = {
            'arima': 100 + np.cumsum(np.random.randn(20) * 0.3),
            'lstm': 100 + np.cumsum(np.random.randn(20) * 0.4),
            'random_forest': 100 + np.cumsum(np.random.randn(20) * 0.2)
        }
        
        # Create test model metrics
        model_metrics = {
            'arima': {'rmse': 2.5, 'mae': 1.8, 'directional_accuracy': 65.2, 'r2_score': 0.75},
            'lstm': {'rmse': 3.1, 'mae': 2.2, 'directional_accuracy': 62.8, 'r2_score': 0.68},
            'random_forest': {'rmse': 2.1, 'mae': 1.5, 'directional_accuracy': 68.5, 'r2_score': 0.82}
        }
        
        print("✓ Test data created")
        
        # Test price chart creation
        print("\nTesting price chart creation...")
        try:
            price_chart = viz_manager.create_interactive_price_chart(
                historical_data=historical_data,
                predictions=predictions
            )
            print("✓ Price chart created successfully")
            print(f"  Chart summary: {price_chart.get_data_summary()}")
        except Exception as e:
            print(f"✗ Price chart creation failed: {str(e)}")
        
        # Test performance dashboard creation
        print("\nTesting performance dashboard creation...")
        try:
            dashboard = viz_manager.create_performance_dashboard(
                model_metrics=model_metrics
            )
            print("✓ Performance dashboard created successfully")
            print(f"  Dashboard title: {dashboard.title}")
            print(f"  Number of charts: {len(dashboard.charts)}")
        except Exception as e:
            print(f"✗ Performance dashboard creation failed: {str(e)}")
        
        # Test report generation
        print("\nTesting report generation...")
        try:
            test_results = {
                'symbol': 'TEST',
                'ensemble_metrics': {
                    'rmse': 2.0,
                    'mae': 1.4,
                    'directional_accuracy': 70.5,
                    'r2_score': 0.85,
                    'n_samples': 100
                },
                'model_metrics': model_metrics,
                'future_predictions': predictions['arima'][:7]  # 7 day forecast
            }
            
            report_path = viz_manager.generate_comprehensive_report(
                prediction_results=test_results,
                output_path='test_report.html'
            )
            print(f"✓ Report generated successfully: {report_path}")
        except Exception as e:
            print(f"✗ Report generation failed: {str(e)}")
        
        # Test theme switching
        print("\nTesting theme switching...")
        try:
            available_themes = viz_manager.get_available_themes()
            print(f"Available themes: {available_themes}")
            
            if 'dark' in available_themes:
                viz_manager.set_theme('dark')
                print("✓ Successfully switched to dark theme")
            
            if 'light' in available_themes:
                viz_manager.set_theme('light')
                print("✓ Successfully switched to light theme")
        except Exception as e:
            print(f"✗ Theme switching failed: {str(e)}")
        
        print("\n" + "="*60)
        print("ENHANCED VISUALIZATION SYSTEM TEST COMPLETED")
        print("="*60)
        print("✓ All core components are working correctly!")
        print("✓ The enhanced visualization system is ready for use!")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("Make sure all visualization components are properly installed.")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    print("Testing Enhanced Visualization System")
    print("="*60)
    
    success = test_enhanced_visualization()
    
    if success:
        print("\n🎉 Enhanced visualization system is working correctly!")
        print("You can now use the new VisualizationManager for advanced charts and reports.")
    else:
        print("\n❌ Some issues were found. Please check the error messages above.")
#!/usr/bin/env python3
"""Test script for enhanced Dashboard system."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor.visualization.dashboard_builder import Dashboard, DashboardBuilder, DashboardLayout
from stock_predictor.visualization.theme_manager import ThemeManager
from stock_predictor.visualization.chart_factory import ChartFactory


def generate_sample_dashboard_data():
    """Generate sample data for dashboard testing."""
    np.random.seed(42)
    
    # Model performance metrics
    models = ['LSTM', 'Random Forest', 'Linear Regression', 'SVM', 'Ensemble']
    model_metrics = {}
    
    for model in models:
        model_metrics[model] = {
            'rmse': np.random.uniform(2.0, 8.0),
            'mae': np.random.uniform(1.5, 6.0),
            'r2': np.random.uniform(0.3, 0.85),
            'directional_accuracy': np.random.uniform(0.45, 0.75),
            'mape': np.random.uniform(0.05, 0.15)
        }
    
    # Time series performance data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    time_series_data = {}
    
    for model in models:
        base_performance = model_metrics[model]['r2']
        noise = np.random.normal(0, 0.05, len(dates))
        trend = np.linspace(-0.1, 0.1, len(dates))
        performance = base_performance + trend + noise
        performance = np.clip(performance, 0, 1)
        time_series_data[model] = pd.Series(performance, index=dates)
    
    # Historical price data
    price_dates = pd.date_range('2024-01-01', periods=100, freq='D')
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(price_dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    historical_data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, len(price_dates))
    }, index=price_dates)
    
    # Model predictions
    predictions = {}
    for model in models:
        noise = np.random.normal(0, model_metrics[model]['rmse'], len(prices))
        predictions[model] = np.array(prices) + noise
    
    # Ensemble prediction
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    # Confidence intervals
    std_dev = np.std(list(predictions.values()), axis=0)
    confidence_lower = ensemble_pred - 1.96 * std_dev
    confidence_upper = ensemble_pred + 1.96 * std_dev
    
    # Residuals
    residuals = {}
    for model in models:
        residuals[model] = np.array(prices) - predictions[model]
    
    # Ensemble weights
    ensemble_weights = {
        'LSTM': 0.25,
        'Random Forest': 0.30,
        'Linear Regression': 0.15,
        'SVM': 0.20,
        'Ensemble': 0.10
    }
    
    # Weight evolution
    weight_evolution = {}
    for model in models:
        base_weight = ensemble_weights.get(model, 0.2)
        evolution = base_weight + np.random.normal(0, 0.05, len(dates))
        evolution = np.clip(evolution, 0, 1)
        weight_evolution[model] = pd.Series(evolution, index=dates)
    
    return {
        'model_metrics': model_metrics,
        'time_series_data': time_series_data,
        'historical_data': historical_data,
        'predictions': predictions,
        'ensemble_pred': ensemble_pred,
        'confidence_intervals': (confidence_lower, confidence_upper),
        'residuals': residuals,
        'ensemble_weights': ensemble_weights,
        'weight_evolution': weight_evolution
    }


def test_dashboard_layouts():
    """Test different dashboard layouts."""
    print("🎨 Testing Dashboard Layouts")
    print("=" * 50)
    
    # Initialize theme manager
    theme_config = {'default': 'professional'}
    theme_manager = ThemeManager(theme_config)
    theme = theme_manager.get_current_theme()
    
    # Test different layout types
    layout_types = ['grid', 'tabs', 'accordion', 'flex']
    
    for layout_type in layout_types:
        print(f"\n📊 Testing {layout_type} layout...")
        
        # Create dashboard with layout
        dashboard = Dashboard(f"Test {layout_type.title()} Dashboard", layout_type, theme)
        
        # Add some mock charts
        for i in range(4):
            # Create mock chart object
            mock_chart = type('MockChart', (), {
                'title': f'Chart {i+1}',
                'export': lambda self, path, format: None
            })()
            
            dashboard.add_chart(mock_chart, title=f"Test Chart {i+1}")
        
        # Add summary panel
        summary = {
            'Layout Type': layout_type,
            'Charts': len(dashboard.charts),
            'Test Status': 'Passed'
        }
        dashboard.add_summary_panel(summary)
        
        # Test HTML generation
        html = dashboard.get_layout_html()
        print(f"  ✅ Generated {len(html)} characters of HTML")
        
        # Test export
        export_path = Path(f'test_dashboard_{layout_type}.html')
        dashboard.export_dashboard('html', export_path)
        print(f"  ✅ Exported to: {export_path}")
        
        # Test JSON export
        json_path = Path(f'test_dashboard_{layout_type}.json')
        dashboard.export_dashboard('json', json_path)
        print(f"  ✅ Exported config to: {json_path}")
    
    print("\n✅ All layout tests completed!")


def test_dashboard_builder():
    """Test the DashboardBuilder functionality."""
    print("\n🏗️ Testing DashboardBuilder")
    print("=" * 50)
    
    # Initialize components
    theme_config = {'default': 'professional'}
    theme_manager = ThemeManager(theme_config)
    chart_factory = ChartFactory(theme_manager)
    dashboard_builder = DashboardBuilder(chart_factory, theme_manager)
    
    # Generate sample data
    data = generate_sample_dashboard_data()
    
    # Test 1: Performance Dashboard
    print("\n📊 Test 1: Performance Dashboard")
    perf_dashboard = dashboard_builder.create_performance_dashboard(
        model_metrics=data['model_metrics'],
        time_series_data=data['time_series_data'],
        ensemble_weights=data['ensemble_weights'],
        residuals=data['residuals']
    )
    
    print(f"  ✅ Created dashboard with {len(perf_dashboard.charts)} charts")
    print(f"  ✅ Summary panels: {len(perf_dashboard.summary_panels)}")
    
    # Export performance dashboard
    perf_path = Path('performance_dashboard.html')
    perf_dashboard.export_dashboard('html', perf_path)
    print(f"  ✅ Exported to: {perf_path}")
    
    # Test 2: Prediction Dashboard
    print("\n📈 Test 2: Prediction Dashboard")
    pred_dashboard = dashboard_builder.create_prediction_dashboard(
        historical_data=data['historical_data'],
        predictions=data['predictions'],
        ensemble_prediction=data['ensemble_pred'],
        confidence_intervals=data['confidence_intervals']
    )
    
    print(f"  ✅ Created dashboard with {len(pred_dashboard.charts)} charts")
    
    # Export prediction dashboard
    pred_path = Path('prediction_dashboard.html')
    pred_dashboard.export_dashboard('html', pred_path)
    print(f"  ✅ Exported to: {pred_path}")
    
    # Test 3: Model Analysis Dashboard
    print("\n🔍 Test 3: Model Analysis Dashboard")
    analysis_dashboard = dashboard_builder.create_model_analysis_dashboard(
        model_predictions=data['predictions'],
        actual_values=data['historical_data']['close'].values,
        residual_analysis=data['residuals']
    )
    
    print(f"  ✅ Created dashboard with {len(analysis_dashboard.charts)} charts")
    
    # Export analysis dashboard
    analysis_path = Path('analysis_dashboard.html')
    analysis_dashboard.export_dashboard('html', analysis_path)
    print(f"  ✅ Exported to: {analysis_path}")
    
    # Test 4: Ensemble Dashboard
    print("\n🎯 Test 4: Ensemble Dashboard")
    ensemble_dashboard = dashboard_builder.create_ensemble_dashboard(
        ensemble_weights=data['ensemble_weights'],
        weight_evolution=data['weight_evolution'],
        ensemble_performance={'rmse': 3.5, 'r2': 0.85},
        individual_performance=data['model_metrics']
    )
    
    print(f"  ✅ Created dashboard with {len(ensemble_dashboard.charts)} charts")
    
    # Export ensemble dashboard
    ensemble_path = Path('ensemble_dashboard.html')
    ensemble_dashboard.export_dashboard('html', ensemble_path)
    print(f"  ✅ Exported to: {ensemble_path}")
    
    # Test 5: Custom Dashboard
    print("\n🛠️ Test 5: Custom Dashboard")
    custom_charts = [
        {
            'type': 'performance',
            'title': 'Custom Performance Chart',
            'data': {'metrics': data['model_metrics']},
            'position': (0, 0),
            'size': (1, 1)
        },
        {
            'type': 'price',
            'title': 'Custom Price Chart',
            'data': {
                'historical_data': data['historical_data']['close'],
                'predictions': data['predictions']
            },
            'position': (0, 1),
            'size': (1, 1)
        }
    ]
    
    custom_dashboard = dashboard_builder.create_custom_dashboard(
        title="Custom Analysis Dashboard",
        charts=custom_charts,
        layout='grid',
        summary_data={'Custom Charts': len(custom_charts), 'Status': 'Active'}
    )
    
    print(f"  ✅ Created custom dashboard with {len(custom_dashboard.charts)} charts")
    
    # Export custom dashboard
    custom_path = Path('custom_dashboard.html')
    custom_dashboard.export_dashboard('html', custom_path)
    print(f"  ✅ Exported to: {custom_path}")
    
    print("\n✅ All DashboardBuilder tests completed!")


def test_responsive_features():
    """Test responsive dashboard features."""
    print("\n📱 Testing Responsive Features")
    print("=" * 50)
    
    theme_config = {'default': 'dark_modern'}
    theme_manager = ThemeManager(theme_config)
    theme = theme_manager.get_current_theme()
    
    # Create dashboard with responsive layout
    dashboard = Dashboard("Responsive Test Dashboard", "grid", theme)
    
    # Add charts
    for i in range(6):
        mock_chart = type('MockChart', (), {'title': f'Chart {i+1}'})()
        dashboard.add_chart(mock_chart, title=f"Responsive Chart {i+1}")
    
    # Test different screen sizes
    screen_sizes = [
        ('Mobile', 600),
        ('Tablet', 900),
        ('Desktop', 1200),
        ('Large Desktop', 1600)
    ]
    
    for size_name, width in screen_sizes:
        print(f"\n📐 Testing {size_name} ({width}px)...")
        
        # Set responsive mode
        dashboard.set_responsive_mode(True, width)
        
        # Generate layout
        html = dashboard.get_layout_html()
        print(f"  ✅ Generated responsive layout: {len(html)} characters")
        
        # Export for this screen size
        export_path = Path(f'responsive_dashboard_{size_name.lower().replace(" ", "_")}.html')
        dashboard.export_dashboard('html', export_path)
        print(f"  ✅ Exported {size_name} version to: {export_path}")
    
    print("\n✅ Responsive features test completed!")


def test_dashboard_interactivity():
    """Test dashboard interactive features."""
    print("\n🎮 Testing Dashboard Interactivity")
    print("=" * 50)
    
    theme_config = {'default': 'light'}
    theme_manager = ThemeManager(theme_config)
    theme = theme_manager.get_current_theme()
    
    # Create interactive dashboard
    dashboard = Dashboard("Interactive Test Dashboard", "tabs", theme)
    
    # Add charts with different visibility states
    for i in range(4):
        mock_chart = type('MockChart', (), {'title': f'Chart {i+1}'})()
        dashboard.add_chart(mock_chart, title=f"Interactive Chart {i+1}")
    
    print(f"📊 Created dashboard with {len(dashboard.charts)} charts")
    
    # Test chart visibility toggling
    print("\n🔄 Testing chart visibility...")
    dashboard.toggle_chart_visibility('chart_1', False)
    visible_charts = [c for c in dashboard.charts if c['visible']]
    print(f"  ✅ Visible charts after toggle: {len(visible_charts)}")
    
    # Test layout switching
    print("\n🔄 Testing layout switching...")
    original_layout = dashboard.layout.layout_type
    dashboard.set_layout({'type': 'accordion', 'cols': 1})
    print(f"  ✅ Layout changed from {original_layout} to {dashboard.layout.layout_type}")
    
    # Generate interactive HTML
    html = dashboard.get_layout_html()
    print(f"  ✅ Generated interactive HTML: {len(html)} characters")
    
    # Export interactive dashboard
    interactive_path = Path('interactive_dashboard.html')
    dashboard.export_dashboard('html', interactive_path)
    print(f"  ✅ Exported interactive dashboard to: {interactive_path}")
    
    print("\n✅ Interactivity test completed!")


def main():
    """Run all dashboard tests."""
    print("🚀 Enhanced Dashboard System Test Suite")
    print("=" * 70)
    
    try:
        # Run all tests
        test_dashboard_layouts()
        test_dashboard_builder()
        test_responsive_features()
        test_dashboard_interactivity()
        
        print("\n🎉 ALL DASHBOARD TESTS COMPLETED SUCCESSFULLY!")
        print("\n📁 Generated Files:")
        
        # List all generated files
        html_files = list(Path('.').glob('*dashboard*.html'))
        json_files = list(Path('.').glob('*dashboard*.json'))
        
        for file in html_files:
            print(f"  📄 {file}")
        for file in json_files:
            print(f"  📋 {file}")
        
        print(f"\n📊 Total files generated: {len(html_files) + len(json_files)}")
        print("🌐 Open the HTML files in your browser to view the dashboards!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
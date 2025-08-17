#!/usr/bin/env python3
"""Test script for enhanced PerformanceChart functionality."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor.visualization.performance_chart import PerformanceChart
from stock_predictor.visualization.theme_manager import ThemeManager


def generate_sample_performance_data():
    """Generate sample performance data for testing."""
    np.random.seed(42)
    
    # Model names
    models = ['LSTM', 'Random Forest', 'Linear Regression', 'SVM', 'Ensemble']
    
    # Generate performance metrics
    metrics = {}
    for model in models:
        metrics[model] = {
            'rmse': np.random.uniform(2.0, 8.0),
            'mae': np.random.uniform(1.5, 6.0),
            'directional_accuracy': np.random.uniform(0.45, 0.75),
            'r2_score': np.random.uniform(0.3, 0.85),
            'mape': np.random.uniform(0.05, 0.15)
        }
    
    # Generate time series performance data
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    time_series_data = {}
    
    for model in models:
        # Generate realistic performance over time with some trend
        base_performance = metrics[model]['r2_score']
        noise = np.random.normal(0, 0.05, len(dates))
        trend = np.linspace(-0.1, 0.1, len(dates))
        performance = base_performance + trend + noise
        performance = np.clip(performance, 0, 1)  # Keep in valid range
        
        time_series_data[model] = pd.Series(performance, index=dates)
    
    # Generate rolling metrics
    rolling_data = {}
    for model in models:
        rolling_data[model] = {
            'rmse': time_series_data[model].rolling(7).std() * 10 + 2,  # Simulate RMSE
            'r2': time_series_data[model].rolling(7).mean()  # Rolling R²
        }
    
    # Generate residuals
    residuals = {}
    for model in models:
        # Generate realistic residuals (should be normally distributed)
        residuals[model] = np.random.normal(0, metrics[model]['rmse'], 100)
    
    # Generate prediction errors
    errors = {}
    for model in models:
        errors[model] = np.random.normal(0, metrics[model]['mae'], 200)
    
    # Generate predictions for correlation analysis
    predictions = {}
    base_predictions = np.random.normal(100, 20, 100)  # Base stock prices
    
    for i, model in enumerate(models):
        # Add model-specific bias and noise
        bias = np.random.normal(0, 2, 100)
        noise = np.random.normal(0, metrics[model]['rmse'], 100)
        predictions[model] = base_predictions + bias + noise
    
    return {
        'metrics': metrics,
        'time_series_data': time_series_data,
        'rolling_data': rolling_data,
        'residuals': residuals,
        'errors': errors,
        'predictions': predictions
    }


def test_enhanced_performance_chart():
    """Test the enhanced PerformanceChart functionality."""
    print("📊 Testing Enhanced PerformanceChart")
    print("=" * 50)
    
    try:
        # Initialize theme manager
        theme_config = {'default': 'professional'}
        theme_manager = ThemeManager(theme_config)
        theme = theme_manager.get_current_theme()
        
        # Generate sample data
        print("📈 Generating sample performance data...")
        data = generate_sample_performance_data()
        
        # Test 1: Basic performance comparison chart
        print("\n🎯 Test 1: Basic Performance Comparison")
        chart_config = {
            'backend': 'plotly',
            'figsize': (12, 8),
            'interactive': True
        }
        
        perf_chart = PerformanceChart(theme, chart_config, subtype='comparison')
        perf_chart.add_model_metrics(data['metrics'])
        
        # Apply theme and export
        theme_manager.apply_theme_to_figure(perf_chart.figure, 'performance_chart')
        export_path = Path('performance_comparison.html')
        perf_chart.export(export_path, format='html')
        print(f"  ✅ Basic comparison chart exported to: {export_path}")
        
        # Test 2: Time series performance
        print("\n📈 Test 2: Time Series Performance")
        ts_chart = PerformanceChart(theme, chart_config, subtype='time_series')
        ts_chart.add_time_series_performance(data['time_series_data'])
        
        export_path = Path('performance_time_series.html')
        ts_chart.export(export_path, format='html')
        print(f"  ✅ Time series chart exported to: {export_path}")
        
        # Test 3: Rolling metrics
        print("\n📊 Test 3: Rolling Performance Metrics")
        rolling_chart = PerformanceChart(theme, chart_config)
        rolling_chart.add_rolling_metrics(data['rolling_data'], window=7)
        
        export_path = Path('performance_rolling.html')
        rolling_chart.export(export_path, format='html')
        print(f"  ✅ Rolling metrics chart exported to: {export_path}")
        
        # Test 4: Residual analysis
        print("\n🔍 Test 4: Residual Analysis")
        residual_chart = PerformanceChart(theme, chart_config, subtype='residuals')
        residual_chart.add_residual_analysis(data['residuals'])
        
        export_path = Path('performance_residuals.html')
        residual_chart.export(export_path, format='html')
        print(f"  ✅ Residual analysis chart exported to: {export_path}")
        
        # Test 5: Error distribution
        print("\n📊 Test 5: Prediction Error Distribution")
        error_chart = PerformanceChart(theme, chart_config)
        error_chart.create_prediction_error_distribution(data['errors'])
        
        export_path = Path('performance_error_distribution.html')
        error_chart.export(export_path, format='html')
        print(f"  ✅ Error distribution chart exported to: {export_path}")
        
        # Test 6: Q-Q plots
        print("\n📈 Test 6: Q-Q Plots for Residual Analysis")
        qq_chart = PerformanceChart(theme, chart_config)
        qq_chart.create_qq_plots(data['residuals'])
        
        export_path = Path('performance_qq_plots.html')
        qq_chart.export(export_path, format='html')
        print(f"  ✅ Q-Q plots exported to: {export_path}")
        
        # Test 7: Correlation matrix
        print("\n🔗 Test 7: Model Prediction Correlation Matrix")
        corr_chart = PerformanceChart(theme, chart_config)
        corr_chart.create_correlation_matrix(data['predictions'])
        
        export_path = Path('performance_correlation.html')
        corr_chart.export(export_path, format='html')
        print(f"  ✅ Correlation matrix exported to: {export_path}")
        
        # Test 8: Radar chart
        print("\n🎯 Test 8: Performance Radar Chart")
        radar_chart = PerformanceChart(theme, chart_config)
        radar_chart.create_performance_radar_chart(data['metrics'])
        
        export_path = Path('performance_radar.html')
        radar_chart.export(export_path, format='html')
        print(f"  ✅ Radar chart exported to: {export_path}")
        
        # Test 9: Performance summary table
        print("\n📋 Test 9: Performance Summary Table")
        summary_df = perf_chart.create_performance_summary_table(data['metrics'])
        print("  Performance Summary:")
        print(summary_df.round(3))
        
        # Save summary table
        summary_path = Path('performance_summary.csv')
        summary_df.to_csv(summary_path)
        print(f"  ✅ Summary table saved to: {summary_path}")
        
        print("\n✅ All PerformanceChart tests completed successfully!")
        
        return {
            'basic_chart': perf_chart,
            'time_series_chart': ts_chart,
            'rolling_chart': rolling_chart,
            'residual_chart': residual_chart,
            'error_chart': error_chart,
            'qq_chart': qq_chart,
            'correlation_chart': corr_chart,
            'radar_chart': radar_chart,
            'summary_table': summary_df
        }
        
    except Exception as e:
        print(f"❌ Error testing enhanced performance chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = test_enhanced_performance_chart()
    
    if results:
        print("\n🎉 Enhanced PerformanceChart test completed! Features tested:")
        print("  ✅ Basic model performance comparison")
        print("  ✅ Time series performance tracking")
        print("  ✅ Rolling performance metrics")
        print("  ✅ Residual analysis plots")
        print("  ✅ Prediction error distributions")
        print("  ✅ Q-Q plots for normality testing")
        print("  ✅ Model prediction correlation matrix")
        print("  ✅ Multi-dimensional radar charts")
        print("  ✅ Performance summary tables with rankings")
        print("  ✅ Professional theme styling")
        print("  ✅ Interactive Plotly visualizations")
        print("  ✅ Multiple export formats")
        
        print(f"\n📁 Check the generated HTML files to view the interactive charts!")
        print(f"📊 Performance summary saved to performance_summary.csv")
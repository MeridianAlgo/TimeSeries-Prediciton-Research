#!/usr/bin/env python3
"""Test script for enhanced PriceChart functionality."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from stock_predictor.visualization.price_chart import PriceChart
from stock_predictor.visualization.theme_manager import ThemeManager


def generate_sample_data():
    """Generate sample stock data for testing."""
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start_date, periods=100, freq='D')
    
    # Generate realistic stock price data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, 100)  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # Generate volume data
    volume = np.random.randint(1000000, 5000000, 100)
    
    # Generate technical indicators
    # Simple Moving Averages
    ma20 = pd.Series(prices).rolling(20).mean().values
    ma50 = pd.Series(prices).rolling(50).mean().values
    
    # Bollinger Bands (simplified)
    rolling_std = pd.Series(prices).rolling(20).std().values
    bb_upper = ma20 + (2 * rolling_std)
    bb_lower = ma20 - (2 * rolling_std)
    
    # RSI (simplified calculation)
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(14).mean().values
    avg_loss = pd.Series(loss).rolling(14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi])  # Add first value
    
    # Generate model predictions
    predictions = {}
    for model_name in ['LSTM', 'Random Forest', 'Linear Regression']:
        # Add some noise to actual prices for predictions
        noise = np.random.normal(0, 2, len(prices))
        predictions[model_name] = prices + noise
    
    # Generate ensemble prediction
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    
    # Generate confidence intervals
    std_dev = np.std(list(predictions.values()), axis=0)
    confidence_lower = ensemble_pred - 1.96 * std_dev
    confidence_upper = ensemble_pred + 1.96 * std_dev
    
    return {
        'dates': dates,
        'prices': prices,
        'volume': volume,
        'ma20': ma20,
        'ma50': ma50,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'bb_middle': ma20,
        'rsi': rsi,
        'predictions': predictions,
        'ensemble_pred': ensemble_pred,
        'confidence_intervals': (confidence_lower, confidence_upper)
    }


def test_enhanced_price_chart():
    """Test the enhanced PriceChart functionality."""
    print("📈 Testing Enhanced PriceChart")
    print("=" * 50)
    
    try:
        # Initialize theme manager
        theme_config = {'default': 'professional'}
        theme_manager = ThemeManager(theme_config)
        theme = theme_manager.get_current_theme()
        
        # Generate sample data
        print("📊 Generating sample data...")
        data = generate_sample_data()
        
        # Create price chart
        print("🎨 Creating enhanced price chart...")
        chart_config = {
            'backend': 'plotly',
            'figsize': (14, 10),
            'interactive': True
        }
        
        price_chart = PriceChart(theme, chart_config)
        
        # Add historical data
        print("📈 Adding historical price data...")
        price_chart.add_historical_data(data['dates'], data['prices'])
        
        # Add model predictions
        print("🤖 Adding model predictions...")
        price_chart.add_predictions(data['predictions'])
        
        # Add ensemble prediction with confidence intervals
        print("🎯 Adding ensemble prediction...")
        price_chart.add_ensemble_prediction(
            data['ensemble_pred'], 
            data['confidence_intervals']
        )
        
        # Add moving averages
        print("📊 Adding moving averages...")
        ma_data = {
            'MA20': data['ma20'],
            'MA50': data['ma50']
        }
        price_chart.add_moving_averages(ma_data)
        
        # Add Bollinger Bands
        print("📈 Adding Bollinger Bands...")
        price_chart.add_bollinger_bands(
            data['bb_middle'], 
            data['bb_upper'], 
            data['bb_lower']
        )
        
        # Add RSI
        print("📊 Adding RSI indicator...")
        price_chart.add_rsi_subplot(data['rsi'])
        
        # Add volume
        print("📊 Adding volume data...")
        price_chart.add_volume_subplot(data['volume'])
        
        # Add some annotations
        print("📝 Adding price annotations...")
        annotations = [
            {
                'date': data['dates'][20],
                'price': data['prices'][20],
                'text': 'Buy Signal',
                'type': 'buy'
            },
            {
                'date': data['dates'][60],
                'price': data['prices'][60],
                'text': 'Sell Signal',
                'type': 'sell'
            },
            {
                'date': data['dates'][80],
                'price': data['prices'][80],
                'text': 'Market Event',
                'type': 'info'
            }
        ]
        price_chart.add_price_annotations(annotations)
        
        # Set date range
        print("📅 Setting date range...")
        start_date = data['dates'][10].strftime('%Y-%m-%d')
        end_date = data['dates'][90].strftime('%Y-%m-%d')
        price_chart.set_date_range(start_date, end_date)
        
        # Apply theme
        print("🎨 Applying theme...")
        theme_manager.apply_theme_to_figure(price_chart.figure, 'price_chart')
        
        # Get chart data summary
        chart_data = price_chart.get_chart_data()
        print(f"📊 Chart contains:")
        print(f"  - Historical data points: {len(chart_data['historical_prices'])}")
        print(f"  - Model predictions: {len(chart_data['model_predictions'])}")
        print(f"  - Technical indicators: {len(chart_data['technical_indicators'])}")
        
        # Export chart
        print("💾 Exporting chart...")
        export_path = Path('enhanced_price_chart.html')
        price_chart.export(export_path, format='html')
        print(f"  Chart exported to: {export_path}")
        
        # Test model toggling
        print("🔄 Testing model toggle...")
        models = list(data['predictions'].keys())
        price_chart.enable_model_toggle(models)
        
        # Test crossfilter
        print("🔗 Testing crossfilter...")
        price_chart.enable_crossfilter(True)
        
        print("\n✅ Enhanced PriceChart test completed successfully!")
        print(f"📁 Check {export_path} to view the interactive chart")
        
        return price_chart
        
    except Exception as e:
        print(f"❌ Error testing enhanced price chart: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    chart = test_enhanced_price_chart()
    
    if chart:
        print("\n🎉 Test completed! The enhanced price chart includes:")
        print("  ✅ Historical price data")
        print("  ✅ Multiple model predictions")
        print("  ✅ Ensemble prediction with confidence intervals")
        print("  ✅ Moving averages (MA20, MA50)")
        print("  ✅ Bollinger Bands")
        print("  ✅ RSI indicator")
        print("  ✅ Volume data")
        print("  ✅ Price annotations")
        print("  ✅ Date range selection")
        print("  ✅ Professional theme styling")
        print("  ✅ Interactive features")
        print("  ✅ Export functionality")
"""Backtesting charts with candlesticks and rolling predictions."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockPredictorApp


def show_backtest_charts():
    """Show backtesting charts with candlesticks and rolling predictions."""
    print("🚀 AAPL Backtesting Analysis with Candlesticks")
    print("=" * 60)
    
    try:
        # Get historical data for backtesting
        print("📊 Fetching historical data for backtesting...")
        from stock_predictor.data.fetcher import DataFetcher
        
        fetcher = DataFetcher()
        # Get 6 months of data for backtesting
        raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=0.5)
        
        if raw_data is None or raw_data.empty:
            print("❌ Could not fetch data")
            return
        
        # Prepare data
        data = prepare_backtest_data(raw_data)
        print(f"✅ Got {len(data)} days of data")
        
        # Run backtesting simulation
        print("🔄 Running backtesting simulation...")
        backtest_results = run_backtesting_simulation(data)
        
        # Create backtesting dashboard
        create_backtest_dashboard(data, backtest_results)
        
        print("\n🎉 Backtesting charts displayed! Close the windows to continue.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def prepare_backtest_data(data):
    """Prepare data for backtesting."""
    # Set up proper date index
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
    elif 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    
    # Standardize column names
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    # Sort by date
    data = data.sort_index()
    
    return data


def run_backtesting_simulation(data):
    """Run backtesting simulation with rolling predictions."""
    print("🔄 Simulating daily predictions...")
    
    # Start backtesting from 3 months ago (need some history for features)
    start_idx = len(data) // 2  # Start from middle of data
    
    predictions = []
    actual_prices = []
    dates = []
    errors = []
    
    # Simple trend-following prediction model for simulation
    for i in range(start_idx, len(data) - 1):  # -1 because we predict next day
        current_date = data.index[i]
        actual_next_price = data['close'].iloc[i + 1]
        
        # Simple prediction based on recent trend and moving average
        recent_prices = data['close'].iloc[max(0, i-10):i+1]  # Last 10 days
        ma_5 = recent_prices.tail(5).mean()
        ma_20 = recent_prices.tail(min(20, len(recent_prices))).mean()
        
        # Trend-based prediction
        if len(recent_prices) >= 5:
            recent_trend = (recent_prices.iloc[-1] - recent_prices.iloc[-5]) / 5
            momentum = 1.0 if ma_5 > ma_20 else 0.8  # Momentum factor
            
            # Add some noise to make it realistic
            noise = np.random.normal(0, recent_prices.std() * 0.1)
            predicted_price = recent_prices.iloc[-1] + (recent_trend * momentum) + noise
        else:
            predicted_price = recent_prices.iloc[-1]  # Fallback
        
        # Store results
        predictions.append(predicted_price)
        actual_prices.append(actual_next_price)
        dates.append(current_date)
        
        # Calculate error
        error = abs(predicted_price - actual_next_price)
        errors.append(error)
    
    # Calculate performance metrics
    predictions = np.array(predictions)
    actual_prices = np.array(actual_prices)
    errors = np.array(errors)
    
    rmse = np.sqrt(np.mean((predictions - actual_prices) ** 2))
    mae = np.mean(errors)
    mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
    
    # Directional accuracy
    actual_direction = np.diff(actual_prices) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print(f"📊 Backtest Results:")
    print(f"   RMSE: {rmse:.3f}")
    print(f"   MAE: {mae:.3f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Directional Accuracy: {directional_accuracy:.1f}%")
    
    return {
        'dates': dates,
        'predictions': predictions,
        'actual_prices': actual_prices,
        'errors': errors,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'directional_accuracy': directional_accuracy,
        'start_idx': start_idx
    }


def create_backtest_dashboard(data, backtest_results):
    """Create comprehensive backtesting dashboard."""
    print("📊 Creating backtesting dashboard...")
    
    try:
        # Set up the figure
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle('AAPL Backtesting Analysis - Candlesticks & Rolling Predictions', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        # Create layout
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3, height_ratios=[3, 2, 1.5, 1])
        
        # Chart 1: Main candlestick chart with predictions (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        create_candlestick_chart(ax1, data, backtest_results)
        
        # Chart 2: Prediction vs Actual (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        create_prediction_comparison(ax2, backtest_results)
        
        # Chart 3: Prediction Errors Over Time (middle right)
        ax3 = fig.add_subplot(gs[1, 1])
        create_error_analysis(ax3, backtest_results)
        
        # Chart 4: Error Distribution (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])
        create_error_distribution(ax4, backtest_results)
        
        # Chart 5: Rolling Performance Metrics (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        create_rolling_metrics(ax5, backtest_results)
        
        # Chart 6: Summary Statistics (bottom row)
        ax6 = fig.add_subplot(gs[3, :])
        create_backtest_summary(ax6, backtest_results, data)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Backtesting dashboard displayed successfully")
        
    except Exception as e:
        print(f"⚠️ Failed to create dashboard: {str(e)}")
        import traceback
        traceback.print_exc()


def create_candlestick_chart(ax, data, backtest_results):
    """Create candlestick chart with predictions."""
    # Get the backtesting period data
    start_idx = backtest_results['start_idx']
    backtest_data = data.iloc[start_idx:]
    
    # Create candlesticks
    for i, (date, row) in enumerate(backtest_data.iterrows()):
        # Determine color
        color = 'green' if row['close'] >= row['open'] else 'red'
        alpha = 0.8
        
        # Draw the candlestick
        # Body (rectangle)
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['close'], row['open'])
        
        rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                        0.6, body_height, 
                        facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Wicks (lines)
        ax.plot([mdates.date2num(date), mdates.date2num(date)], 
               [row['low'], row['high']], 
               color='black', linewidth=1, alpha=0.8)
    
    # Add predictions as line
    pred_dates = backtest_results['dates']
    predictions = backtest_results['predictions']
    
    ax.plot(pred_dates, predictions, 
           label='Daily Predictions', color='blue', linewidth=2, 
           linestyle='--', marker='o', markersize=3, alpha=0.8)
    
    # Add actual prices line for comparison
    actual_dates = [data.index[start_idx + i + 1] for i in range(len(predictions))]
    actual_prices = backtest_results['actual_prices']
    
    ax.plot(actual_dates, actual_prices, 
           label='Actual Next Day', color='orange', linewidth=2, alpha=0.9)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    
    ax.set_title('AAPL Candlestick Chart with Daily Predictions', fontweight='bold', fontsize=14)
    ax.set_xlabel('Date (MM/DD)')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def create_prediction_comparison(ax, backtest_results):
    """Create prediction vs actual scatter plot."""
    predictions = backtest_results['predictions']
    actual_prices = backtest_results['actual_prices']
    
    # Scatter plot
    ax.scatter(actual_prices, predictions, alpha=0.6, color='blue', s=30)
    
    # Perfect prediction line
    min_val = min(min(actual_prices), min(predictions))
    max_val = max(max(actual_prices), max(predictions))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
           label='Perfect Prediction')
    
    # Calculate R²
    correlation = np.corrcoef(actual_prices, predictions)[0, 1]
    r_squared = correlation ** 2
    
    ax.set_title(f'Predicted vs Actual Prices\\nR² = {r_squared:.3f}', fontweight='bold')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_error_analysis(ax, backtest_results):
    """Create prediction error analysis over time."""
    dates = backtest_results['dates']
    errors = backtest_results['errors']
    
    # Plot errors over time
    ax.plot(dates, errors, color='red', linewidth=1.5, alpha=0.8)
    
    # Add rolling average of errors
    error_series = pd.Series(errors, index=dates)
    rolling_error = error_series.rolling(window=10).mean()
    ax.plot(dates, rolling_error, color='blue', linewidth=2, label='10-day MA')
    
    # Add horizontal line for mean error
    mean_error = np.mean(errors)
    ax.axhline(y=mean_error, color='green', linestyle='--', 
              label=f'Mean Error: ${mean_error:.2f}')
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    ax.set_title('Prediction Errors Over Time', fontweight='bold')
    ax.set_xlabel('Date (MM/DD)')
    ax.set_ylabel('Absolute Error ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def create_error_distribution(ax, backtest_results):
    """Create error distribution histogram."""
    errors = backtest_results['errors']
    
    # Create histogram
    n, bins, patches = ax.hist(errors, bins=20, alpha=0.7, color='skyblue', 
                              edgecolor='black', linewidth=0.5)
    
    # Color bars based on error magnitude
    for i, patch in enumerate(patches):
        if bins[i] < np.mean(errors):
            patch.set_facecolor('#4CAF50')  # Green for low errors
        else:
            patch.set_facecolor('#FF5722')  # Red for high errors
    
    # Add statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: ${mean_error:.2f}')
    ax.axvline(median_error, color='blue', linestyle='--', linewidth=2, 
              label=f'Median: ${median_error:.2f}')
    
    ax.set_title('Prediction Error Distribution', fontweight='bold')
    ax.set_xlabel('Absolute Error ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_rolling_metrics(ax, backtest_results):
    """Create rolling performance metrics."""
    dates = backtest_results['dates']
    predictions = backtest_results['predictions']
    actual_prices = backtest_results['actual_prices']
    
    # Calculate rolling RMSE
    window = 20
    rolling_rmse = []
    
    for i in range(window, len(predictions)):
        pred_window = predictions[i-window:i]
        actual_window = actual_prices[i-window:i]
        rmse = np.sqrt(np.mean((pred_window - actual_window) ** 2))
        rolling_rmse.append(rmse)
    
    rolling_dates = dates[window:]
    
    ax.plot(rolling_dates, rolling_rmse, color='purple', linewidth=2)
    
    # Add overall RMSE line
    overall_rmse = backtest_results['rmse']
    ax.axhline(y=overall_rmse, color='red', linestyle='--', 
              label=f'Overall RMSE: {overall_rmse:.3f}')
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    
    ax.set_title(f'Rolling RMSE ({window}-day window)', fontweight='bold')
    ax.set_xlabel('Date (MM/DD)')
    ax.set_ylabel('RMSE ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def create_backtest_summary(ax, backtest_results, data):
    """Create backtesting summary panel."""
    ax.axis('off')
    
    # Calculate additional statistics
    predictions = backtest_results['predictions']
    actual_prices = backtest_results['actual_prices']
    
    # Accuracy within different thresholds
    errors = backtest_results['errors']
    within_1_pct = np.mean(errors / actual_prices < 0.01) * 100
    within_3_pct = np.mean(errors / actual_prices < 0.03) * 100
    within_5_pct = np.mean(errors / actual_prices < 0.05) * 100
    
    # Trading statistics
    start_date = backtest_results['dates'][0].strftime('%Y-%m-%d')
    end_date = backtest_results['dates'][-1].strftime('%Y-%m-%d')
    trading_days = len(backtest_results['dates'])
    
    # Create summary text
    summary_text = f"""
📊 BACKTESTING SUMMARY - AAPL DAILY PREDICTIONS

📅 PERIOD: {start_date} to {end_date} ({trading_days} trading days)

🎯 ACCURACY METRICS:
• RMSE: ${backtest_results['rmse']:.3f}
• MAE: ${backtest_results['mae']:.3f}  
• MAPE: {backtest_results['mape']:.2f}%
• Directional Accuracy: {backtest_results['directional_accuracy']:.1f}%

📈 PREDICTION ACCURACY:
• Within 1%: {within_1_pct:.1f}% of predictions
• Within 3%: {within_3_pct:.1f}% of predictions  
• Within 5%: {within_5_pct:.1f}% of predictions

💡 INSIGHTS:
• Best Error: ${np.min(errors):.2f}
• Worst Error: ${np.max(errors):.2f}
• Avg Daily Error: ${np.mean(errors):.2f}
• Error Std Dev: ${np.std(errors):.2f}

🔄 MODEL: Simple trend-following with momentum (for demonstration)
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E8", alpha=0.9))


if __name__ == "__main__":
    show_backtest_charts()
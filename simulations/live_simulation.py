"""Live moving simulation of stock predictions."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockPredictorApp


def run_live_simulation():
    """Run a live simulation that actually moves and updates."""
    print("🎬 Starting LIVE Stock Prediction Simulation")
    print("=" * 60)
    
    try:
        # Load data
        print("📊 Loading historical data...")
        data = load_simulation_data()
        
        # Find starting point (around May 13th, 2024)
        start_idx = find_start_date(data)
        print(f"🎯 Starting from: {data.index[start_idx].strftime('%Y-%m-%d')}")
        
        # Set up the plot
        plt.ion()  # Turn on interactive mode
        fig, axes = setup_live_plot()
        
        # Run the simulation
        run_weekly_simulation(data, start_idx, fig, axes)
        
        plt.ioff()  # Turn off interactive mode
        
        # Only show if there are still figures open
        if plt.get_fignums():
            plt.show()
        else:
            print("🛑 All windows closed - simulation ended")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def load_simulation_data():
    """Load data for simulation."""
    from stock_predictor.data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=0.75)  # 9 months
    
    if raw_data is None or raw_data.empty:
        raise Exception("Could not fetch data")
    
    # Clean up data
    if 'date' in raw_data.columns:
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        raw_data = raw_data.set_index('date')
    
    # Standardize columns
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in raw_data.columns:
            raw_data = raw_data.rename(columns={old_name: new_name})
    
    return raw_data.sort_index()


def find_start_date(data):
    """Find starting index around May 13th."""
    # Look for a date around May 2024
    target_month = 5  # May
    target_year = 2024
    
    for i, date in enumerate(data.index):
        if hasattr(date, 'month') and hasattr(date, 'year'):
            if date.year == target_year and date.month == target_month and date.day >= 10:
                return i
    
    # Fallback: use 1/3 into the data
    return len(data) // 3


def setup_live_plot():
    """Set up the live plotting environment."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('🎬 LIVE AAPL Stock Prediction Simulation', fontsize=18, fontweight='bold')
    
    # Create subplots
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, height_ratios=[2, 1])
    
    ax_main = fig.add_subplot(gs[0, :])  # Main chart
    ax_error = fig.add_subplot(gs[1, 0])  # Error tracking
    ax_stats = fig.add_subplot(gs[1, 1])  # Statistics
    
    return fig, (ax_main, ax_error, ax_stats)


def run_weekly_simulation(data, start_idx, fig, axes):
    """Run the actual simulation week by week."""
    ax_main, ax_error, ax_stats = axes
    
    # Simulation state
    current_idx = start_idx
    week_count = 0
    all_errors = []
    all_predictions = []
    all_actuals = []
    
    print("🎬 Starting simulation... (Close window or Press Ctrl+C to stop)")
    print("📺 Watch the chart update in real-time!")
    
    # Add window close event handler
    def on_close(event):
        print("\\n🛑 Window closed - stopping simulation")
        plt.close('all')
        return
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    try:
        while plt.get_fignums():  # Continue while window is open
            week_count += 1
            current_date = data.index[current_idx]
            current_price = data['close'].iloc[current_idx]
            
            print(f"📅 Week {week_count}: {current_date.strftime('%Y-%m-%d')} - ${current_price:.2f}")
            
            # Clear previous plots
            ax_main.clear()
            ax_error.clear()
            ax_stats.clear()
            
            # Get historical data up to current point
            historical_data = data.iloc[:current_idx+1]
            
            # Make prediction for next week
            next_week_predictions = make_weekly_prediction(historical_data)
            
            # Get actual next week data
            next_week_actual = data.iloc[current_idx+1:current_idx+8]
            
            # Plot main chart
            plot_main_chart(ax_main, historical_data, next_week_predictions, 
                          next_week_actual, current_date, week_count)
            
            # Calculate and store errors
            if len(next_week_actual) > 0:
                actual_prices = next_week_actual['close'].values
                pred_prices = next_week_predictions[:len(actual_prices)]
                
                if len(pred_prices) > 0:
                    week_error = np.sqrt(np.mean((np.array(pred_prices) - actual_prices) ** 2))
                    all_errors.append(week_error)
                    all_predictions.extend(pred_prices)
                    all_actuals.extend(actual_prices)
            
            # Plot error tracking
            if all_errors:
                plot_error_tracking(ax_error, all_errors, week_count)
            
            # Plot statistics
            plot_statistics(ax_stats, all_predictions, all_actuals, week_count)
            
            # Update display
            plt.draw()
            plt.pause(1.5)  # Pause for 1.5 seconds to see the update
            
            # Check if window is still open
            if not plt.get_fignums():
                print("\\n🛑 Window closed - stopping simulation")
                break
            
            # Move to next week
            current_idx += 7
            
            # Reset and repeat simulation when we reach the end
            if current_idx >= len(data) - 14 or week_count >= 25:
                print("\\n🔄 SIMULATION CYCLE COMPLETE!")
                if all_errors:
                    final_avg_error = np.mean(all_errors)
                    final_accuracy = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)) / np.array(all_actuals) < 0.03) * 100 if len(all_predictions) > 0 else 0
                    print(f"📊 Cycle Results - Avg RMSE: ${final_avg_error:.2f}, Accuracy: {final_accuracy:.1f}%")
                
                print("🔄 Restarting simulation in 3 seconds...")
                
                # Show restart message on chart
                ax_main.text(0.5, 0.5, '🔄 RESTARTING SIMULATION...', 
                           transform=ax_main.transAxes, fontsize=20, fontweight='bold',
                           ha='center', va='center', color='blue',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
                plt.draw()
                plt.pause(3)  # 3 second pause before restart
                
                # Check if window is still open after restart pause
                if not plt.get_fignums():
                    print("\\n🛑 Window closed during restart - stopping simulation")
                    break
                
                current_idx = start_idx
                week_count = 0
                all_errors = []
                all_predictions = []
                all_actuals = []
                continue
                
    except KeyboardInterrupt:
        print("\\n⏹️ Simulation stopped by user (Ctrl+C)")
    except Exception as e:
        if "window" in str(e).lower() or "figure" in str(e).lower():
            print("\\n🛑 Window closed - simulation stopped")
        else:
            print(f"\\n❌ Simulation error: {str(e)}")
    
    # Clean up
    plt.close('all')
    print(f"\\n🎉 Simulation completed! Ran for {week_count} weeks")
    
    # Final summary
    if all_errors:
        avg_error = np.mean(all_errors)
        print(f"📊 Average weekly RMSE: ${avg_error:.2f}")
        
        if len(all_predictions) > 0:
            accuracy = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)) / np.array(all_actuals) < 0.03) * 100
            print(f"🎯 Overall accuracy (within 3%): {accuracy:.1f}%")


def make_weekly_prediction(historical_data):
    """Make an improved prediction for the next week with higher accuracy."""
    recent_prices = historical_data['close'].tail(30)  # Use more data
    recent_volume = historical_data['volume'].tail(30)
    
    if len(recent_prices) < 10:
        # Not enough data, return flat prediction
        return [recent_prices.iloc[-1]] * 7
    
    # Multiple timeframe analysis for better accuracy
    ma_3 = recent_prices.tail(3).mean()
    ma_5 = recent_prices.tail(5).mean()
    ma_10 = recent_prices.tail(10).mean()
    ma_20 = recent_prices.tail(min(20, len(recent_prices))).mean()
    
    # Multi-timeframe trend analysis
    short_trend = (ma_3 - ma_5) / ma_5 if ma_5 != 0 else 0
    medium_trend = (ma_5 - ma_10) / ma_10 if ma_10 != 0 else 0
    long_trend = (ma_10 - ma_20) / ma_20 if ma_20 != 0 else 0
    
    # Weighted trend (recent trends matter more)
    combined_trend = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
    
    # Momentum analysis
    price_momentum = (recent_prices.iloc[-1] - recent_prices.iloc[-5]) / recent_prices.iloc[-5]
    
    # Volume analysis for confirmation
    volume_ma = recent_volume.tail(10).mean()
    volume_trend = 1.0 if recent_volume.iloc[-1] > volume_ma else 0.8
    
    # Reduced volatility for more accurate predictions
    volatility = recent_prices.pct_change().std() * 0.3  # Reduce noise significantly
    
    # Support and resistance levels
    recent_high = recent_prices.tail(10).max()
    recent_low = recent_prices.tail(10).min()
    current_price = recent_prices.iloc[-1]
    
    # Generate more accurate predictions
    predictions = []
    base_price = current_price
    
    for day in range(7):
        # Combine multiple signals for better accuracy
        trend_effect = combined_trend * base_price * 0.02 * volume_trend  # Reduced trend impact
        momentum_effect = price_momentum * base_price * 0.01 * (1 - day * 0.05)  # Momentum fades
        
        # Mean reversion component (prices tend to revert to moving average)
        reversion_target = ma_10
        reversion_effect = (reversion_target - base_price) * 0.05
        
        # Support/resistance consideration
        if base_price > recent_high * 0.98:  # Near resistance
            resistance_effect = -base_price * 0.005
        elif base_price < recent_low * 1.02:  # Near support
            resistance_effect = base_price * 0.005
        else:
            resistance_effect = 0
        
        # Much reduced noise for higher accuracy
        noise = np.random.normal(0, volatility * base_price * 0.1)  # Reduced from 0.2 to 0.1
        
        # Combine all effects
        total_effect = trend_effect + momentum_effect + reversion_effect + resistance_effect + noise
        
        # Predict next price with bounds
        next_price = base_price + total_effect
        
        # Apply reasonable bounds (prevent extreme moves)
        max_change = base_price * 0.05  # Max 5% daily change
        next_price = max(min(next_price, base_price + max_change), base_price - max_change)
        
        predictions.append(next_price)
        
        # Use prediction as base for next day with some smoothing
        base_price = next_price * 0.7 + base_price * 0.3  # Smooth transition
    
    return predictions


def plot_main_chart(ax, historical_data, predictions, actual_data, current_date, week_count):
    """Plot the main chart with candlesticks and predictions overlaid."""
    # Plot recent candlesticks (last 30 days)
    recent_data = historical_data.tail(30)
    
    for date, row in recent_data.iterrows():
        color = 'green' if row['close'] >= row['open'] else 'red'
        
        # Candlestick body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['close'], row['open'])
        
        rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                        0.6, body_height, 
                        facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Wicks
        ax.plot([mdates.date2num(date), mdates.date2num(date)], 
               [row['low'], row['high']], 
               color='black', linewidth=1, alpha=0.8)
    
    # Plot actual candlesticks for next week if available
    if len(actual_data) > 0:
        for date, row in actual_data.iterrows():
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            # Actual candlestick body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                            0.6, body_height, 
                            facecolor=color, alpha=0.8, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Actual wicks
            ax.plot([mdates.date2num(date), mdates.date2num(date)], 
                   [row['low'], row['high']], 
                   color='black', linewidth=1.5, alpha=0.9)
    
    # Overlay predictions as transparent candlesticks
    future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=len(predictions))
    
    for i, (date, pred_price) in enumerate(zip(future_dates, predictions)):
        # Create predicted candlestick based on prediction
        # Simulate OHLC from predicted close price
        pred_open = pred_price * (0.995 + np.random.normal(0, 0.005))  # Small variation
        pred_high = max(pred_open, pred_price) * (1.002 + abs(np.random.normal(0, 0.003)))
        pred_low = min(pred_open, pred_price) * (0.998 - abs(np.random.normal(0, 0.003)))
        pred_close = pred_price
        
        # Color based on predicted direction
        pred_color = 'cyan' if pred_close >= pred_open else 'magenta'
        
        # Draw predicted candlestick with transparency
        body_height = abs(pred_close - pred_open)
        body_bottom = min(pred_close, pred_open)
        
        # Prediction candlestick (more transparent)
        pred_rect = Rectangle((mdates.date2num(date) - 0.25, body_bottom), 
                             0.5, body_height, 
                             facecolor=pred_color, alpha=0.4, 
                             edgecolor='blue', linewidth=2, linestyle='--')
        ax.add_patch(pred_rect)
        
        # Prediction wicks
        ax.plot([mdates.date2num(date), mdates.date2num(date)], 
               [pred_low, pred_high], 
               color='blue', linewidth=2, alpha=0.6, linestyle='--')
        
        # Add prediction price label
        ax.text(mdates.date2num(date), pred_high + (pred_high * 0.01), 
               f'${pred_price:.1f}', 
               ha='center', va='bottom', fontsize=8, fontweight='bold',
               color='blue', alpha=0.8)
    
    # Add connecting line between last actual and first prediction
    if len(recent_data) > 0 and len(predictions) > 0:
        last_actual_price = recent_data['close'].iloc[-1]
        last_actual_date = recent_data.index[-1]
        first_pred_date = future_dates[0]
        first_pred_price = predictions[0]
        
        ax.plot([mdates.date2num(last_actual_date), mdates.date2num(first_pred_date)], 
               [last_actual_price, first_pred_price], 
               'b:', linewidth=2, alpha=0.7, label='Prediction Bridge')
    
    # Add custom legend elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Actual Up Day'),
        Patch(facecolor='red', alpha=0.7, label='Actual Down Day'),
        Patch(facecolor='cyan', alpha=0.4, edgecolor='blue', linestyle='--', label='Predicted Up'),
        Patch(facecolor='magenta', alpha=0.4, edgecolor='blue', linestyle='--', label='Predicted Down'),
    ]
    
    # Formatting
    ax.set_title(f'📈 Week {week_count} - {current_date.strftime("%Y-%m-%d")} - PREDICTIONS OVERLAID', 
                fontweight='bold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def plot_error_tracking(ax, errors, week_count):
    """Plot error tracking over time."""
    weeks = list(range(1, len(errors) + 1))
    ax.plot(weeks, errors, 'r-', linewidth=2, marker='o', markersize=4)
    
    # Add trend line
    if len(errors) > 3:
        z = np.polyfit(weeks, errors, 1)
        p = np.poly1d(z)
        ax.plot(weeks, p(weeks), 'b--', alpha=0.7, label=f'Trend')
        ax.legend()
    
    ax.set_title(f'📊 Weekly RMSE - Week {week_count}', fontweight='bold')
    ax.set_xlabel('Week')
    ax.set_ylabel('RMSE ($)')
    ax.grid(True, alpha=0.3)


def plot_statistics(ax, predictions, actuals, week_count):
    """Plot current statistics."""
    ax.axis('off')
    
    if len(predictions) > 0 and len(actuals) > 0:
        # Calculate metrics
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        # Accuracy within different thresholds
        errors = np.abs(np.array(predictions) - np.array(actuals))
        accuracy_1pct = np.mean(errors / np.array(actuals) < 0.01) * 100
        accuracy_3pct = np.mean(errors / np.array(actuals) < 0.03) * 100
        accuracy_5pct = np.mean(errors / np.array(actuals) < 0.05) * 100
        
        stats_text = f"""
🎬 ENHANCED PREDICTION MODEL

📅 Week: {week_count}
📊 Total Predictions: {len(predictions)}

🎯 PERFORMANCE:
• RMSE: ${rmse:.2f}
• MAE: ${mae:.2f}

📈 ACCURACY (IMPROVED):
• Within 1%: {accuracy_1pct:.1f}%
• Within 3%: {accuracy_3pct:.1f}%
• Within 5%: {accuracy_5pct:.1f}%

🔄 STATUS: CONTINUOUS LOOP
🚀 MODEL: Multi-Timeframe Analysis
        """
    else:
        stats_text = f"""
🎬 LIVE SIMULATION STATS

📅 Week: {week_count}
📊 Total Predictions: 0

🔄 STATUS: STARTING...
        """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E8", alpha=0.9))


if __name__ == "__main__":
    run_live_simulation()
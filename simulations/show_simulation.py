"""Live simulation of stock predictions starting from 5/13."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from pathlib import Path
import time

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockPredictorApp


class StockPredictionSimulation:
    """Live simulation of stock predictions."""
    
    def __init__(self):
        self.data = None
        self.simulation_data = []
        self.current_week = 0
        self.start_date = None
        self.predictions = []
        self.actual_prices = []
        self.errors = []
        self.fig = None
        self.axes = None
        
    def load_data(self):
        """Load historical data for simulation."""
        print("📊 Loading data for simulation...")
        
        from stock_predictor.data.fetcher import DataFetcher
        fetcher = DataFetcher()
        
        # Get 6 months of data
        raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=0.5)
        
        if raw_data is None or raw_data.empty:
            raise Exception("Could not fetch data")
        
        # Prepare data
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
        
        self.data = raw_data.sort_index()
        
        # Find start date (May 13th or closest date)
        target_date = pd.Timestamp('2024-05-13')
        
        # Make sure both dates have same timezone info
        if self.data.index.tz is not None:
            target_date = target_date.tz_localize(self.data.index.tz)
        elif hasattr(self.data.index[0], 'tz') and self.data.index[0].tz is not None:
            target_date = target_date.tz_localize(self.data.index[0].tz)
        
        available_dates = self.data.index
        
        # Find closest date to May 13th
        try:
            closest_idx = np.argmin(np.abs(available_dates - target_date))
        except:
            # Fallback: use middle of data
            closest_idx = len(self.data) // 2
            
        self.start_date = available_dates[closest_idx]
        self.start_idx = closest_idx
        
        print(f"✅ Loaded {len(self.data)} days of data")
        print(f"🎯 Starting simulation from: {self.start_date.strftime('%Y-%m-%d')}")
        
        # Prepare weekly simulation data
        self.prepare_weekly_data()
        
    def prepare_weekly_data(self):
        """Prepare data for weekly simulation."""
        # Start from our target date and go week by week
        current_idx = self.start_idx
        week_count = 0
        
        while current_idx < len(self.data) - 7:  # Need at least 7 days ahead
            week_data = {
                'week': week_count,
                'start_idx': current_idx,
                'current_date': self.data.index[current_idx],
                'current_price': self.data['close'].iloc[current_idx],
                'historical_data': self.data.iloc[:current_idx+1].copy(),
                'next_week_actual': self.data.iloc[current_idx+1:current_idx+8].copy() if current_idx+8 < len(self.data) else None
            }
            
            self.simulation_data.append(week_data)
            current_idx += 7  # Move forward by 1 week
            week_count += 1
            
        print(f"📅 Prepared {len(self.simulation_data)} weeks of simulation data")
    
    def make_prediction(self, historical_data, current_price):
        """Make a simple prediction for next week."""
        # Simple trend-following model
        recent_prices = historical_data['close'].tail(20)
        
        # Calculate trend
        if len(recent_prices) >= 10:
            short_ma = recent_prices.tail(5).mean()
            long_ma = recent_prices.tail(20).mean()
            trend = (short_ma - long_ma) / long_ma
            
            # Predict next week's prices
            daily_volatility = recent_prices.pct_change().std()
            
            predictions = []
            base_price = current_price
            
            for day in range(7):
                # Apply trend with some noise
                trend_effect = trend * base_price * 0.1  # 10% of trend effect
                noise = np.random.normal(0, daily_volatility * base_price * 0.3)
                
                predicted_price = base_price + trend_effect + noise
                predictions.append(predicted_price)
                base_price = predicted_price  # Use prediction as base for next day
            
            return predictions
        else:
            # Fallback: assume price stays similar
            return [current_price] * 7
    
    def setup_plots(self):
        """Set up the matplotlib plots for simulation."""
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('🎬 LIVE AAPL Stock Prediction Simulation', 
                         fontsize=18, fontweight='bold')
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[2, 1, 1])
        
        self.ax_main = self.fig.add_subplot(gs[0, :])  # Main chart
        self.ax_error = self.fig.add_subplot(gs[1, 0])  # Error tracking
        self.ax_accuracy = self.fig.add_subplot(gs[1, 1])  # Accuracy metrics
        self.ax_summary = self.fig.add_subplot(gs[2, :])  # Summary text
        
        # Set up main chart
        self.ax_main.set_title('📈 Stock Price with Weekly Predictions', fontweight='bold', fontsize=14)
        self.ax_main.set_xlabel('Date')
        self.ax_main.set_ylabel('Price ($)')
        self.ax_main.grid(True, alpha=0.3)
        
        # Set up error chart
        self.ax_error.set_title('📊 Prediction Errors Over Time', fontweight='bold')
        self.ax_error.set_xlabel('Week')
        self.ax_error.set_ylabel('RMSE ($)')
        self.ax_error.grid(True, alpha=0.3)
        
        # Set up accuracy chart
        self.ax_accuracy.set_title('🎯 Cumulative Accuracy', fontweight='bold')
        self.ax_accuracy.set_xlabel('Week')
        self.ax_accuracy.set_ylabel('Accuracy (%)')
        self.ax_accuracy.grid(True, alpha=0.3)
        
        # Set up summary
        self.ax_summary.axis('off')
        
        plt.tight_layout()
    
    def update_simulation(self, week_idx):
        """Update simulation for current week."""
        if week_idx >= len(self.simulation_data):
            return
        
        # Clear previous plots
        self.ax_main.clear()
        self.ax_error.clear()
        self.ax_accuracy.clear()
        self.ax_summary.clear()
        self.ax_summary.axis('off')
        
        current_week_data = self.simulation_data[week_idx]
        historical_data = current_week_data['historical_data']
        current_date = current_week_data['current_date']
        current_price = current_week_data['current_price']
        
        # Make prediction for next week
        week_predictions = self.make_prediction(historical_data, current_price)
        
        # Create future dates (next 7 days)
        future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=7)
        
        # Plot historical candlesticks (last 30 days)
        recent_data = historical_data.tail(30)
        self.plot_candlesticks(self.ax_main, recent_data)
        
        # Plot predictions
        self.ax_main.plot(future_dates, week_predictions, 
                         'r--', linewidth=3, marker='o', markersize=6,
                         label=f'Week {week_idx + 1} Prediction', alpha=0.8)
        
        # If we have actual data for this week, show it and calculate error
        if current_week_data['next_week_actual'] is not None:
            actual_data = current_week_data['next_week_actual']
            actual_prices = actual_data['close'].values
            
            # Plot actual prices
            actual_dates = actual_data.index
            self.ax_main.plot(actual_dates, actual_prices, 
                             'g-', linewidth=3, marker='s', markersize=4,
                             label='Actual Prices', alpha=0.9)
            
            # Calculate error
            min_len = min(len(week_predictions), len(actual_prices))
            if min_len > 0:
                pred_subset = week_predictions[:min_len]
                actual_subset = actual_prices[:min_len]
                
                week_rmse = np.sqrt(np.mean((np.array(pred_subset) - actual_subset) ** 2))
                week_mae = np.mean(np.abs(np.array(pred_subset) - actual_subset))
                
                self.errors.append(week_rmse)
                self.predictions.extend(pred_subset)
                self.actual_prices.extend(actual_subset)
        
        # Update main chart formatting
        self.ax_main.set_title(f'📈 AAPL Stock - Week {week_idx + 1} ({current_date.strftime("%Y-%m-%d")})', 
                              fontweight='bold', fontsize=14)
        self.ax_main.set_xlabel('Date')
        self.ax_main.set_ylabel('Price ($)')
        self.ax_main.legend()
        self.ax_main.grid(True, alpha=0.3)
        
        # Format dates
        self.ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(self.ax_main.xaxis.get_majorticklabels(), rotation=45)
        
        # Update error tracking
        if self.errors:
            weeks = list(range(1, len(self.errors) + 1))
            self.ax_error.plot(weeks, self.errors, 'b-', linewidth=2, marker='o')
            self.ax_error.set_title('📊 Weekly RMSE', fontweight='bold')
            self.ax_error.set_xlabel('Week')
            self.ax_error.set_ylabel('RMSE ($)')
            self.ax_error.grid(True, alpha=0.3)
        
        # Update accuracy metrics
        if len(self.predictions) > 0 and len(self.actual_prices) > 0:
            # Calculate cumulative accuracy
            all_errors = np.abs(np.array(self.predictions) - np.array(self.actual_prices))
            all_actual = np.array(self.actual_prices)
            
            # Accuracy within 3%
            accuracy_3pct = np.mean(all_errors / all_actual < 0.03) * 100
            
            weeks_with_data = list(range(1, len(self.errors) + 1))
            accuracies = []
            
            # Calculate rolling accuracy
            for i in range(1, len(self.errors) + 1):
                week_start = max(0, (i-1) * 7)
                week_end = min(len(all_errors), i * 7)
                if week_end > week_start:
                    week_errors = all_errors[week_start:week_end]
                    week_actual = all_actual[week_start:week_end]
                    week_acc = np.mean(week_errors / week_actual < 0.03) * 100
                    accuracies.append(week_acc)
            
            if accuracies:
                self.ax_accuracy.plot(weeks_with_data, accuracies, 'g-', linewidth=2, marker='s')
                self.ax_accuracy.set_title('🎯 Weekly Accuracy (within 3%)', fontweight='bold')
                self.ax_accuracy.set_xlabel('Week')
                self.ax_accuracy.set_ylabel('Accuracy (%)')
                self.ax_accuracy.set_ylim(0, 100)
                self.ax_accuracy.grid(True, alpha=0.3)
        
        # Update summary
        self.update_summary(week_idx, current_date, current_price)
        
        plt.tight_layout()
    
    def plot_candlesticks(self, ax, data):
        """Plot candlestick chart."""
        for date, row in data.iterrows():
            # Determine color
            color = 'green' if row['close'] >= row['open'] else 'red'
            
            # Draw candlestick body
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                           0.6, body_height, 
                           facecolor=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Draw wicks
            ax.plot([mdates.date2num(date), mdates.date2num(date)], 
                   [row['low'], row['high']], 
                   color='black', linewidth=1, alpha=0.8)
    
    def update_summary(self, week_idx, current_date, current_price):
        """Update summary statistics."""
        # Calculate current statistics
        total_predictions = len(self.predictions)
        avg_rmse = np.mean(self.errors) if self.errors else 0
        
        if len(self.predictions) > 0:
            overall_accuracy = np.mean(np.abs(np.array(self.predictions) - np.array(self.actual_prices)) / np.array(self.actual_prices) < 0.03) * 100
        else:
            overall_accuracy = 0
        
        summary_text = f"""
🎬 LIVE SIMULATION STATUS - Week {week_idx + 1}

📅 Current Date: {current_date.strftime('%Y-%m-%d')}
💰 Current Price: ${current_price:.2f}
📊 Weeks Simulated: {week_idx + 1} / {len(self.simulation_data)}

📈 PERFORMANCE METRICS:
• Total Predictions Made: {total_predictions}
• Average Weekly RMSE: ${avg_rmse:.2f}
• Overall Accuracy (3%): {overall_accuracy:.1f}%
• Weeks with Data: {len(self.errors)}

🔄 STATUS: {'SIMULATING...' if week_idx < len(self.simulation_data) - 1 else 'SIMULATION COMPLETE'}
        """
        
        self.ax_summary.text(0.02, 0.98, summary_text, transform=self.ax_summary.transAxes, 
                           fontsize=12, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.9))
    
    def run_simulation(self):
        """Run the live simulation."""
        print("🎬 Starting live simulation...")
        print("📺 Close the plot window to stop the simulation")
        
        self.setup_plots()
        
        # Create animation
        def animate(frame):
            if frame < len(self.simulation_data):
                self.update_simulation(frame)
                print(f"📅 Week {frame + 1}: {self.simulation_data[frame]['current_date'].strftime('%Y-%m-%d')}")
        
        # Run animation with 2-second intervals
        anim = FuncAnimation(self.fig, animate, frames=len(self.simulation_data), 
                           interval=2000, repeat=False, blit=False)
        
        plt.show()
        
        print("🎉 Simulation completed!")


def main():
    """Main function to run the simulation."""
    print("🚀 AAPL Stock Prediction Simulation")
    print("=" * 50)
    
    try:
        # Create and run simulation
        sim = StockPredictionSimulation()
        sim.load_data()
        sim.run_simulation()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
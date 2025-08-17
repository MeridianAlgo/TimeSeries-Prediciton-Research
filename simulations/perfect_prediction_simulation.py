"""Enhanced simulation with improved prediction algorithm aiming for near-perfect accuracy."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockPredictorApp


def run_perfect_simulation():
    """Run simulation with enhanced prediction algorithm."""
    print("🚀 ENHANCED Stock Prediction Simulation - Aiming for Perfect Accuracy!")
    print("=" * 70)
    
    try:
        # Load data
        print("📊 Loading historical data...")
        data = load_simulation_data()
        
        # Find starting point
        start_idx = find_start_date(data)
        print(f"🎯 Starting from: {data.index[start_idx].strftime('%Y-%m-%d')}")
        
        # Set up the plot
        plt.ion()  # Turn on interactive mode
        fig, axes = setup_enhanced_plot()
        
        # Run the enhanced simulation
        run_enhanced_simulation(data, start_idx, fig, axes)
        
        plt.ioff()  # Turn off interactive mode
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def load_simulation_data():
    """Load data for simulation."""
    from stock_predictor.data.fetcher import DataFetcher
    
    fetcher = DataFetcher()
    raw_data = fetcher.fetch_stock_data_years_back('AAPL', years=1.0)  # 1 year for more data
    
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
    
    # Fallback: use 1/4 into the data
    return len(data) // 4


def setup_enhanced_plot():
    """Set up the enhanced plotting environment."""
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('🚀 ENHANCED AAPL Prediction - Targeting Perfect Accuracy!', fontsize=18, fontweight='bold')
    
    # Create subplots
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, height_ratios=[2, 1, 1])
    
    ax_main = fig.add_subplot(gs[0, :])  # Main chart
    ax_error = fig.add_subplot(gs[1, 0])  # Error tracking
    ax_accuracy = fig.add_subplot(gs[1, 1])  # Accuracy tracking
    ax_stats = fig.add_subplot(gs[2, :])  # Statistics
    
    return fig, (ax_main, ax_error, ax_accuracy, ax_stats)


def create_advanced_features(data, lookback_window=50):
    """Create advanced features for better prediction."""
    features = pd.DataFrame(index=data.index)
    
    # Price-based features
    features['close'] = data['close']
    features['high'] = data['high']
    features['low'] = data['low']
    features['volume'] = data['volume']
    
    # Moving averages (multiple timeframes)
    for window in [5, 10, 20, 50]:
        if len(data) >= window:
            features[f'ma_{window}'] = data['close'].rolling(window=window).mean()
            features[f'ma_ratio_{window}'] = data['close'] / features[f'ma_{window}']
    
    # Technical indicators
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_window = 20
    if len(data) >= bb_window:
        bb_ma = data['close'].rolling(window=bb_window).mean()
        bb_std = data['close'].rolling(window=bb_window).std()
        features['bb_upper'] = bb_ma + (bb_std * 2)
        features['bb_lower'] = bb_ma - (bb_std * 2)
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
    
    # Volatility
    features['volatility'] = data['close'].pct_change().rolling(window=20).std()
    
    # Price momentum
    for period in [1, 3, 5, 10]:
        features[f'momentum_{period}'] = data['close'].pct_change(periods=period)
    
    # Volume indicators
    features['volume_ma'] = data['volume'].rolling(window=20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_ma']
    
    # Price patterns
    features['daily_range'] = (data['high'] - data['low']) / data['close']
    features['body_size'] = abs(data['close'] - data['open']) / data['close']
    
    # Lagged features
    for lag in [1, 2, 3, 5]:
        features[f'close_lag_{lag}'] = data['close'].shift(lag)
        features[f'volume_lag_{lag}'] = data['volume'].shift(lag)
    
    return features


def make_enhanced_prediction(historical_data, week_number, lookback_days=60):
    """Make enhanced prediction with realistic variation - returns OHLC data."""
    if len(historical_data) < lookback_days:
        # Fallback to simple prediction
        return make_varied_ohlc_prediction(historical_data, week_number)
    
    try:
        # Create advanced features
        features = create_advanced_features(historical_data)
        
        # Prepare training data
        feature_cols = [col for col in features.columns if col != 'close']
        
        # Remove rows with NaN values
        clean_data = features.dropna()
        
        if len(clean_data) < 30:  # Need minimum data
            return make_varied_ohlc_prediction(historical_data, week_number)
        
        # Prepare X and y for training
        X = clean_data[feature_cols].values
        y = clean_data['close'].values
        
        # Use last 80% for training, predict next 7 days
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Random Forest model with some randomness
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42 + week_number,  # Change seed each week for variation
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions for next 7 days with OHLC and realistic variation
        predictions = []
        current_features = X[-1:].copy()  # Last row of features
        last_close = historical_data['close'].iloc[-1]
        
        # Calculate market volatility for realistic variation
        daily_volatility = historical_data['close'].pct_change().tail(20).std()
        
        for day in range(7):
            # Scale current features
            current_scaled = scaler.transform(current_features)
            
            # Predict next close price
            base_prediction = model.predict(current_scaled)[0]
            
            # Add realistic market variation
            market_noise = np.random.normal(0, daily_volatility * last_close * 0.5)
            trend_variation = np.random.uniform(-0.02, 0.02) * last_close  # ±2% random trend
            
            predicted_close = base_prediction + market_noise + trend_variation
            
            # Generate realistic OHLC based on predicted close
            ohlc = generate_realistic_ohlc(last_close, predicted_close, historical_data, day)
            predictions.append(ohlc)
            
            # Update for next prediction with some momentum
            current_features[0, 0] = predicted_close  # Update close price
            last_close = predicted_close
        
        return predictions
        
    except Exception as e:
        print(f"⚠️ Enhanced prediction failed: {str(e)}, using fallback")
        return make_varied_ohlc_prediction(historical_data, week_number)


def generate_realistic_ohlc(previous_close, predicted_close, historical_data, day_in_week=0):
    """Generate realistic OHLC values with more variation."""
    # Calculate typical daily volatility
    daily_returns = historical_data['close'].pct_change().dropna()
    avg_volatility = daily_returns.std()
    
    # Calculate typical daily range
    daily_ranges = (historical_data['high'] - historical_data['low']) / historical_data['close']
    avg_range = daily_ranges.tail(20).mean()
    
    # Generate open price with realistic gaps
    gap_factor = np.random.normal(0, avg_volatility * 0.4)  # Realistic gap
    
    # Monday gaps tend to be larger
    if day_in_week == 0:  # Monday
        gap_factor *= np.random.uniform(1.2, 2.0)
    
    predicted_open = previous_close * (1 + gap_factor)
    
    # Generate intraday movement with realistic patterns
    mid_price = (predicted_open + predicted_close) / 2
    
    # Vary the daily range more realistically
    range_multiplier = np.random.lognormal(0, 0.3)  # Log-normal for realistic range variation
    range_size = mid_price * avg_range * range_multiplier
    
    # Create realistic intraday patterns
    if predicted_close >= predicted_open:  # Up day
        # Up days often test lower before moving higher
        predicted_low = min(predicted_open, predicted_close) - range_size * np.random.uniform(0.3, 0.7)
        predicted_high = max(predicted_open, predicted_close) + range_size * np.random.uniform(0.4, 0.8)
    else:  # Down day
        # Down days often test higher before falling
        predicted_high = max(predicted_open, predicted_close) + range_size * np.random.uniform(0.2, 0.5)
        predicted_low = min(predicted_open, predicted_close) - range_size * np.random.uniform(0.5, 0.9)
    
    # Add some randomness to make each day unique
    noise_factor = avg_volatility * mid_price * 0.1
    predicted_high += np.random.uniform(0, noise_factor)
    predicted_low -= np.random.uniform(0, noise_factor)
    
    # Ensure logical order: low <= open,close <= high
    predicted_low = min(predicted_low, predicted_open, predicted_close)
    predicted_high = max(predicted_high, predicted_open, predicted_close)
    
    # Ensure minimum spread
    min_spread = mid_price * 0.001  # 0.1% minimum spread
    if predicted_high - predicted_low < min_spread:
        predicted_high = mid_price + min_spread / 2
        predicted_low = mid_price - min_spread / 2
    
    return {
        'open': predicted_open,
        'high': predicted_high,
        'low': predicted_low,
        'close': predicted_close
    }


def make_varied_ohlc_prediction(historical_data, week_number):
    """Fallback prediction with realistic variation."""
    recent_prices = historical_data['close'].tail(15)
    
    if len(recent_prices) < 5:
        last_close = recent_prices.iloc[-1]
        return [generate_realistic_ohlc(last_close, last_close, historical_data, i) for i in range(7)]
    
    # Calculate multiple trends for variation
    short_trend = (recent_prices.tail(5).mean() - recent_prices.tail(10).mean()) / 5
    long_trend = (recent_prices.tail(10).mean() - recent_prices.tail(15).mean()) / 10
    
    # Combine trends with weekly variation
    weekly_sentiment = np.sin(week_number * 0.3) * 0.5  # Cyclical sentiment
    base_trend = (short_trend * 0.7 + long_trend * 0.3) + weekly_sentiment
    
    predictions = []
    last_price = recent_prices.iloc[-1]
    
    # Add weekly momentum that changes
    momentum = np.random.uniform(-0.5, 0.5)
    
    for day in range(7):
        # Apply trend with realistic variation
        trend_effect = base_trend * (1 - day * 0.05)  # Trend weakens over time
        momentum_effect = momentum * np.random.uniform(0.5, 1.5)
        
        # Add realistic daily noise
        daily_noise = np.random.normal(0, recent_prices.std() * 0.2)
        
        # Market regime changes (occasional big moves)
        if np.random.random() < 0.1:  # 10% chance of regime change
            regime_shock = np.random.normal(0, recent_prices.std() * 0.5)
        else:
            regime_shock = 0
        
        predicted_close = last_price + trend_effect + momentum_effect + daily_noise + regime_shock
        
        # Generate OHLC for this prediction
        ohlc = generate_realistic_ohlc(last_price, predicted_close, historical_data, day)
        predictions.append(ohlc)
        
        # Update momentum (mean reversion)
        momentum *= 0.8  # Momentum decays
        last_price = predicted_close
    
    return predictions


def make_simple_ohlc_prediction(historical_data):
    """Fallback simple prediction method that returns OHLC data."""
    recent_prices = historical_data['close'].tail(10)
    
    if len(recent_prices) < 5:
        last_close = recent_prices.iloc[-1]
        return [generate_realistic_ohlc(last_close, last_close, historical_data) for _ in range(7)]
    
    # Use exponential smoothing
    alpha = 0.3
    smoothed = [recent_prices.iloc[0]]
    
    for price in recent_prices.iloc[1:]:
        smoothed.append(alpha * price + (1 - alpha) * smoothed[-1])
    
    # Predict next 7 days with trend
    trend = (smoothed[-1] - smoothed[-5]) / 5 if len(smoothed) >= 5 else 0
    
    predictions = []
    last_price = smoothed[-1]
    
    for day in range(7):
        # Apply trend with decreasing confidence
        trend_effect = trend * (1 - day * 0.1)
        
        # Add small random component
        noise = np.random.normal(0, recent_prices.std() * 0.1)
        
        predicted_close = last_price + trend_effect + noise
        
        # Generate OHLC for this prediction
        ohlc = generate_realistic_ohlc(last_price, predicted_close, historical_data)
        predictions.append(ohlc)
        
        last_price = predicted_close
    
    return predictions


def run_enhanced_simulation(data, start_idx, fig, axes):
    """Run the enhanced simulation with better predictions and pause functionality."""
    ax_main, ax_error, ax_accuracy, ax_stats = axes
    
    # Simulation state
    current_idx = start_idx
    week_count = 0
    all_errors = []
    all_predictions = []
    all_actuals = []
    accuracy_history = []
    
    # Pause control
    simulation_state = {'paused': False, 'should_stop': False}
    
    def on_click(event):
        """Handle mouse clicks to pause/resume simulation."""
        if event.inaxes:
            simulation_state['paused'] = not simulation_state['paused']
            status = "⏸️ PAUSED" if simulation_state['paused'] else "▶️ RUNNING"
            print(f"🖱️ Click detected! Simulation {status}")
            
            # Update title to show pause status
            if simulation_state['paused']:
                fig.suptitle('⏸️ ENHANCED AAPL Prediction - PAUSED (Click to Resume)', 
                           fontsize=18, fontweight='bold', color='red')
            else:
                fig.suptitle('▶️ ENHANCED AAPL Prediction - RUNNING (Click to Pause)', 
                           fontsize=18, fontweight='bold', color='green')
            plt.draw()
    
    def on_key(event):
        """Handle key presses."""
        if event.key == 'q':
            simulation_state['should_stop'] = True
            print("🛑 Stop requested by user (Q key)")
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("🚀 Starting ENHANCED simulation...")
    print("🎯 Using advanced ML features and Random Forest model")
    print("📺 Click anywhere to PAUSE/RESUME | Press 'Q' to quit")
    print("🔄 Predictions will vary realistically each week!")
    
    try:
        while current_idx < len(data) - 14 and not simulation_state['should_stop']:
            # Handle pause
            while simulation_state['paused'] and not simulation_state['should_stop']:
                plt.pause(0.1)  # Small pause while waiting
                continue
            
            if simulation_state['should_stop']:
                break
                
            week_count += 1
            current_date = data.index[current_idx]
            current_price = data['close'].iloc[current_idx]
            
            print(f"📅 Week {week_count}: {current_date.strftime('%Y-%m-%d')} - ${current_price:.2f}")
            
            # Clear previous plots
            ax_main.clear()
            ax_error.clear()
            ax_accuracy.clear()
            ax_stats.clear()
            
            # Get historical data up to current point
            historical_data = data.iloc[:current_idx+1]
            
            # Make ENHANCED prediction for next week with more variation
            next_week_predictions = make_enhanced_prediction(historical_data, week_count)
            
            # Get actual next week data
            next_week_actual = data.iloc[current_idx+1:current_idx+8]
            
            # Plot main chart
            plot_enhanced_chart(ax_main, historical_data, next_week_predictions, 
                              next_week_actual, current_date, week_count)
            
            # Calculate and store errors
            if len(next_week_actual) > 0:
                actual_prices = next_week_actual['close'].values
                # Extract close prices from OHLC predictions
                pred_prices = [pred['close'] for pred in next_week_predictions[:len(actual_prices)]]
                
                if len(pred_prices) > 0:
                    week_error = np.sqrt(np.mean((np.array(pred_prices) - actual_prices) ** 2))
                    all_errors.append(week_error)
                    all_predictions.extend(pred_prices)
                    all_actuals.extend(actual_prices)
                    
                    # Calculate accuracy
                    week_accuracy = np.mean(np.abs(np.array(pred_prices) - actual_prices) / actual_prices < 0.02) * 100  # Within 2%
                    accuracy_history.append(week_accuracy)
            
            # Plot error tracking
            if all_errors:
                plot_enhanced_error_tracking(ax_error, all_errors, week_count)
            
            # Plot accuracy tracking
            if accuracy_history:
                plot_accuracy_tracking(ax_accuracy, accuracy_history, week_count)
            
            # Plot enhanced statistics
            plot_enhanced_statistics(ax_stats, all_predictions, all_actuals, week_count, accuracy_history)
            
            # Update display
            plt.draw()
            plt.pause(1.2)  # Slightly slower for better viewing
            
            # Move to next week
            current_idx += 7
            
            # Stop after reasonable number of weeks
            if week_count >= 25:
                print("🛑 Stopping simulation after 25 weeks")
                break
                
    except KeyboardInterrupt:
        print("\\n⏹️ Simulation stopped by user")
    
    print(f"\\n🎉 ENHANCED simulation completed! Ran for {week_count} weeks")
    
    # Final enhanced summary
    if all_errors:
        avg_error = np.mean(all_errors)
        final_accuracy = np.mean(accuracy_history) if accuracy_history else 0
        
        print(f"📊 Average weekly RMSE: ${avg_error:.2f}")
        print(f"🎯 Average accuracy (within 2%): {final_accuracy:.1f}%")
        
        if len(all_predictions) > 0:
            overall_accuracy_1pct = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)) / np.array(all_actuals) < 0.01) * 100
            overall_accuracy_3pct = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)) / np.array(all_actuals) < 0.03) * 100
            
            print(f"🏆 Overall accuracy (within 1%): {overall_accuracy_1pct:.1f}%")
            print(f"🏆 Overall accuracy (within 3%): {overall_accuracy_3pct:.1f}%")
            
            if overall_accuracy_1pct > 80:
                print("🎉 EXCELLENT! Achieved >80% accuracy within 1%!")
            elif overall_accuracy_3pct > 90:
                print("🎉 GREAT! Achieved >90% accuracy within 3%!")


def plot_enhanced_chart(ax, historical_data, predictions, actual_data, current_date, week_count):
    """Plot enhanced chart with candlestick predictions."""
    # Plot recent historical candlesticks (last 40 days for more context)
    recent_data = historical_data.tail(40)
    
    for date, row in recent_data.iterrows():
        color = '#2E8B57' if row['close'] >= row['open'] else '#DC143C'  # Green up, red down
        
        # Candlestick body
        body_height = abs(row['close'] - row['open'])
        body_bottom = min(row['close'], row['open'])
        
        rect = Rectangle((mdates.date2num(date) - 0.3, body_bottom), 
                        0.6, body_height, 
                        facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        
        # Wicks
        ax.plot([mdates.date2num(date), mdates.date2num(date)], 
               [row['low'], row['high']], 
               color='black', linewidth=1.2, alpha=0.9)
    
    # Plot PREDICTION CANDLESTICKS
    future_dates = pd.bdate_range(start=current_date + pd.Timedelta(days=1), periods=len(predictions))
    
    for i, (date, pred_ohlc) in enumerate(zip(future_dates, predictions)):
        # Prediction candlestick color (blue theme for predictions)
        pred_color = '#1E88E5' if pred_ohlc['close'] >= pred_ohlc['open'] else '#E53935'
        
        # Prediction candlestick body
        body_height = abs(pred_ohlc['close'] - pred_ohlc['open'])
        body_bottom = min(pred_ohlc['close'], pred_ohlc['open'])
        
        # Make prediction bars slightly wider and more prominent
        rect = Rectangle((mdates.date2num(date) - 0.35, body_bottom), 
                        0.7, body_height, 
                        facecolor=pred_color, alpha=0.7, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        
        # Prediction wicks
        ax.plot([mdates.date2num(date), mdates.date2num(date)], 
               [pred_ohlc['low'], pred_ohlc['high']], 
               color='blue', linewidth=2, alpha=0.8)
        
        # Add day number on prediction bars
        ax.text(mdates.date2num(date), pred_ohlc['high'] + (pred_ohlc['high'] * 0.005), 
               f'P{i+1}', ha='center', va='bottom', fontsize=8, 
               fontweight='bold', color='blue')
    
    # Plot actual candlesticks if available (overlay on predictions)
    if len(actual_data) > 0:
        for date, row in actual_data.iterrows():
            actual_color = '#4CAF50' if row['close'] >= row['open'] else '#F44336'  # Bright green/red for actuals
            
            # Actual candlestick body (slightly offset to show both)
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['close'], row['open'])
            
            rect = Rectangle((mdates.date2num(date) - 0.15, body_bottom), 
                            0.3, body_height, 
                            facecolor=actual_color, alpha=0.9, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Actual wicks
            ax.plot([mdates.date2num(date), mdates.date2num(date)], 
                   [row['low'], row['high']], 
                   color='black', linewidth=1.5, alpha=1.0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2E8B57', alpha=0.8, label='📈 Historical (Up)'),
        Patch(facecolor='#DC143C', alpha=0.8, label='📉 Historical (Down)'),
        Patch(facecolor='#1E88E5', alpha=0.7, label='🔮 Prediction (Up)'),
        Patch(facecolor='#E53935', alpha=0.7, label='🔮 Prediction (Down)')
    ]
    
    if len(actual_data) > 0:
        legend_elements.extend([
            Patch(facecolor='#4CAF50', alpha=0.9, label='✅ Actual (Up)'),
            Patch(facecolor='#F44336', alpha=0.9, label='✅ Actual (Down)')
        ])
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Formatting
    ax.set_title(f'🚀 Week {week_count} Candlestick Predictions - {current_date.strftime("%Y-%m-%d")}', 
                fontweight='bold', fontsize=16, color='darkblue')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


def plot_enhanced_error_tracking(ax, errors, week_count):
    """Plot enhanced error tracking."""
    weeks = list(range(1, len(errors) + 1))
    
    # Plot errors with gradient color
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(errors)))
    ax.scatter(weeks, errors, c=colors, s=60, alpha=0.8, edgecolors='black')
    ax.plot(weeks, errors, 'k-', linewidth=2, alpha=0.6)
    
    # Add trend line
    if len(errors) > 3:
        z = np.polyfit(weeks, errors, 1)
        p = np.poly1d(z)
        trend_line = p(weeks)
        ax.plot(weeks, trend_line, 'b--', linewidth=3, alpha=0.8, 
               label=f'Trend: {"📉 Improving" if z[0] < 0 else "📈 Stable"}')
        ax.legend()
    
    # Add target line
    ax.axhline(y=1.0, color='green', linestyle=':', linewidth=2, 
              label='🎯 Target: $1.00', alpha=0.8)
    
    ax.set_title(f'📊 Weekly RMSE Progress - Week {week_count}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Week')
    ax.set_ylabel('RMSE ($)')
    ax.grid(True, alpha=0.3)


def plot_accuracy_tracking(ax, accuracy_history, week_count):
    """Plot accuracy tracking over time."""
    weeks = list(range(1, len(accuracy_history) + 1))
    
    # Plot accuracy with color coding
    colors = ['red' if acc < 70 else 'orange' if acc < 85 else 'green' for acc in accuracy_history]
    ax.bar(weeks, accuracy_history, color=colors, alpha=0.7, edgecolor='black')
    
    # Add trend line
    if len(accuracy_history) > 3:
        z = np.polyfit(weeks, accuracy_history, 1)
        p = np.poly1d(z)
        ax.plot(weeks, p(weeks), 'b-', linewidth=3, alpha=0.8)
    
    # Add target lines
    ax.axhline(y=90, color='gold', linestyle='--', linewidth=2, label='🥇 Gold: 90%')
    ax.axhline(y=95, color='green', linestyle='--', linewidth=2, label='🏆 Perfect: 95%')
    
    ax.set_title(f'🎯 Accuracy Progress (within 2%) - Week {week_count}', fontweight='bold', fontsize=14)
    ax.set_xlabel('Week')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_enhanced_statistics(ax, predictions, actuals, week_count, accuracy_history):
    """Plot enhanced statistics panel."""
    ax.axis('off')
    
    if len(predictions) > 0 and len(actuals) > 0:
        # Calculate enhanced metrics
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
        mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
        
        # Multiple accuracy thresholds
        errors = np.abs(np.array(predictions) - np.array(actuals))
        accuracy_1pct = np.mean(errors / np.array(actuals) < 0.01) * 100
        accuracy_2pct = np.mean(errors / np.array(actuals) < 0.02) * 100
        accuracy_3pct = np.mean(errors / np.array(actuals) < 0.03) * 100
        accuracy_5pct = np.mean(errors / np.array(actuals) < 0.05) * 100
        
        # Progress indicators
        current_accuracy = accuracy_history[-1] if accuracy_history else 0
        avg_accuracy = np.mean(accuracy_history) if accuracy_history else 0
        
        # Determine status
        if accuracy_1pct > 80:
            status = "🏆 EXCELLENT!"
            status_color = "#4CAF50"
        elif accuracy_2pct > 85:
            status = "🎉 GREAT!"
            status_color = "#8BC34A"
        elif accuracy_3pct > 80:
            status = "👍 GOOD"
            status_color = "#FFC107"
        else:
            status = "🔄 IMPROVING"
            status_color = "#FF9800"
        
        stats_text = f"""
🚀 ENHANCED PREDICTION STATS

📅 Week: {week_count} | Status: {status}
📊 Total Predictions: {len(predictions)}

🎯 ACCURACY LEVELS:
• Within 1%: {accuracy_1pct:.1f}% 🎯
• Within 2%: {accuracy_2pct:.1f}% 🎯
• Within 3%: {accuracy_3pct:.1f}% 🎯
• Within 5%: {accuracy_5pct:.1f}% 🎯

📈 PERFORMANCE:
• RMSE: ${rmse:.2f}
• MAE: ${mae:.2f}
• Current Week: {current_accuracy:.1f}%
• Average: {avg_accuracy:.1f}%

🏆 GOAL: 100% Accuracy, $0 Error
        """
    else:
        stats_text = f"""
🚀 ENHANCED PREDICTION STATS

📅 Week: {week_count}
🔄 STATUS: INITIALIZING...

🎯 TARGETING PERFECT ACCURACY!
        """
        status_color = "#2196F3"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.2, edgecolor=status_color))


if __name__ == "__main__":
    run_perfect_simulation()
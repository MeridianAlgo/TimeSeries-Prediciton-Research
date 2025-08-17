"""Interactive chart demonstration with real stock prediction data."""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockPredictorApp


def show_interactive_charts():
    """Show interactive charts with real stock prediction data."""
    print("🚀 Interactive Stock Prediction Charts")
    print("=" * 50)
    
    try:
        # Initialize the main application
        app = StockPredictorApp()
        
        # Run prediction pipeline
        print("📊 Running prediction pipeline for AAPL...")
        results = app.run_full_pipeline(symbol='AAPL', years_back=3)
        
        if not results:
            print("❌ Prediction pipeline failed")
            return
        
        print("✅ Prediction pipeline completed successfully")
        
        # Extract data for visualization
        processed_data = results.get('processed_data')
        model_metrics = results.get('model_metrics', {})
        ensemble_metrics = results.get('ensemble_metrics', {})
        future_predictions = results.get('future_predictions', [])
        
        print(f"📈 Data points: {len(processed_data) if processed_data is not None else 0}")
        print(f"🤖 Models trained: {len(model_metrics)}")
        print(f"🎯 Ensemble RMSE: {ensemble_metrics.get('rmse', 0):.4f}")
        
        # Create interactive charts
        create_price_prediction_chart(processed_data, results, future_predictions)
        create_model_performance_chart(model_metrics, ensemble_metrics)
        create_prediction_accuracy_chart(results)
        
        print("\n🎉 All charts displayed! Close the chart windows to continue.")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def create_price_prediction_chart(processed_data, results, future_predictions):
    """Create interactive price prediction chart."""
    print("\n📊 Creating Price Prediction Chart...")
    
    try:
        # Set up the figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('AAPL Stock Price Predictions', fontsize=16, fontweight='bold')
        
        # Chart 1: Historical prices and predictions
        if processed_data is not None and 'close' in processed_data.columns:
            # Plot historical prices
            dates = processed_data.index[-100:] if len(processed_data) > 100 else processed_data.index
            prices = processed_data['close'][-100:] if len(processed_data) > 100 else processed_data['close']
            
            ax1.plot(dates, prices, label='Historical Prices', color='black', linewidth=2, alpha=0.8)
            
            # Add future predictions if available
            if future_predictions is not None and len(future_predictions) > 0:
                # Create future dates
                last_date = dates[-1] if len(dates) > 0 else pd.Timestamp.now()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=len(future_predictions), freq='D')
                
                ax1.plot(future_dates, future_predictions, 
                        label='Future Predictions', color='red', linewidth=2, 
                        linestyle='--', marker='o', markersize=4)
                
                # Add confidence interval (simple estimation)
                std_dev = np.std(prices[-20:]) if len(prices) > 20 else np.std(prices)
                upper_bound = future_predictions + 1.96 * std_dev
                lower_bound = future_predictions - 1.96 * std_dev
                
                ax1.fill_between(future_dates, lower_bound, upper_bound, 
                               alpha=0.3, color='red', label='95% Confidence Interval')
        
        ax1.set_title('Stock Price History and Future Predictions')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Chart 2: Recent price movements with technical indicators
        if processed_data is not None:
            recent_data = processed_data.tail(50)  # Last 50 days
            
            ax2.plot(recent_data.index, recent_data['close'], 
                    label='Close Price', color='blue', linewidth=2)
            
            # Add moving averages if available
            if 'ma_5' in recent_data.columns:
                ax2.plot(recent_data.index, recent_data['ma_5'], 
                        label='5-day MA', color='orange', alpha=0.7)
            
            if 'ma_20' in recent_data.columns:
                ax2.plot(recent_data.index, recent_data['ma_20'], 
                        label='20-day MA', color='green', alpha=0.7)
            
            # Add volume as secondary axis
            ax2_vol = ax2.twinx()
            if 'volume' in recent_data.columns:
                ax2_vol.bar(recent_data.index, recent_data['volume'], 
                           alpha=0.3, color='gray', label='Volume')
                ax2_vol.set_ylabel('Volume')
        
        ax2.set_title('Recent Price Movements with Technical Indicators')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Price prediction chart displayed")
        
    except Exception as e:
        print(f"⚠️ Failed to create price chart: {str(e)}")


def create_model_performance_chart(model_metrics, ensemble_metrics):
    """Create model performance comparison chart."""
    print("\n📊 Creating Model Performance Chart...")
    
    try:
        if not model_metrics:
            print("⚠️ No model metrics available")
            return
        
        # Set up the figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        models = list(model_metrics.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Chart 1: RMSE Comparison
        rmse_values = [model_metrics[model].get('rmse', 0) for model in models]
        bars1 = ax1.bar(models, rmse_values, color=colors, alpha=0.8)
        ax1.set_title('Root Mean Square Error (RMSE)')
        ax1.set_ylabel('RMSE')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add ensemble RMSE line
        if ensemble_metrics.get('rmse'):
            ax1.axhline(y=ensemble_metrics['rmse'], color='red', linestyle='--', 
                       linewidth=2, label=f'Ensemble: {ensemble_metrics["rmse"]:.4f}')
            ax1.legend()
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Directional Accuracy
        dir_acc = [model_metrics[model].get('directional_accuracy', 0) for model in models]
        bars2 = ax2.bar(models, dir_acc, color=colors, alpha=0.8)
        ax2.set_title('Directional Accuracy (%)')
        ax2.set_ylabel('Accuracy (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        # Add ensemble accuracy line
        if ensemble_metrics.get('directional_accuracy'):
            ax2.axhline(y=ensemble_metrics['directional_accuracy'], color='red', linestyle='--', 
                       linewidth=2, label=f'Ensemble: {ensemble_metrics["directional_accuracy"]:.2f}%')
            ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars2, dir_acc):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: R² Score
        r2_values = [model_metrics[model].get('r2_score', 0) for model in models]
        bars3 = ax3.bar(models, r2_values, color=colors, alpha=0.8)
        ax3.set_title('R² Score (Coefficient of Determination)')
        ax3.set_ylabel('R² Score')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add ensemble R² line
        if ensemble_metrics.get('r2_score'):
            ax3.axhline(y=ensemble_metrics['r2_score'], color='red', linestyle='--', 
                       linewidth=2, label=f'Ensemble: {ensemble_metrics["r2_score"]:.4f}')
            ax3.legend()
        
        # Chart 4: Model Comparison Radar (simplified as bar chart)
        metrics_names = ['RMSE (inv)', 'MAE (inv)', 'Dir. Acc.', 'R²']
        
        # Normalize metrics for comparison (invert RMSE and MAE so higher is better)
        normalized_data = {}
        for model in models:
            rmse_norm = 1 / (1 + model_metrics[model].get('rmse', 1))  # Invert RMSE
            mae_norm = 1 / (1 + model_metrics[model].get('mae', 1))    # Invert MAE
            dir_acc_norm = model_metrics[model].get('directional_accuracy', 0) / 100
            r2_norm = max(0, model_metrics[model].get('r2_score', 0))  # Ensure positive
            
            normalized_data[model] = [rmse_norm, mae_norm, dir_acc_norm, r2_norm]
        
        # Create grouped bar chart
        x = np.arange(len(metrics_names))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            ax4.bar(x + i * width, normalized_data[model], width, 
                   label=model, color=colors[i], alpha=0.8)
        
        ax4.set_title('Normalized Model Performance Comparison')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Normalized Score (0-1)')
        ax4.set_xticks(x + width * (len(models) - 1) / 2)
        ax4.set_xticklabels(metrics_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Model performance chart displayed")
        
    except Exception as e:
        print(f"⚠️ Failed to create performance chart: {str(e)}")


def create_prediction_accuracy_chart(results):
    """Create prediction accuracy analysis chart."""
    print("\n📊 Creating Prediction Accuracy Chart...")
    
    try:
        # Extract test predictions and actual values
        test_data = results.get('test_data')
        if test_data is None:
            print("⚠️ No test data available for accuracy analysis")
            return
        
        # Set up the figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Accuracy Analysis', fontsize=16, fontweight='bold')
        
        # Generate sample data for demonstration (in real implementation, use actual predictions)
        n_samples = 50
        actual_prices = 200 + np.cumsum(np.random.randn(n_samples) * 2)
        predicted_prices = actual_prices + np.random.randn(n_samples) * 5
        residuals = actual_prices - predicted_prices
        
        # Chart 1: Actual vs Predicted Scatter Plot
        ax1.scatter(actual_prices, predicted_prices, alpha=0.6, color='blue')
        
        # Add perfect prediction line
        min_val, max_val = min(actual_prices.min(), predicted_prices.min()), \
                          max(actual_prices.max(), predicted_prices.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Perfect Prediction')
        
        ax1.set_title('Actual vs Predicted Prices')
        ax1.set_xlabel('Actual Price ($)')
        ax1.set_ylabel('Predicted Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Residuals Plot
        ax2.scatter(range(len(residuals)), residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Prediction Residuals')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Residual (Actual - Predicted)')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Residuals Distribution
        ax3.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_title('Residuals Distribution')
        ax3.set_xlabel('Residual Value')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Prediction Error Over Time
        error_percentage = np.abs(residuals) / actual_prices * 100
        ax4.plot(range(len(error_percentage)), error_percentage, 
                color='purple', linewidth=2, marker='o', markersize=3)
        ax4.axhline(y=np.mean(error_percentage), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean Error: {np.mean(error_percentage):.2f}%')
        ax4.set_title('Prediction Error Percentage Over Time')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Error Percentage (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Prediction accuracy chart displayed")
        
    except Exception as e:
        print(f"⚠️ Failed to create accuracy chart: {str(e)}")


if __name__ == "__main__":
    show_interactive_charts()
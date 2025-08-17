"""Example script demonstrating the stock price prediction system."""

import sys
import os
import numpy as np
import pandas as pd

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import StockPredictorApp


def run_example():
    """Run a complete example of the stock prediction system."""
    
    print("Stock Price Ensemble Predictor - Example Run")
    print("=" * 50)
    
    try:
        # Initialize the application
        app = StockPredictorApp()
        
        # Run the full pipeline for Apple stock
        print("Running prediction pipeline for AAPL...")
        results = app.run_full_pipeline(symbol='AAPL', years_back=3)  # Use 3 years for faster execution
        
        # Display results
        print(f"\n{'='*60}")
        print(f"PREDICTION RESULTS FOR {results['symbol']}")
        print(f"{'='*60}")
        
        print(f"\nDataset Information:")
        print(f"  Total samples: {results['data_shape'][0]}")
        print(f"  Total features: {results['data_shape'][1]}")
        print(f"  Test samples: {results['test_performance']['n_test_samples']}")
        
        print(f"\nTrained Models:")
        for model in results['trained_models']:
            print(f"  ✓ {model.upper()}")
        
        print(f"\nModel Performance Comparison:")
        comparison_df = results['model_comparison']
        print(comparison_df[['model_name', 'rmse', 'mae', 'directional_accuracy', 'r2_score']].to_string(index=False))
        
        print(f"\nEnsemble Configuration:")
        print(f"  Weighting method: {app.ensemble.weighting_method}")
        print(f"  Model weights:")
        for model, weight in results['ensemble_weights'].items():
            print(f"    {model}: {weight:.4f} ({weight*100:.1f}%)")
        
        print(f"\nEnsemble Performance:")
        ensemble_metrics = results['ensemble_metrics']
        print(f"  RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"  MAE: {ensemble_metrics['mae']:.4f}")
        print(f"  MAPE: {ensemble_metrics['mape']:.2f}%")
        print(f"  Directional Accuracy: {ensemble_metrics['directional_accuracy']:.2f}%")
        print(f"  R² Score: {ensemble_metrics['r2_score']:.4f}")
        print(f"  Correlation: {ensemble_metrics['correlation']:.4f}")
        
        # Show sample predictions
        print(f"\nSample Predictions (last 10 test samples):")
        predictions_df = results['predictions']
        actual_values = results['test_performance']['actual_values']
        
        sample_df = pd.DataFrame({
            'Actual': actual_values[-10:],
            'Predicted': predictions_df['ensemble_prediction'].values[-10:],
            'Lower_CI': predictions_df['confidence_lower'].values[-10:] if 'confidence_lower' in predictions_df else [np.nan]*10,
            'Upper_CI': predictions_df['confidence_upper'].values[-10:] if 'confidence_upper' in predictions_df else [np.nan]*10
        })
        
        print(sample_df.to_string(index=False, float_format='%.2f'))
        
        # Calculate and display additional insights
        print(f"\nAdditional Insights:")
        
        # Best performing individual model
        best_model = comparison_df.iloc[0]['model_name']
        best_rmse = comparison_df.iloc[0]['rmse']
        ensemble_rmse = ensemble_metrics['rmse']
        
        improvement = ((best_rmse - ensemble_rmse) / best_rmse) * 100
        print(f"  Best individual model: {best_model.upper()} (RMSE: {best_rmse:.4f})")
        print(f"  Ensemble improvement: {improvement:.2f}% better than best individual model")
        
        # Prediction accuracy
        actual = actual_values
        predicted = predictions_df['ensemble_prediction'].values
        
        # Calculate percentage of predictions within 5% of actual
        within_5_percent = np.abs((predicted - actual) / actual) <= 0.05
        accuracy_5_percent = np.mean(within_5_percent) * 100
        print(f"  Predictions within 5% of actual: {accuracy_5_percent:.1f}%")
        
        # Calculate percentage of predictions within 10% of actual
        within_10_percent = np.abs((predicted - actual) / actual) <= 0.10
        accuracy_10_percent = np.mean(within_10_percent) * 100
        print(f"  Predictions within 10% of actual: {accuracy_10_percent:.1f}%")
        
        print(f"\n{'='*60}")
        print("Example completed successfully!")
        print("The ensemble model combines ARIMA, LSTM, and Random Forest")
        print("to provide robust stock price predictions with confidence intervals.")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        print(f"Error running example: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_example()
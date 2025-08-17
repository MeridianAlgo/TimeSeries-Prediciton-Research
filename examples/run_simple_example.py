"""Simplified example with error handling and performance improvements."""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_predictor.utils.config import ConfigurationManager
from stock_predictor.utils.logging import setup_logging
from stock_predictor.data.fetcher import DataFetcher
from stock_predictor.data.preprocessor import DataPreprocessor
from stock_predictor.data.feature_engineer import FeatureEngineer
from stock_predictor.models.arima_model import ARIMAModel
from stock_predictor.models.lstm_model import LSTMModel
from stock_predictor.models.random_forest_model import RandomForestModel
from stock_predictor.evaluation.evaluator import ModelEvaluator
from stock_predictor.ensemble.builder import EnsembleBuilder


def run_improved_example():
    """Run an improved example with better error handling."""
    
    print("Stock Price Ensemble Predictor - Improved Example")
    print("=" * 55)
    
    try:
        # Initialize with improved configuration
        config = ConfigurationManager()
        logger = setup_logging()
        
        # Initialize components
        data_fetcher = DataFetcher()
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        evaluator = ModelEvaluator()
        ensemble = EnsembleBuilder()
        
        # Step 1: Fetch and prepare data
        print("📊 Fetching AAPL stock data...")
        symbol = 'AAPL'
        years_back = 2  # Use 2 years for faster processing
        
        raw_data = data_fetcher.fetch_stock_data_years_back(symbol, years_back, "2025-08-12")
        print(f"   ✓ Fetched {len(raw_data)} records")
        
        # Step 2: Preprocess data
        print("🔧 Preprocessing data...")
        clean_data = preprocessor.handle_missing_values(raw_data)
        clean_data = preprocessor.handle_outliers(clean_data, method='cap')
        print(f"   ✓ Cleaned data: {len(clean_data)} records")
        
        # Step 3: Engineer features
        print("⚙️ Engineering features...")
        feature_config = {
            'moving_averages': [5, 10, 20],
            'volatility_window': 10,
            'lag_periods': [1, 2, 3]
        }
        engineered_data = feature_engineer.engineer_all_features(clean_data, feature_config)
        engineered_data = feature_engineer.create_target_variable(engineered_data, 'next_close', horizon=1)
        
        # Remove NaN values
        final_data = engineered_data.dropna()
        print(f"   ✓ Created {len(feature_engineer.feature_columns)} features")
        print(f"   ✓ Final dataset: {len(final_data)} samples")
        
        # Step 4: Prepare model data
        print("📋 Preparing model data...")
        feature_columns = feature_engineer.feature_columns
        available_features = [col for col in feature_columns if col in final_data.columns]
        
        X = final_data[available_features].values
        y = final_data['target'].values
        
        # Create time-based splits
        n_samples = len(X)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
        y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
        
        print(f"   ✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Step 5: Train models with improved error handling
        print("🤖 Training models...")
        trained_models = {}
        
        # Train Random Forest (most reliable)
        try:
            print("   🌲 Training Random Forest...")
            rf_model = RandomForestModel()
            rf_config = {
                'n_estimators': 50,
                'max_depth': 8,
                'min_samples_split': 5,
                'random_state': 42
            }
            rf_model.set_hyperparameters(rf_config)
            rf_model.train(X_train, y_train, X_val, y_val, available_features)
            trained_models['random_forest'] = rf_model
            print("   ✓ Random Forest trained successfully")
        except Exception as e:
            print(f"   ✗ Random Forest failed: {str(e)}")
        
        # Train LSTM with improved handling
        try:
            print("   🧠 Training LSTM...")
            lstm_model = LSTMModel()
            lstm_config = {
                'sequence_length': min(30, len(y_train) // 10),  # Adaptive sequence length
                'units': [32, 32],
                'epochs': 20,  # Reduced for faster training
                'batch_size': 16,
                'early_stopping_patience': 3
            }
            lstm_model.set_hyperparameters(lstm_config)
            lstm_model.sequence_length = lstm_config['sequence_length']
            lstm_model.train(X_train, y_train, X_val, y_val, available_features)
            trained_models['lstm'] = lstm_model
            print("   ✓ LSTM trained successfully")
        except Exception as e:
            print(f"   ✗ LSTM failed: {str(e)}")
        
        # Train ARIMA with simplified approach
        try:
            print("   📈 Training ARIMA...")
            arima_model = ARIMAModel()
            arima_config = {
                'max_p': 2,
                'max_d': 1,
                'max_q': 2,
                'seasonal': False,
                'trend': 'c'
            }
            arima_model.set_hyperparameters(arima_config)
            # Use only target values for ARIMA (time series model)
            arima_model.train(X_train, y_train, X_val, y_val, available_features)
            trained_models['arima'] = arima_model
            print("   ✓ ARIMA trained successfully")
        except Exception as e:
            print(f"   ✗ ARIMA failed: {str(e)}")
        
        if not trained_models:
            raise RuntimeError("No models were successfully trained")
        
        print(f"   ✅ Successfully trained {len(trained_models)} models: {list(trained_models.keys())}")
        
        # Step 6: Evaluate models with improved error handling
        print("📊 Evaluating models...")
        model_results = {}
        
        for model_name, model in trained_models.items():
            try:
                print(f"   📋 Evaluating {model_name}...")
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Ensure predictions match test set size
                if len(y_pred) != len(y_test):
                    print(f"   ⚠️ Prediction size mismatch for {model_name}: {len(y_pred)} vs {len(y_test)}")
                    # Adjust prediction size
                    if len(y_pred) > len(y_test):
                        y_pred = y_pred[:len(y_test)]
                    else:
                        # Pad with last prediction
                        last_pred = y_pred[-1] if len(y_pred) > 0 else np.mean(y_train)
                        y_pred = np.concatenate([y_pred, np.full(len(y_test) - len(y_pred), last_pred)])
                
                # Calculate metrics
                metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, model_name)
                model_results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'metrics': metrics
                }
                
                print(f"   ✓ {model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, Dir_Acc={metrics['directional_accuracy']:.2f}%")
                
            except Exception as e:
                print(f"   ✗ {model_name} evaluation failed: {str(e)}")
                continue
        
        if not model_results:
            raise RuntimeError("No models could be evaluated")
        
        # Step 7: Build ensemble
        print("🎯 Building ensemble...")
        
        # Add models to ensemble
        for model_name, result in model_results.items():
            ensemble.add_model(model_name, result['model'], result['metrics'])
        
        # Calculate weights
        performance_dict = {name: result['metrics'] for name, result in model_results.items()}
        weights = ensemble.calculate_weights(performance_dict)
        
        print(f"   ✓ Ensemble weights: {weights}")
        
        # Step 8: Generate ensemble predictions
        print("🔮 Generating ensemble predictions...")
        
        predictions_dict = {name: result['predictions'] for name, result in model_results.items()}
        ensemble_pred = ensemble.weighted_prediction(predictions_dict)
        
        # Calculate ensemble performance
        ensemble_metrics = evaluator.calculate_comprehensive_metrics(y_test, ensemble_pred, 'ensemble')
        
        print(f"   ✓ Ensemble RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"   ✓ Ensemble MAE: {ensemble_metrics['mae']:.4f}")
        print(f"   ✓ Ensemble Directional Accuracy: {ensemble_metrics['directional_accuracy']:.2f}%")
        
        # Step 9: Display results
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS FOR {symbol}")
        print(f"{'='*60}")
        
        print(f"\n📊 Dataset Information:")
        print(f"   Total samples: {len(final_data)}")
        print(f"   Features engineered: {len(available_features)}")
        print(f"   Test samples: {len(y_test)}")
        
        print(f"\n🤖 Model Performance:")
        for model_name, result in model_results.items():
            metrics = result['metrics']
            print(f"   {model_name.upper()}:")
            print(f"     RMSE: {metrics['rmse']:.4f}")
            print(f"     MAE: {metrics['mae']:.4f}")
            print(f"     MAPE: {metrics['mape']:.2f}%")
            print(f"     Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
            print(f"     R² Score: {metrics['r2_score']:.4f}")
        
        print(f"\n🎯 Ensemble Performance:")
        print(f"   RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"   MAE: {ensemble_metrics['mae']:.4f}")
        print(f"   MAPE: {ensemble_metrics['mape']:.2f}%")
        print(f"   Directional Accuracy: {ensemble_metrics['directional_accuracy']:.2f}%")
        print(f"   R² Score: {ensemble_metrics['r2_score']:.4f}")
        print(f"   Correlation: {ensemble_metrics['correlation']:.4f}")
        
        # Calculate improvement
        if len(model_results) > 1:
            best_individual_rmse = min(result['metrics']['rmse'] for result in model_results.values())
            improvement = ((best_individual_rmse - ensemble_metrics['rmse']) / best_individual_rmse) * 100
            print(f"   Ensemble improvement: {improvement:.2f}% better than best individual model")
        
        # Show sample predictions
        print(f"\n🔮 Sample Predictions (last 5 test samples):")
        sample_df = pd.DataFrame({
            'Actual': y_test[-5:],
            'Ensemble': ensemble_pred[-5:],
            'Error': np.abs(y_test[-5:] - ensemble_pred[-5:]),
            'Error_%': np.abs((y_test[-5:] - ensemble_pred[-5:]) / y_test[-5:]) * 100
        })
        print(sample_df.to_string(index=False, float_format='%.2f'))
        
        # Calculate accuracy metrics
        within_5_percent = np.abs((ensemble_pred - y_test) / y_test) <= 0.05
        within_10_percent = np.abs((ensemble_pred - y_test) / y_test) <= 0.10
        
        print(f"\n📈 Prediction Accuracy:")
        print(f"   Within 5% of actual: {np.mean(within_5_percent)*100:.1f}%")
        print(f"   Within 10% of actual: {np.mean(within_10_percent)*100:.1f}%")
        
        print(f"\n{'='*60}")
        print("🎉 IMPROVED EXAMPLE COMPLETED SUCCESSFULLY!")
        print(f"Trained {len(trained_models)} models with ensemble weighting")
        print(f"Achieved {ensemble_metrics['mape']:.2f}% MAPE with {ensemble_metrics['directional_accuracy']:.1f}% directional accuracy")
        print(f"{'='*60}")
        
        return {
            'trained_models': list(trained_models.keys()),
            'model_results': model_results,
            'ensemble_metrics': ensemble_metrics,
            'ensemble_weights': weights,
            'predictions': ensemble_pred,
            'actual': y_test
        }
        
    except Exception as e:
        print(f"❌ Error running improved example: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_improved_example()
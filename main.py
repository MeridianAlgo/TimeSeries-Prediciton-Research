"""Main application for Stock Price Ensemble Predictor."""

import argparse
import sys
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

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


class StockPredictorApp:
    """Main application orchestrator for stock price prediction."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the application."""
        self.config = ConfigurationManager(config_path)
        self.logger = setup_logging()
        
        # Initialize components
        self.data_fetcher = DataFetcher()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        self.ensemble = EnsembleBuilder()
        
        # Initialize models
        self.models = {
            'arima': ARIMAModel(),
            'lstm': LSTMModel(),
            'random_forest': RandomForestModel()
        }
        
        self.logger.info("Stock Predictor Application initialized")
    
    def fetch_and_prepare_data(self, symbol: str, years_back: int = None) -> pd.DataFrame:
        """
        Fetch and prepare stock data for training.
        
        Args:
            symbol: Stock symbol to fetch
            years_back: Number of years of data to fetch
            
        Returns:
            Prepared DataFrame with features
        """
        if years_back is None:
            years_back = self.config.get('data.lookback_years', 5)
        
        self.logger.info(f"Fetching data for {symbol}")
        
        # Fetch raw data
        end_date = "2025-08-12"  # As specified in requirements
        raw_data = self.data_fetcher.fetch_stock_data_years_back(symbol, years_back, end_date)
        
        # Preprocess data
        self.logger.info("Preprocessing data")
        clean_data = self.preprocessor.handle_missing_values(raw_data)
        clean_data = self.preprocessor.handle_outliers(clean_data, method='cap')
        
        # Engineer features
        self.logger.info("Engineering features")
        feature_config = self.config.get('data.features', {})
        engineered_data = self.feature_engineer.engineer_all_features(clean_data, feature_config)
        
        # Create target variable (next day's close price)
        engineered_data = self.feature_engineer.create_target_variable(
            engineered_data, target_type='next_close', horizon=1
        )
        
        # Remove rows with NaN values (common after feature engineering)
        final_data = engineered_data.dropna()
        
        self.logger.info(f"Data preparation completed. Final dataset: {len(final_data)} samples, {len(final_data.columns)} features")
        
        return final_data
    
    def prepare_model_data(self, data: pd.DataFrame) -> tuple:
        """
        Prepare data for model training.
        
        Args:
            data: Engineered DataFrame
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        # Separate features and target
        feature_columns = self.feature_engineer.feature_columns
        target_column = 'target'
        
        # Ensure we have the required columns
        available_features = [col for col in feature_columns if col in data.columns]
        if not available_features:
            raise ValueError("No engineered features found in data")
        
        X = data[available_features].values
        y = data[target_column].values
        
        # Create time-based splits
        train_ratio = self.config.get('data.train_ratio', 0.7)
        val_ratio = self.config.get('data.validation_ratio', 0.15)
        
        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        self.logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, available_features
    
    def train_models(self, X_train, X_val, y_train, y_val, feature_names):
        """Train all models in the ensemble."""
        self.logger.info("Starting model training")
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name} model")
                
                # Set hyperparameters from config
                model_config = self.config.get(f'models.{model_name}', {})
                if model_config:
                    model.set_hyperparameters(model_config)
                
                # Train model
                model.train(X_train, y_train, X_val, y_val, feature_names)
                trained_models[model_name] = model
                
                self.logger.info(f"{model_name} training completed")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        if not trained_models:
            raise RuntimeError("No models were successfully trained")
        
        self.logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all trained models."""
        self.logger.info("Evaluating models")
        
        # Evaluate individual models
        comparison_df = self.evaluator.compare_models(models, X_test, y_test)
        
        self.logger.info("Model evaluation results:")
        self.logger.info(f"\n{comparison_df.to_string()}")
        
        return comparison_df
    
    def build_ensemble(self, models, comparison_df):
        """Build the ensemble from trained models."""
        self.logger.info("Building ensemble")
        
        # Add models to ensemble
        for _, row in comparison_df.iterrows():
            model_name = row['model_name']
            if model_name in models:
                performance_metrics = {
                    'rmse': row['rmse'],
                    'mae': row['mae'],
                    'r2_score': row['r2_score'],
                    'directional_accuracy': row['directional_accuracy']
                }
                self.ensemble.add_model(model_name, models[model_name], performance_metrics)
        
        # Calculate ensemble weights
        performance_dict = {}
        for _, row in comparison_df.iterrows():
            model_name = row['model_name']
            performance_dict[model_name] = {
                'rmse': row['rmse'],
                'mae': row['mae'],
                'r2_score': row['r2_score'],
                'directional_accuracy': row['directional_accuracy']
            }
        
        # Set weighting method from config
        weighting_method = self.config.get('ensemble.weighting_method', 'inverse_error')
        self.ensemble.weighting_method = weighting_method
        
        weights = self.ensemble.calculate_weights(performance_dict)
        
        self.logger.info(f"Ensemble built with weights: {weights}")
        
        return self.ensemble
    
    def generate_predictions(self, ensemble, X_test, n_future_days: int = 7):
        """Generate ensemble predictions."""
        self.logger.info(f"Generating predictions for {n_future_days} future days")
        
        # Get ensemble predictions for test set
        ensemble_results = ensemble.predict_ensemble(X_test, return_confidence=True)
        
        # For future predictions, we would need to implement recursive prediction
        # This is a simplified version that predicts on the test set
        
        predictions_df = pd.DataFrame({
            'ensemble_prediction': ensemble_results['ensemble_prediction'],
            'confidence_lower': ensemble_results.get('confidence_lower', np.nan),
            'confidence_upper': ensemble_results.get('confidence_upper', np.nan)
        })
        
        return predictions_df, ensemble_results
    
    def run_full_pipeline(self, symbol: str = None, years_back: int = None):
        """Run the complete prediction pipeline."""
        try:
            # Use default symbol if not provided
            if symbol is None:
                symbol = self.config.get('data.default_symbol', 'AAPL')
            
            self.logger.info(f"Starting full pipeline for {symbol}")
            
            # Step 1: Fetch and prepare data
            data = self.fetch_and_prepare_data(symbol, years_back)
            
            # Step 2: Prepare model data
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.prepare_model_data(data)
            
            # Step 3: Train models
            trained_models = self.train_models(X_train, X_val, y_train, y_val, feature_names)
            
            # Step 4: Evaluate models
            comparison_df = self.evaluate_models(trained_models, X_test, y_test)
            
            # Step 5: Build ensemble
            ensemble = self.build_ensemble(trained_models, comparison_df)
            
            # Step 6: Generate predictions
            predictions_df, ensemble_results = self.generate_predictions(ensemble, X_test)
            
            # Step 7: Calculate ensemble performance
            ensemble_pred = ensemble_results['ensemble_prediction']
            ensemble_metrics = self.evaluator.calculate_comprehensive_metrics(y_test, ensemble_pred, 'ensemble')
            
            self.logger.info(f"Ensemble performance: RMSE={ensemble_metrics['rmse']:.4f}, "
                           f"MAE={ensemble_metrics['mae']:.4f}, "
                           f"Directional Accuracy={ensemble_metrics['directional_accuracy']:.2f}%")
            
            # Return results
            results = {
                'symbol': symbol,
                'data_shape': data.shape,
                'trained_models': list(trained_models.keys()),
                'model_comparison': comparison_df,
                'ensemble_weights': ensemble.weights,
                'ensemble_metrics': ensemble_metrics,
                'predictions': predictions_df,
                'test_performance': {
                    'n_test_samples': len(y_test),
                    'actual_values': y_test,
                    'ensemble_predictions': ensemble_pred
                }
            }
            
            self.logger.info("Full pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stock Price Ensemble Predictor')
    parser.add_argument('--symbol', type=str, default='AAPL', 
                       help='Stock symbol to predict (default: AAPL)')
    parser.add_argument('--years', type=int, default=5,
                       help='Number of years of historical data (default: 5)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize application
        app = StockPredictorApp(args.config)
        
        # Run pipeline
        results = app.run_full_pipeline(args.symbol, args.years)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"STOCK PRICE PREDICTION RESULTS FOR {args.symbol}")
        print(f"{'='*60}")
        print(f"Data samples: {results['data_shape'][0]}")
        print(f"Features: {results['data_shape'][1]}")
        print(f"Trained models: {', '.join(results['trained_models'])}")
        print(f"\nModel Comparison:")
        print(results['model_comparison'].to_string(index=False))
        print(f"\nEnsemble Weights:")
        for model, weight in results['ensemble_weights'].items():
            print(f"  {model}: {weight:.4f}")
        print(f"\nEnsemble Performance:")
        metrics = results['ensemble_metrics']
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        
        print(f"\nPrediction pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
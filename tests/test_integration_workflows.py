"""
Integration tests for end-to-end workflows.
Tests complete training and prediction pipelines, multi-asset workflows,
backtesting, and monitoring system integration.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import warnings

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_timeseries.models.advanced_transformer import AdvancedTransformer
from enhanced_timeseries.models.lstm_model import EnhancedBidirectionalLSTM as AdvancedLSTM
from enhanced_timeseries.models.cnn_lstm_hybrid import CNNLSTMHybrid
from enhanced_timeseries.ensemble.ensemble_framework import EnsembleFramework
from enhanced_timeseries.features.technical_indicators import TechnicalIndicators
from enhanced_timeseries.features.microstructure_features import MicrostructureFeatures
from enhanced_timeseries.backtesting.walk_forward import WalkForwardAnalyzer
from enhanced_timeseries.monitoring.performance_monitor import PerformanceMonitor
from enhanced_timeseries.multi_asset.data_coordinator import MultiAssetDataCoordinator
from enhanced_timeseries.optimization.bayesian_optimizer import BayesianOptimizer

warnings.filterwarnings('ignore')


class TestEndToEndTrainingPipeline(unittest.TestCase):
    """Test complete training pipeline from data to trained models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic time series data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        n_samples = len(dates)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, n_samples)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples)
        })
        
        # Feature engineering
        self.feature_engineer = TechnicalIndicators()
        self.microstructure_engineer = MicrostructureFeatures()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline from raw data to trained models."""
        # 1. Feature engineering
        features = self.feature_engineer.calculate_all_indicators(self.data)
        microstructure_features = self.microstructure_engineer.calculate_features(self.data)
        
        # Combine features
        all_features = pd.concat([features, microstructure_features], axis=1)
        all_features = all_features.dropna()
        
        # 2. Prepare training data
        sequence_length = 60
        target_column = 'close'
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(all_features)):
            X.append(all_features.iloc[i-sequence_length:i].values)
            y.append(all_features[target_column].iloc[i])
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 3. Initialize models
        input_dim = X.shape[2]
        
        transformer = AdvancedTransformer(
            input_dim=input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            seq_len=sequence_length
        )
        
        lstm = AdvancedLSTM(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=3,
            bidirectional=True,
            attention=True
        )
        
        cnn_lstm = CNNLSTMHybrid(
            input_dim=input_dim,
            cnn_channels=[32, 64, 128],
            lstm_hidden=128,
            seq_len=sequence_length
        )
        
        # 4. Train models
        models = {
            'transformer': transformer,
            'lstm': lstm,
            'cnn_lstm': cnn_lstm
        }
        
        trained_models = {}
        for name, model in models.items():
            # Simple training loop
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(5):  # Short training for testing
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
            
            trained_models[name] = model
        
        # 5. Verify models are trained
        for name, model in trained_models.items():
            model.eval()
            with torch.no_grad():
                predictions = model(X_test[:10])  # Test on small batch
                self.assertEqual(predictions.shape, (10, 1))
                self.assertFalse(torch.isnan(predictions).any())
    
    def test_ensemble_training_pipeline(self):
        """Test ensemble training and prediction pipeline."""
        # Create and train individual models (simplified)
        input_dim = 50
        sequence_length = 60
        
        models = {
            'transformer': AdvancedTransformer(input_dim, d_model=64, nhead=4, num_layers=2),
            'lstm': AdvancedLSTM(input_dim, hidden_dim=64, num_layers=2),
            'cnn_lstm': CNNLSTMHybrid(input_dim, cnn_channels=[16, 32], lstm_hidden=64)
        }
        
        # Create ensemble
        ensemble = EnsembleFramework(
            models=models,
            weighting_method='performance_based',
            uncertainty_method='ensemble_variance'
        )
        
        # Create synthetic training data
        X_train = torch.randn(100, sequence_length, input_dim)
        y_train = torch.randn(100, 1)
        
        # Train ensemble
        ensemble.train_models(X_train, y_train, epochs=3)
        
        # Test ensemble prediction
        X_test = torch.randn(10, sequence_length, input_dim)
        predictions, uncertainties = ensemble.predict_with_uncertainty(X_test)
        
        self.assertEqual(predictions.shape, (10, 1))
        self.assertEqual(uncertainties.shape, (10, 1))
        self.assertFalse(torch.isnan(predictions).any())
        self.assertFalse(torch.isnan(uncertainties).any())


class TestMultiAssetWorkflow(unittest.TestCase):
    """Test multi-asset prediction workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic multi-asset data
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        self.assets_data = {}
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for symbol in symbols:
            returns = np.random.normal(0.001, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            self.assets_data[symbol] = pd.DataFrame({
                'date': dates,
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                'close': prices,
                'volume': np.random.lognormal(10, 1, len(dates))
            })
    
    def test_multi_asset_data_coordination(self):
        """Test multi-asset data coordination and synchronization."""
        coordinator = MultiAssetDataCoordinator(
            max_assets=50,
            batch_size=10,
            memory_efficient=True
        )
        
        # Add assets to coordinator
        for symbol, data in self.assets_data.items():
            coordinator.add_asset(symbol, data)
        
        # Test data synchronization
        synchronized_data = coordinator.get_synchronized_data()
        
        self.assertIsInstance(synchronized_data, dict)
        self.assertEqual(len(synchronized_data), len(self.assets_data))
        
        # Check that all assets have the same date range
        date_ranges = [len(data) for data in synchronized_data.values()]
        self.assertEqual(len(set(date_ranges)), 1)
    
    def test_cross_asset_feature_generation(self):
        """Test cross-asset feature generation and relationship modeling."""
        from enhanced_timeseries.features.cross_asset_features import CrossAssetFeatures
        
        cross_asset_engine = CrossAssetFeatures(
            correlation_window=30,
            sector_classification=None
        )
        
        # Generate cross-asset features
        features = cross_asset_engine.calculate_features(self.assets_data)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(len(features), len(self.assets_data))
        
        # Check that features contain cross-asset information
        for symbol, asset_features in features.items():
            self.assertIsInstance(asset_features, pd.DataFrame)
            self.assertGreater(len(asset_features.columns), 0)
    
    def test_portfolio_prediction_workflow(self):
        """Test portfolio-level prediction workflow."""
        from enhanced_timeseries.multi_asset.portfolio_prediction import PortfolioPredictor
        
        predictor = PortfolioPredictor(
            correlation_threshold=0.7,
            max_assets=10,
            confidence_threshold=0.6
        )
        
        # Mock trained models
        mock_models = {
            'transformer': Mock(),
            'lstm': Mock(),
            'cnn_lstm': Mock()
        }
        
        # Mock predictions
        for model in mock_models.values():
            model.predict.return_value = torch.randn(10, 1)
            model.predict_with_uncertainty.return_value = (torch.randn(10, 1), torch.randn(10, 1))
        
        # Test portfolio prediction
        portfolio_predictions = predictor.predict_portfolio(
            assets_data=self.assets_data,
            models=mock_models,
            confidence_threshold=0.6
        )
        
        self.assertIsInstance(portfolio_predictions, dict)
        self.assertIn('predictions', portfolio_predictions)
        self.assertIn('confidence_scores', portfolio_predictions)
        self.assertIn('asset_rankings', portfolio_predictions)


class TestBacktestingWorkflow(unittest.TestCase):
    """Test backtesting and performance analysis workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic data for backtesting
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, len(dates))
        })
    
    def test_walk_forward_analysis_workflow(self):
        """Test complete walk-forward analysis workflow."""
        walk_forward = WalkForwardAnalyzer(
            training_window_days=252,  # 1 year
            testing_window_days=63,    # 3 months
            retraining_frequency_days=21,  # Monthly retraining
            min_training_size=100
        )
        
        # Mock model factory
        def create_model():
            return Mock()
        
        # Mock training function
        def train_model(model, X_train, y_train):
            # Simulate training
            model.train.return_value = None
            model.predict.return_value = torch.randn(len(y_train), 1)
            return model
        
        # Mock prediction function
        def predict(model, X_test):
            return torch.randn(len(X_test), 1)
        
        # Run walk-forward analysis
        results = walk_forward.run_analysis(
            data=self.data,
            model_factory=create_model,
            train_function=train_model,
            predict_function=predict,
            target_column='close',
            feature_columns=['open', 'high', 'low', 'volume']
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('performance_metrics', results)
        self.assertIn('predictions', results)
        self.assertIn('model_performance', results)
    
    def test_regime_analysis_workflow(self):
        """Test market regime analysis workflow."""
        from enhanced_timeseries.backtesting.regime_analysis import RegimeAnalyzer
        
        regime_analyzer = RegimeAnalyzer(
            volatility_window=30,
            regime_threshold=0.02,
            min_regime_duration=10
        )
        
        # Analyze market regimes
        regimes = regime_analyzer.analyze_regimes(self.data)
        
        self.assertIsInstance(regimes, pd.DataFrame)
        self.assertIn('regime', regimes.columns)
        self.assertIn('volatility', regimes.columns)
        
        # Test regime-specific performance analysis
        mock_predictions = pd.DataFrame({
            'date': self.data['date'],
            'predicted_return': np.random.normal(0.001, 0.02, len(self.data)),
            'actual_return': self.data['close'].pct_change()
        })
        
        regime_performance = regime_analyzer.analyze_regime_performance(
            predictions=mock_predictions,
            regimes=regimes
        )
        
        self.assertIsInstance(regime_performance, dict)
        self.assertIn('bull_market', regime_performance)
        self.assertIn('bear_market', regime_performance)
        self.assertIn('sideways_market', regime_performance)


class TestMonitoringWorkflow(unittest.TestCase):
    """Test monitoring and alerting system workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(
            accuracy_threshold=0.7,
            alert_window_minutes=30,
            performance_window_days=30
        )
    
    def test_performance_monitoring_workflow(self):
        """Test real-time performance monitoring workflow."""
        # Mock model predictions and actual values
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        predictions = pd.DataFrame({
            'date': dates,
            'predicted_value': np.random.normal(100, 5, len(dates)),
            'confidence': np.random.uniform(0.5, 0.9, len(dates))
        })
        
        actual_values = pd.DataFrame({
            'date': dates,
            'actual_value': np.random.normal(100, 5, len(dates))
        })
        
        # Update monitor with predictions and actual values
        for i in range(len(dates)):
            self.monitor.update_prediction(
                timestamp=dates[i],
                predicted_value=predictions['predicted_value'].iloc[i],
                actual_value=actual_values['actual_value'].iloc[i],
                confidence=predictions['confidence'].iloc[i]
            )
        
        # Get performance metrics
        metrics = self.monitor.get_performance_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('confidence_correlation', metrics)
    
    def test_alerting_system_workflow(self):
        """Test alerting system workflow."""
        from enhanced_timeseries.monitoring.alerting_system import AlertingSystem
        
        alerting_system = AlertingSystem(
            email_config=None,  # Disable email for testing
            webhook_url=None,   # Disable webhooks for testing
            alert_cooldown_minutes=15
        )
        
        # Add alert rules
        alerting_system.add_alert_rule(
            rule_id="accuracy_drop",
            name="Accuracy Drop Alert",
            alert_type="performance_degradation",
            severity="high",
            metric_name="accuracy",
            threshold_value=0.6,
            comparison_operator="<",
            time_window_minutes=60
        )
        
        # Simulate performance degradation
        for i in range(10):
            alerting_system.check_metrics({
                'accuracy': 0.5,  # Below threshold
                'mae': 0.1,
                'rmse': 0.15
            })
        
        # Check if alerts were generated
        alerts = alerting_system.get_recent_alerts()
        self.assertGreater(len(alerts), 0)
        
        # Test alert acknowledgment
        if alerts:
            alert_id = alerts[0].alert_id
            alerting_system.acknowledge_alert(alert_id)
            
            acknowledged_alerts = alerting_system.get_acknowledged_alerts()
            self.assertIn(alert_id, [alert.alert_id for alert in acknowledged_alerts])


class TestHyperparameterOptimizationWorkflow(unittest.TestCase):
    """Test hyperparameter optimization workflows."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = BayesianOptimizer(
            n_trials=10,
            n_jobs=1,  # Single job for testing
            random_state=42
        )
    
    def test_optimization_workflow(self):
        """Test complete hyperparameter optimization workflow."""
        # Define search space
        search_space = {
            'learning_rate': (0.0001, 0.01, 'log-uniform'),
            'hidden_dim': (32, 256, 'integer'),
            'num_layers': (1, 4, 'integer'),
            'dropout': (0.1, 0.5, 'uniform')
        }
        
        # Mock objective function
        def objective_function(trial):
            # Simulate training and evaluation
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
                'num_layers': trial.suggest_int('num_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5)
            }
            
            # Simulate training time and performance
            import time
            time.sleep(0.1)  # Simulate training time
            
            # Return simulated performance (higher is better)
            performance = 0.8 + 0.1 * np.random.random() - 0.05 * params['dropout']
            return performance
        
        # Run optimization
        best_params, best_score = self.optimizer.optimize(
            objective_function=objective_function,
            search_space=search_space,
            n_trials=5
        )
        
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_score, float)
        self.assertGreater(best_score, 0)
        
        # Check that all parameters are within bounds
        self.assertGreaterEqual(best_params['learning_rate'], 0.0001)
        self.assertLessEqual(best_params['learning_rate'], 0.01)
        self.assertGreaterEqual(best_params['hidden_dim'], 32)
        self.assertLessEqual(best_params['hidden_dim'], 256)
    
    def test_multi_objective_optimization(self):
        """Test multi-objective optimization workflow."""
        # Mock multi-objective function
        def multi_objective_function(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
                'hidden_dim': trial.suggest_int('hidden_dim', 32, 256)
            }
            
            # Simulate multiple objectives (accuracy and training time)
            accuracy = 0.8 + 0.1 * np.random.random()
            training_time = 1.0 + 0.5 * params['hidden_dim'] / 256  # Larger models take longer
            
            return accuracy, -training_time  # Negative because we minimize
        
        # Run multi-objective optimization
        results = self.optimizer.optimize_multi_objective(
            objective_function=multi_objective_function,
            n_trials=5,
            n_objectives=2
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('pareto_front', results)
        self.assertIn('best_solutions', results)


if __name__ == '__main__':
    unittest.main()

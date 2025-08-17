# Enhanced Time Series Prediction System - API Documentation

## Overview

The Enhanced Time Series Prediction System is a comprehensive financial market prediction platform that combines multiple advanced neural network architectures, sophisticated feature engineering, and robust backtesting capabilities. This document provides detailed API documentation for all public interfaces.

## Table of Contents

1. [Model Architectures](#model-architectures)
2. [Feature Engineering](#feature-engineering)
3. [Ensemble Framework](#ensemble-framework)
4. [Backtesting](#backtesting)
5. [Monitoring](#monitoring)
6. [Multi-Asset Support](#multi-asset-support)
7. [Optimization](#optimization)
8. [Uncertainty Quantification](#uncertainty-quantification)

## Model Architectures

### EnhancedTransformer

Advanced Transformer model with multi-scale attention for time series forecasting.

```python
from enhanced_timeseries.models.advanced_transformer import EnhancedTransformer

model = EnhancedTransformer(
    input_dim=100,           # Number of input features
    d_model=256,             # Model dimension
    nhead=16,                # Number of attention heads
    num_layers=8,            # Number of transformer layers
    seq_len=60,              # Sequence length
    dropout=0.1,             # Dropout rate
    use_positional_encoding=True  # Use learnable positional encoding
)
```

**Key Features:**
- Multi-scale attention mechanism
- Learnable positional embeddings
- Residual connections with layer normalization
- Adaptive dropout for uncertainty estimation

**Methods:**
- `forward(x)`: Forward pass through the model
- `predict(x)`: Make predictions
- `predict_with_uncertainty(x, n_samples=100)`: Predictions with uncertainty estimates

### AdvancedLSTM

Bidirectional LSTM with attention mechanism for time series prediction.

```python
from enhanced_timeseries.models.lstm_model import AdvancedLSTM

model = AdvancedLSTM(
    input_dim=100,           # Number of input features
    hidden_dim=256,          # Hidden dimension
    num_layers=4,            # Number of LSTM layers
    dropout=0.2,             # Dropout rate
    bidirectional=True,      # Use bidirectional LSTM
    attention=True           # Use attention mechanism
)
```

**Key Features:**
- Bidirectional LSTM architecture
- Attention-based feature weighting
- Skip connections between layers
- Variational dropout for uncertainty quantification

### CNNLSTMHybrid

CNN-LSTM hybrid model for multi-scale pattern recognition.

```python
from enhanced_timeseries.models.cnn_lstm_hybrid import CNNLSTMHybrid

model = CNNLSTMHybrid(
    input_dim=100,           # Number of input features
    cnn_channels=[64, 128, 256],  # CNN channel sizes
    lstm_hidden=256,         # LSTM hidden dimension
    seq_len=60,              # Sequence length
    dropout=0.2              # Dropout rate
)
```

**Key Features:**
- 1D CNN layers for local pattern extraction
- LSTM layers for temporal dependency modeling
- Feature fusion layer combining CNN and LSTM outputs
- Multi-scale temporal convolutions

## Feature Engineering

### TechnicalIndicators

Comprehensive technical indicator calculations.

```python
from enhanced_timeseries.features.technical_indicators import TechnicalIndicators

engineer = TechnicalIndicators()

# Calculate all indicators
features = engineer.calculate_all_indicators(data)

# Calculate specific indicators
rsi = engineer.calculate_rsi(data['close'], period=14)
macd = engineer.calculate_macd(data['close'])
bollinger = engineer.calculate_bollinger_bands(data['close'])
```

**Available Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- Williams %R
- ATR (Average True Range)
- CCI (Commodity Channel Index)
- And 18+ more indicators

### MicrostructureFeatures

Market microstructure feature extraction.

```python
from enhanced_timeseries.features.microstructure_features import MicrostructureFeatures

engineer = MicrostructureFeatures()

# Calculate microstructure features
features = engineer.calculate_features(data)

# Calculate specific features
spread_proxy = engineer.calculate_bid_ask_spread_proxy(data)
volume_profile = engineer.calculate_volume_profile(data)
vwap = engineer.calculate_vwap(data)
```

**Available Features:**
- Bid-ask spread proxies
- Volume profile analysis
- VWAP calculations
- Order flow imbalance indicators
- Price impact measures
- Liquidity proxies

### CrossAssetFeatures

Cross-asset relationship modeling.

```python
from enhanced_timeseries.features.cross_asset_features import CrossAssetFeatures

engineer = CrossAssetFeatures(
    correlation_window=30,
    sector_classification=sector_map
)

# Calculate cross-asset features
features = engineer.calculate_features(assets_data)
```

**Available Features:**
- Correlation-based features
- Sector momentum indicators
- Market-wide factor analysis
- Principal component features

## Ensemble Framework

### EnsembleFramework

Ensemble prediction system with dynamic weighting.

```python
from enhanced_timeseries.ensemble.ensemble_framework import EnsembleFramework

# Create ensemble with multiple models
models = {
    'transformer': transformer_model,
    'lstm': lstm_model,
    'cnn_lstm': cnn_lstm_model
}

ensemble = EnsembleFramework(
    models=models,
    weighting_method='performance_based',  # 'equal', 'performance_based', 'uncertainty_based'
    uncertainty_method='ensemble_variance'  # 'dropout', 'ensemble_variance', 'monte_carlo'
)

# Train ensemble
ensemble.train_models(X_train, y_train, epochs=100)

# Make predictions
predictions = ensemble.predict(X_test)
predictions, uncertainties = ensemble.predict_with_uncertainty(X_test)
```

**Weighting Methods:**
- `equal`: Equal weights for all models
- `performance_based`: Weights based on historical performance
- `uncertainty_based`: Weights based on prediction uncertainty

**Uncertainty Methods:**
- `dropout`: Monte Carlo dropout
- `ensemble_variance`: Variance across ensemble predictions
- `monte_carlo`: Monte Carlo sampling

## Backtesting

### WalkForwardAnalyzer

Walk-forward analysis for robust model validation.

```python
from enhanced_timeseries.backtesting.walk_forward import WalkForwardAnalyzer

analyzer = WalkForwardAnalyzer(
    training_window_days=252,      # 1 year training window
    testing_window_days=63,        # 3 months testing window
    retraining_frequency_days=21,  # Monthly retraining
    min_training_size=100          # Minimum training samples
)

# Run walk-forward analysis
results = analyzer.run_analysis(
    data=data,
    model_factory=create_model,
    train_function=train_model,
    predict_function=predict,
    target_column='close',
    feature_columns=['open', 'high', 'low', 'volume']
)
```

**Results include:**
- Performance metrics (accuracy, MAE, RMSE)
- Predictions for each testing window
- Model performance tracking
- Statistical significance tests

### RegimeAnalyzer

Market regime detection and analysis.

```python
from enhanced_timeseries.backtesting.regime_analysis import RegimeAnalyzer

analyzer = RegimeAnalyzer(
    volatility_window=30,
    regime_threshold=0.02,
    min_regime_duration=10
)

# Analyze market regimes
regimes = analyzer.analyze_regimes(data)

# Analyze regime-specific performance
regime_performance = analyzer.analyze_regime_performance(
    predictions=predictions,
    regimes=regimes
)
```

**Regime Types:**
- Bull market
- Bear market
- Sideways market
- High volatility
- Low volatility

## Monitoring

### PerformanceMonitor

Real-time performance monitoring.

```python
from enhanced_timeseries.monitoring.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(
    accuracy_threshold=0.7,
    alert_window_minutes=30,
    performance_window_days=30
)

# Update with new predictions
monitor.update_prediction(
    timestamp=datetime.now(),
    predicted_value=100.5,
    actual_value=100.2,
    confidence=0.85
)

# Get performance metrics
metrics = monitor.get_performance_metrics()
```

**Available Metrics:**
- Accuracy
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Confidence correlation
- Performance trends

### AlertingSystem

Automated alerting and anomaly detection.

```python
from enhanced_timeseries.monitoring.alerting_system import AlertingSystem

alerting = AlertingSystem(
    email_config=email_config,
    webhook_url=webhook_url,
    alert_cooldown_minutes=15
)

# Add alert rules
alerting.add_alert_rule(
    rule_id="accuracy_drop",
    name="Accuracy Drop Alert",
    alert_type="performance_degradation",
    severity="high",
    metric_name="accuracy",
    threshold_value=0.6,
    comparison_operator="<",
    time_window_minutes=60
)

# Check metrics and generate alerts
alerting.check_metrics({
    'accuracy': 0.55,
    'mae': 0.1,
    'rmse': 0.15
})
```

**Alert Types:**
- Performance degradation
- Data quality issues
- System health problems
- Model failures
- Anomaly detection

## Multi-Asset Support

### MultiAssetDataCoordinator

Multi-asset data coordination and synchronization.

```python
from enhanced_timeseries.multi_asset.data_coordinator import MultiAssetDataCoordinator

coordinator = MultiAssetDataCoordinator(
    max_assets=50,
    batch_size=10,
    memory_efficient=True
)

# Add assets
coordinator.add_asset('AAPL', aapl_data)
coordinator.add_asset('GOOGL', googl_data)
coordinator.add_asset('MSFT', msft_data)

# Get synchronized data
synchronized_data = coordinator.get_synchronized_data()

# Get batch for processing
batch = coordinator.get_batch(['AAPL', 'GOOGL'])
```

### PortfolioPredictor

Portfolio-level prediction and asset ranking.

```python
from enhanced_timeseries.multi_asset.portfolio_prediction import PortfolioPredictor

predictor = PortfolioPredictor(
    correlation_threshold=0.7,
    max_assets=10,
    confidence_threshold=0.6
)

# Predict portfolio performance
portfolio_predictions = predictor.predict_portfolio(
    assets_data=assets_data,
    models=trained_models,
    confidence_threshold=0.6
)
```

**Output includes:**
- Portfolio-level predictions
- Asset rankings
- Confidence scores
- Correlation analysis

## Optimization

### BayesianOptimizer

Hyperparameter optimization using Bayesian methods.

```python
from enhanced_timeseries.optimization.bayesian_optimizer import BayesianOptimizer

optimizer = BayesianOptimizer(
    n_trials=100,
    n_jobs=4,
    random_state=42
)

# Define search space
search_space = {
    'learning_rate': (0.0001, 0.01, 'log-uniform'),
    'hidden_dim': (32, 256, 'integer'),
    'num_layers': (1, 4, 'integer'),
    'dropout': (0.1, 0.5, 'uniform')
}

# Define objective function
def objective_function(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 256),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5)
    }
    
    # Train model and return performance
    model = create_model(params)
    performance = train_and_evaluate(model)
    return performance

# Run optimization
best_params, best_score = optimizer.optimize(
    objective_function=objective_function,
    search_space=search_space,
    n_trials=50
)
```

### ConfigManager

Configuration management and versioning.

```python
from enhanced_timeseries.optimization.config_manager import ConfigManager

manager = ConfigManager(
    config_dir='configs/',
    version_control=True
)

# Save configuration
config = {
    'model_type': 'transformer',
    'hyperparameters': best_params,
    'performance': best_score
}

manager.save_config('transformer_v1', config)

# Load configuration
loaded_config = manager.load_config('transformer_v1')

# Compare configurations
comparison = manager.compare_configs('transformer_v1', 'transformer_v2')
```

## Uncertainty Quantification

### MonteCarloDropout

Monte Carlo dropout for uncertainty estimation.

```python
from enhanced_timeseries.uncertainty.monte_carlo import MonteCarloDropout

mc_dropout = MonteCarloDropout(
    model=model,
    n_samples=100,
    dropout_rate=0.1
)

# Get predictions with uncertainty
predictions, uncertainties = mc_dropout.predict_with_uncertainty(X_test)

# Get confidence intervals
confidence_intervals = mc_dropout.get_confidence_intervals(X_test, confidence_level=0.95)
```

### ConfidenceScoring

Prediction confidence scoring and risk assessment.

```python
from enhanced_timeseries.uncertainty.confidence_scoring import ConfidenceScoring

scorer = ConfidenceScoring(
    ensemble_models=models,
    uncertainty_threshold=0.1,
    confidence_calibration=True
)

# Calculate confidence scores
confidence_scores = scorer.calculate_confidence(predictions, uncertainties)

# Get risk-adjusted recommendations
recommendations = scorer.get_risk_adjusted_recommendations(
    predictions=predictions,
    confidence_scores=confidence_scores,
    position_size=10000
)
```

## Usage Examples

### Complete Training Pipeline

```python
import torch
from enhanced_timeseries.models.advanced_transformer import EnhancedTransformer
from enhanced_timeseries.features.technical_indicators import TechnicalIndicators
from enhanced_timeseries.ensemble.ensemble_framework import EnsembleFramework

# 1. Feature engineering
engineer = TechnicalIndicators()
features = engineer.calculate_all_indicators(data)

# 2. Prepare data
X, y = prepare_sequences(features, target_column='close', sequence_length=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Create models
models = {
    'transformer': EnhancedTransformer(input_dim=X.shape[2], d_model=128, nhead=8),
    'lstm': AdvancedLSTM(input_dim=X.shape[2], hidden_dim=128, num_layers=3),
    'cnn_lstm': CNNLSTMHybrid(input_dim=X.shape[2], cnn_channels=[32, 64], lstm_hidden=128)
}

# 4. Create ensemble
ensemble = EnsembleFramework(models=models, weighting_method='performance_based')

# 5. Train ensemble
ensemble.train_models(X_train, y_train, epochs=100)

# 6. Make predictions
predictions, uncertainties = ensemble.predict_with_uncertainty(X_test)
```

### Backtesting Example

```python
from enhanced_timeseries.backtesting.walk_forward import WalkForwardAnalyzer

# Define model factory
def create_model():
    return EnhancedTransformer(input_dim=50, d_model=128, nhead=8)

# Define training function
def train_model(model, X_train, y_train):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    return model

# Define prediction function
def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        return model(X_test)

# Run walk-forward analysis
analyzer = WalkForwardAnalyzer(
    training_window_days=252,
    testing_window_days=63,
    retraining_frequency_days=21
)

results = analyzer.run_analysis(
    data=data,
    model_factory=create_model,
    train_function=train_model,
    predict_function=predict,
    target_column='close',
    feature_columns=['open', 'high', 'low', 'volume']
)

print(f"Overall Accuracy: {results['performance_metrics']['accuracy']:.3f}")
print(f"MAE: {results['performance_metrics']['mae']:.3f}")
print(f"RMSE: {results['performance_metrics']['rmse']:.3f}")
```

### Monitoring Example

```python
from enhanced_timeseries.monitoring.performance_monitor import PerformanceMonitor
from enhanced_timeseries.monitoring.alerting_system import AlertingSystem

# Set up monitoring
monitor = PerformanceMonitor(accuracy_threshold=0.7)
alerting = AlertingSystem()

# Add alert rule
alerting.add_alert_rule(
    rule_id="accuracy_drop",
    name="Accuracy Drop Alert",
    alert_type="performance_degradation",
    severity="high",
    metric_name="accuracy",
    threshold_value=0.6,
    comparison_operator="<",
    time_window_minutes=60
)

# In production loop
for timestamp, prediction, actual in prediction_stream:
    # Update monitor
    monitor.update_prediction(
        timestamp=timestamp,
        predicted_value=prediction,
        actual_value=actual,
        confidence=0.85
    )
    
    # Check for alerts
    metrics = monitor.get_performance_metrics()
    alerting.check_metrics(metrics)
    
    # Get recent alerts
    alerts = alerting.get_recent_alerts()
    for alert in alerts:
        print(f"Alert: {alert.name} - {alert.severity}")
```

## Configuration

### Model Configuration

```yaml
# config/model_config.yaml
models:
  transformer:
    input_dim: 100
    d_model: 256
    nhead: 16
    num_layers: 8
    seq_len: 60
    dropout: 0.1
    
  lstm:
    input_dim: 100
    hidden_dim: 256
    num_layers: 4
    bidirectional: true
    attention: true
    dropout: 0.2
    
  cnn_lstm:
    input_dim: 100
    cnn_channels: [32, 64, 128]
    lstm_hidden: 256
    seq_len: 60
    dropout: 0.2

ensemble:
  weighting_method: "performance_based"
  uncertainty_method: "ensemble_variance"
  performance_window: 30

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
```

### Feature Configuration

```yaml
# config/feature_config.yaml
technical_indicators:
  rsi_period: 14
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  bollinger_period: 20
  bollinger_std: 2
  
microstructure:
  volume_window: 20
  spread_proxy_method: "high_low"
  vwap_period: 14
  
cross_asset:
  correlation_window: 30
  sector_classification: "sectors.yaml"
  max_assets: 50
```

## Error Handling

The system provides comprehensive error handling with custom exceptions:

```python
from enhanced_timeseries.utils.exceptions import (
    ModelTrainingError,
    PredictionError,
    DataValidationError,
    ConfigurationError
)

try:
    predictions = model.predict(X_test)
except PredictionError as e:
    print(f"Prediction failed: {e}")
    # Handle prediction failure
except DataValidationError as e:
    print(f"Data validation failed: {e}")
    # Handle data issues
```

## Performance Considerations

### Memory Management

- Use `memory_efficient=True` in data coordinators for large datasets
- Process data in batches to avoid memory overflow
- Use `torch.cuda.empty_cache()` for GPU memory management

### Speed Optimization

- Use appropriate batch sizes for your hardware
- Enable mixed precision training with `torch.cuda.amp`
- Use data loaders with multiple workers for I/O optimization

### Scalability

- The system supports up to 50 assets simultaneously
- Use distributed training for large models
- Implement model parallelism for very large architectures

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Enable memory-efficient processing

2. **Slow Training**
   - Check GPU utilization
   - Optimize data loading
   - Use appropriate model sizes

3. **Poor Performance**
   - Check feature engineering
   - Validate data quality
   - Adjust hyperparameters

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set specific logger levels
logging.getLogger('enhanced_timeseries').setLevel(logging.DEBUG)
```

## Support and Contributing

For issues, questions, or contributions:

1. Check the troubleshooting section
2. Review the test suite for usage examples
3. Submit issues with detailed error messages
4. Follow the contribution guidelines

The system is designed to be modular and extensible. New models, features, and components can be easily added by following the established interfaces and patterns.

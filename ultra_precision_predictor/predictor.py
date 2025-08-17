"""Main ultra-precision predictor class."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

from .core.exceptions import FeatureEngineeringError
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class UltraPrecisionPredictor:
    """Main ultra-precision predictor class targeting <1% error rates."""
    
    def __init__(self):
        """Initialize ultra-precision predictor."""
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Ultra-Precision Predictor")
        
        # Initialize components (lazy loading to avoid import issues)
        self.feature_engineer = None
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Training state
        self.is_trained = False
        self.training_stats = {}
        self.feature_names = []
        
        self.logger.info("Ultra-Precision Predictor initialized")
    
    def _get_feature_engineer(self):
        """Lazy load feature engineer to avoid import issues."""
        if self.feature_engineer is None:
            from .feature_engineering.extreme_feature_engineer import ExtremeFeatureEngineer
            self.feature_engineer = ExtremeFeatureEngineer()
        return self.feature_engineer
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the ultra-precision predictor.
        
        Args:
            data: Historical stock data with OHLCV columns
            
        Returns:
            Dictionary containing training statistics and performance metrics
        """
        try:
            self.logger.info("Starting ultra-precision predictor training")
            
            # Validate input data
            if data.empty:
                raise FeatureEngineeringError("Training data is empty")
            
            # Generate features
            self.logger.info("Generating features...")
            feature_engineer = self._get_feature_engineer()
            features_df = feature_engineer.generate_features(data)
            
            # Prepare target (next period return)
            target = features_df['Close'].pct_change().shift(-1).dropna()
            
            # Align features with target
            features_df = features_df.iloc[:-1]  # Remove last row to align with target
            
            # Get feature columns (exclude original OHLCV)
            feature_cols = [col for col in features_df.columns 
                          if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            X = features_df[feature_cols].fillna(0)
            y = target
            
            # Ensure we have valid data
            if len(X) == 0 or len(y) == 0:
                raise FeatureEngineeringError("No valid training data after feature generation")
            
            self.logger.info(f"Training with {len(X)} samples and {len(feature_cols)} features")
            
            # Train model
            self.model.fit(X, y)
            self.feature_names = feature_cols
            
            # Training statistics
            training_stats = {
                'data_shape': data.shape,
                'feature_count': len(feature_cols),
                'training_samples': len(X),
                'status': 'completed'
            }
            
            self.is_trained = True
            self.training_stats = training_stats
            
            self.logger.info("Ultra-precision predictor training completed")
            return training_stats
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise FeatureEngineeringError(f"Training failed: {str(e)}") from e
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate ultra-precision predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of predictions (returns)
        """
        if not self.is_trained:
            raise FeatureEngineeringError("Model must be trained before making predictions")
        
        try:
            self.logger.info("Generating ultra-precision predictions")
            
            # Generate features
            feature_engineer = self._get_feature_engineer()
            features_df = feature_engineer.generate_features(data)
            
            # Get feature columns
            X = features_df[self.feature_names].fillna(0)
            
            # Make predictions
            predictions = self.model.predict(X)
            
            self.logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise FeatureEngineeringError(f"Prediction failed: {str(e)}") from e
    
    def validate(self, data: pd.DataFrame) -> Dict[str, float]:
        """Validate predictor performance.
        
        Args:
            data: Validation data
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            raise FeatureEngineeringError("Model must be trained before validation")
        
        try:
            self.logger.info("Starting validation")
            
            # Generate predictions
            predictions = self.predict(data)
            
            # Calculate actual returns
            actual_returns = data['Close'].pct_change().dropna()
            
            # Align predictions and actuals
            min_len = min(len(predictions), len(actual_returns))
            pred_aligned = predictions[:min_len]
            actual_aligned = actual_returns.iloc[:min_len].values
            
            # Calculate metrics
            mae = np.mean(np.abs(pred_aligned - actual_aligned))
            mse = np.mean((pred_aligned - actual_aligned) ** 2)
            
            # Directional accuracy
            pred_direction = np.sign(pred_aligned)
            actual_direction = np.sign(actual_aligned)
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            report = {
                'mae': mae,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'directional_accuracy': directional_accuracy
            }
            
            self.logger.info(f"Validation completed - MAE: {mae:.6f}")
            return report
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise FeatureEngineeringError(f"Validation failed: {str(e)}") from e
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise FeatureEngineeringError("Model must be trained to get feature importance")
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current ensemble model weights.
        
        Returns:
            Dictionary mapping model names to weights
        """
        if not self.is_trained:
            raise FeatureEngineeringError("Model must be trained to get model weights")
        
        return {'random_forest': 1.0}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise FeatureEngineeringError("Model must be trained before saving")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'training_stats': self.training_stats
            }, f)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model from file.
        
        Args:
            filepath: Path to load the model from
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.training_stats = data['training_stats']
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and status.
        
        Returns:
            Dictionary containing system information
        """
        return {
            'version': '1.0.0',
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'feature_count': len(self.feature_names),
            'target_error_rate': 1.0
        }
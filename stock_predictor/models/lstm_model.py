"""LSTM model implementation for stock price prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

from stock_predictor.models.base import BaseModel
from stock_predictor.utils.exceptions import ModelTrainingError, ModelPredictionError


class LSTMModel(BaseModel):
    """LSTM neural network model for time series forecasting."""
    
    def __init__(self, name: str = "lstm"):
        super().__init__(name)
        self.scaler = MinMaxScaler()
        self.sequence_length = 60
        
        # Default hyperparameters
        self.hyperparameters = {
            'sequence_length': 60,
            'units': [50, 50],
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'reduce_lr_factor': 0.5
        }
    
    def _build_model(self) -> Sequential:
        """Build LSTM neural network architecture."""
        model = Sequential()
        
        units = self.hyperparameters.get('units', [50, 50])
        dropout = self.hyperparameters.get('dropout', 0.2)
        
        # First LSTM layer
        model.add(LSTM(
            units[0],
            return_sequences=len(units) > 1,
            input_shape=(self.sequence_length, 1)
        ))
        model.add(Dropout(dropout))
        
        # Additional LSTM layers
        for i, unit_count in enumerate(units[1:], 1):
            return_sequences = i < len(units) - 1
            model.add(LSTM(unit_count, return_sequences=return_sequences))
            model.add(Dropout(dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.hyperparameters.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def _fit_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit LSTM model to training data."""
        try:
            # Create sequences for LSTM
            X_sequences, y_sequences = self.create_sequences(X_train, y_train)
            
            # Prepare validation data if provided
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
                validation_data = (X_val_seq, y_val_seq)
            
            # Set up callbacks
            callbacks = self._get_callbacks()
            
            # Train model
            history = self.model.fit(
                X_sequences, y_sequences,
                epochs=self.hyperparameters.get('epochs', 100),
                batch_size=self.hyperparameters.get('batch_size', 32),
                validation_data=validation_data,
                validation_split=self.hyperparameters.get('validation_split', 0.2) if validation_data is None else 0,
                callbacks=callbacks,
                verbose=0
            )
            
            # Store training history
            self.training_history = {
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history.get('val_loss', [])],
                'mae': [float(x) for x in history.history.get('mae', [])],
                'val_mae': [float(x) for x in history.history.get('val_mae', [])],
                'epochs_trained': len(history.history['loss']),
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history.get('val_loss', [0])[-1]) if history.history.get('val_loss') else None
            }
            
            self.logger.info(f"LSTM training completed. Final loss: {self.training_history['final_loss']:.6f}")
            
        except Exception as e:
            raise ModelTrainingError(f"LSTM model fitting failed: {str(e)}")
    
    def _predict_model(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using trained LSTM model."""
        if self.model is None:
            raise ModelPredictionError("LSTM model not trained")
        
        try:
            # For LSTM, we need to handle the prediction differently
            # Since we need to return predictions for each sample in X_test
            
            # If X_test is smaller than sequence_length, we'll make a single prediction
            if len(X_test) < self.sequence_length:
                X_sequences = self._create_prediction_sequences(X_test)
                predictions = self.model.predict(X_sequences, verbose=0)
                # Repeat the prediction for each sample in X_test
                single_pred = predictions[0, 0] if predictions.ndim > 1 else predictions[0]
                return np.full(len(X_test), single_pred)
            
            # For larger datasets, we can make predictions for each possible sequence
            X_sequences = self._create_prediction_sequences(X_test)
            predictions = self.model.predict(X_sequences, verbose=0)
            
            # Flatten predictions if needed
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            # If we have fewer predictions than test samples, pad with the last prediction
            if len(predictions) < len(X_test):
                last_pred = predictions[-1] if len(predictions) > 0 else 0
                padding_needed = len(X_test) - len(predictions)
                predictions = np.concatenate([predictions, np.full(padding_needed, last_pred)])
            
            # If we have more predictions than needed, truncate
            elif len(predictions) > len(X_test):
                predictions = predictions[:len(X_test)]
            
            return predictions
            
        except Exception as e:
            raise ModelPredictionError(f"LSTM prediction failed: {str(e)}")
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        self.sequence_length = self.hyperparameters.get('sequence_length', 60)
        
        # For LSTM, we primarily use the target values to create sequences
        # X can be used for additional features in more complex implementations
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(y)):
            # Use previous sequence_length values as input
            X_sequences.append(y[i-self.sequence_length:i])
            y_sequences.append(y[i])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Reshape for LSTM (samples, time steps, features)
        X_sequences = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))
        
        return X_sequences, y_sequences
    
    def _create_prediction_sequences(self, X_test: np.ndarray) -> np.ndarray:
        """Create sequences for prediction from test data."""
        # For LSTM prediction, we need to create sequences properly
        # X_test represents the feature matrix, but for LSTM we need sequences
        
        # If X_test is too small, we can't create proper sequences
        if len(X_test) < self.sequence_length:
            # Use what we have and pad if necessary
            if X_test.ndim > 1:
                # Use the first feature column as the sequence
                sequence_data = X_test[:, 0]
            else:
                sequence_data = X_test
            
            # Pad with the last value if needed
            if len(sequence_data) < self.sequence_length:
                padding_needed = self.sequence_length - len(sequence_data)
                last_value = sequence_data[-1] if len(sequence_data) > 0 else 0
                sequence_data = np.concatenate([
                    np.full(padding_needed, last_value),
                    sequence_data
                ])
            
            # Create single sequence
            X_sequences = sequence_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
            return X_sequences
        
        # For larger datasets, create multiple sequences
        if X_test.ndim > 1:
            # Use the first feature column
            sequence_data = X_test[:, 0]
        else:
            sequence_data = X_test
        
        # Create sequences for prediction
        X_sequences = []
        for i in range(len(sequence_data) - self.sequence_length + 1):
            sequence = sequence_data[i:i + self.sequence_length]
            X_sequences.append(sequence)
        
        if not X_sequences:
            # Fallback: use the last sequence_length values
            sequence = sequence_data[-self.sequence_length:]
            X_sequences = [sequence]
        
        X_sequences = np.array(X_sequences)
        
        # Reshape for LSTM (samples, time steps, features)
        X_sequences = X_sequences.reshape((X_sequences.shape[0], X_sequences.shape[1], 1))
        
        return X_sequences
    
    def predict_sequence(self, initial_sequence: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Predict multiple steps ahead using recursive prediction.
        
        Args:
            initial_sequence: Initial sequence of length sequence_length
            n_steps: Number of steps to predict ahead
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ModelPredictionError("LSTM model not trained")
        
        if len(initial_sequence) != self.sequence_length:
            raise ModelPredictionError(f"Initial sequence must have length {self.sequence_length}")
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(n_steps):
            # Reshape for prediction
            X_input = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_pred = self.model.predict(X_input, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence (remove first element, add prediction)
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        return np.array(predictions)
    
    def _get_callbacks(self) -> list:
        """Get training callbacks."""
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.hyperparameters.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.hyperparameters.get('reduce_lr_factor', 0.5),
            patience=self.hyperparameters.get('reduce_lr_patience', 5),
            min_lr=1e-7,
            verbose=0
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance for LSTM (simplified approach).
        
        Returns:
            Dictionary with importance scores
        """
        # LSTM feature importance is complex to compute
        # This is a simplified approach using gradient-based importance
        
        if self.model is None:
            return {}
        
        # For now, return equal importance for all time steps
        importance = {}
        for i in range(self.sequence_length):
            importance[f'timestep_{i}'] = 1.0 / self.sequence_length
        
        return importance
    
    def plot_training_history(self):
        """Plot training history (requires matplotlib)."""
        if not self.training_history:
            raise ModelTrainingError("No training history available")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.training_history['loss'], label='Training Loss')
            if self.training_history.get('val_loss'):
                ax1.plot(self.training_history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot MAE
            if self.training_history.get('mae'):
                ax2.plot(self.training_history['mae'], label='Training MAE')
            if self.training_history.get('val_mae'):
                ax2.plot(self.training_history['val_mae'], label='Validation MAE')
            ax2.set_title('Model MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            self.logger.warning("Matplotlib not available for plotting")
            return None
    
    def save_model(self, filepath: str = None) -> str:
        """Save LSTM model with custom handling for TensorFlow models."""
        if not self.is_trained:
            raise ModelTrainingError(f"Cannot save untrained model {self.name}")
        
        if filepath:
            # Save TensorFlow model
            self.model.save(filepath)
            return filepath
        else:
            # Use base class persistence for metadata, but save TF model separately
            import tempfile
            import os
            
            # Create temporary directory for TF model
            temp_dir = tempfile.mkdtemp()
            tf_model_path = os.path.join(temp_dir, f"{self.name}_tf_model")
            self.model.save(tf_model_path)
            
            # Save metadata using base class
            metadata = {
                'model_name': self.name,
                'hyperparameters': self.hyperparameters,
                'feature_names': self.feature_names,
                'training_time': self.training_time,
                'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                'training_history': self.training_history,
                'tf_model_path': tf_model_path,
                'sequence_length': self.sequence_length
            }
            
            return self.persistence.save_model(tf_model_path, self.name, metadata)
    
    def load_model(self, filepath: str = None) -> None:
        """Load LSTM model with custom handling for TensorFlow models."""
        try:
            if filepath and filepath.endswith('.h5'):
                # Load TensorFlow model directly
                self.model = tf.keras.models.load_model(filepath)
            else:
                # Load using base class persistence
                if filepath is None:
                    filepath = self.persistence.get_latest_model(self.name)
                
                # Load metadata first
                metadata = self.persistence.load_model_metadata(filepath)
                if metadata and 'tf_model_path' in metadata:
                    # Load TensorFlow model from metadata path
                    self.model = tf.keras.models.load_model(metadata['tf_model_path'])
                    
                    # Restore other attributes
                    self.hyperparameters = metadata.get('hyperparameters', {})
                    self.feature_names = metadata.get('feature_names', [])
                    self.training_time = metadata.get('training_time', 0.0)
                    self.training_history = metadata.get('training_history', {})
                    self.sequence_length = metadata.get('sequence_length', 60)
                    
                    if metadata.get('last_trained'):
                        from datetime import datetime
                        self.last_trained = datetime.fromisoformat(metadata['last_trained'])
                else:
                    # Fallback to direct loading
                    self.model = tf.keras.models.load_model(filepath)
            
            self.is_trained = True
            self.logger.info(f"LSTM model loaded from {filepath}")
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load LSTM model: {str(e)}")
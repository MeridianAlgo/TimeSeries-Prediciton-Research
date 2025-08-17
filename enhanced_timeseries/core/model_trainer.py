"""
Model Trainer for Time Series Prediction
=======================================

Handles training of various neural network models for time series prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Comprehensive model trainer for time series prediction models.
    
    Handles training, validation, and evaluation of neural network models.
    """
    
    def __init__(self, 
                 device: str = None,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        Initialize model trainer.
        
        Args:
            device: Device to use for training ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.history = {'train_loss': [], 'val_loss': []}
        
    def create_data_loaders(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray = None, y_val: np.ndarray = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create PyTorch data loaders for training and validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        
        # Create training dataset and loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create validation loader if validation data provided
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_model(self, model: nn.Module, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 100, patience: int = 10,
                   verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train a neural network model.
        
        Args:
            model: PyTorch model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Move model to device
        model = model.to(self.device)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=verbose)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch {epoch + 1:3d}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch + 1:3d}: Train Loss: {train_loss:.6f}")
        
        self.history = history
        return history
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray,
                      scaler=None) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained PyTorch model
            X_test: Test features
            y_test: Test targets
            scaler: Scaler for inverse transformation (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        model = model.to(self.device)
        
        # Make predictions
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = model(X_test_tensor).cpu().numpy().flatten()
        
        # Inverse transform if scaler provided
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_test_original = y_test
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, predictions)
        mae = mean_absolute_error(y_test_original, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, predictions)
        
        # Calculate directional accuracy
        direction_correct = np.sum(np.sign(np.diff(predictions)) == np.sign(np.diff(y_test_original)))
        directional_accuracy = direction_correct / (len(predictions) - 1) * 100
        
        # Calculate correlation
        correlation = np.corrcoef(y_test_original, predictions)[0, 1]
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation
        }
        
        return metrics
    
    def predict(self, model: nn.Module, X: np.ndarray, scaler=None) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained PyTorch model
            X: Input features
            scaler: Scaler for inverse transformation (optional)
            
        Returns:
            Predictions array
        """
        model.eval()
        model = model.to(self.device)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = model(X_tensor).cpu().numpy().flatten()
        
        if scaler is not None:
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model: nn.Module, filepath: str):
        """Save a trained model."""
        torch.save(model.state_dict(), filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, model: nn.Module, filepath: str):
        """Load a trained model."""
        model.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"✅ Model loaded from {filepath}")
        return model
    
    def get_model_summary(self, model: nn.Module) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = f"""
Model Summary:
==============
Total Parameters: {total_params:,}
Trainable Parameters: {trainable_params:,}
Device: {self.device}
        """
        
        return summary

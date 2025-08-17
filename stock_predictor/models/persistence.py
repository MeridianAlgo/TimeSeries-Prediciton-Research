"""Model persistence utilities for saving and loading trained models."""

import os
import pickle
import joblib
import json
from datetime import datetime
from typing import Any, Dict
from pathlib import Path
from stock_predictor.utils.logging import get_logger
from stock_predictor.utils.exceptions import ModelTrainingError


class ModelPersistence:
    """Handles saving and loading of trained models and metadata."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logger = get_logger('models.persistence')
    
    def save_model(self, model: Any, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained model object
            model_name: Name identifier for the model
            metadata: Additional metadata to save with model
            
        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        try:
            # Save model using joblib (better for sklearn models)
            joblib.dump(model, model_path)
            
            # Save metadata
            if metadata:
                metadata_path = model_path.with_suffix('.json')
                metadata['saved_at'] = datetime.now().isoformat()
                metadata['model_file'] = model_filename
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model saved: {model_path}")
            return str(model_path)
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to save model {model_name}: {str(e)}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Model loaded: {model_path}")
            return model
            
        except Exception as e:
            raise ModelTrainingError(f"Failed to load model from {model_path}: {str(e)}")
    
    def load_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """
        Load model metadata.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Model metadata dictionary
        """
        metadata_path = Path(model_path).with_suffix('.json')
        
        if not metadata_path.exists():
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to load metadata for {model_path}: {str(e)}")
            return {}
    
    def list_saved_models(self, model_name: str = None) -> list:
        """
        List all saved models, optionally filtered by name.
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            List of model file paths
        """
        pattern = f"{model_name}_*.pkl" if model_name else "*.pkl"
        model_files = list(self.models_dir.glob(pattern))
        
        # Sort by modification time (newest first)
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return [str(f) for f in model_files]
    
    def get_latest_model(self, model_name: str) -> str:
        """
        Get the path to the latest saved model of a given name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to latest model file
            
        Raises:
            ModelTrainingError: If no models found
        """
        models = self.list_saved_models(model_name)
        
        if not models:
            raise ModelTrainingError(f"No saved models found for {model_name}")
        
        return models[0]  # First item is newest due to sorting
    
    def delete_old_models(self, model_name: str, keep_count: int = 5) -> None:
        """
        Delete old model files, keeping only the most recent ones.
        
        Args:
            model_name: Name of the model
            keep_count: Number of recent models to keep
        """
        models = self.list_saved_models(model_name)
        
        if len(models) <= keep_count:
            return
        
        models_to_delete = models[keep_count:]
        
        for model_path in models_to_delete:
            try:
                os.remove(model_path)
                
                # Also remove metadata file if it exists
                metadata_path = Path(model_path).with_suffix('.json')
                if metadata_path.exists():
                    os.remove(metadata_path)
                
                self.logger.info(f"Deleted old model: {model_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to delete {model_path}: {str(e)}")


class HyperparameterManager:
    """Manages hyperparameters for models."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.logger = get_logger('models.hyperparameters')
    
    def save_hyperparameters(self, model_name: str, hyperparameters: Dict[str, Any]) -> None:
        """
        Save hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            hyperparameters: Dictionary of hyperparameters
        """
        config_file = self.config_dir / f"{model_name}_hyperparameters.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(hyperparameters, f, indent=2)
            
            self.logger.info(f"Hyperparameters saved for {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to save hyperparameters for {model_name}: {str(e)}")
    
    def load_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """
        Load hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of hyperparameters
        """
        config_file = self.config_dir / f"{model_name}_hyperparameters.json"
        
        if not config_file.exists():
            self.logger.warning(f"No hyperparameters file found for {model_name}")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                hyperparameters = json.load(f)
            
            self.logger.info(f"Hyperparameters loaded for {model_name}")
            return hyperparameters
            
        except Exception as e:
            self.logger.error(f"Failed to load hyperparameters for {model_name}: {str(e)}")
            return {}
    
    def update_hyperparameters(self, model_name: str, updates: Dict[str, Any]) -> None:
        """
        Update specific hyperparameters for a model.
        
        Args:
            model_name: Name of the model
            updates: Dictionary of hyperparameter updates
        """
        current_params = self.load_hyperparameters(model_name)
        current_params.update(updates)
        self.save_hyperparameters(model_name, current_params)
"""
Automated model configuration management system.
Implements configuration saving, versioning, comparison, and performance tracking.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pickle
import hashlib
from pathlib import Path
import logging
import shutil
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ModelConfiguration:
    """Model configuration data structure."""
    
    config_id: str
    model_type: str
    parameters: Dict[str, Any]
    created_at: datetime
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    parent_config_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfiguration':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class PerformanceRecord:
    """Performance record for a configuration."""
    
    record_id: str
    config_id: str
    metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    timestamp: datetime
    training_time: Optional[float] = None
    inference_time: Optional[float] = None
    model_size: Optional[int] = None  # Number of parameters
    memory_usage: Optional[float] = None  # Peak memory usage in MB
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceRecord':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class ConfigurationManager:
    """Manager for model configurations and performance tracking."""
    
    def __init__(self, storage_path: str = "model_configs"):
        """
        Initialize configuration manager.
        
        Args:
            storage_path: Path to store configurations and performance data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.configs_path = self.storage_path / "configurations"
        self.performance_path = self.storage_path / "performance"
        self.models_path = self.storage_path / "models"
        
        # Create subdirectories
        self.configs_path.mkdir(exist_ok=True)
        self.performance_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # In-memory caches
        self._config_cache = {}
        self._performance_cache = defaultdict(list)
        
        # Load existing data
        self._load_configurations()
        self._load_performance_records()
    
    def _generate_config_id(self, model_type: str, parameters: Dict[str, Any]) -> str:
        """Generate unique configuration ID."""
        # Create hash from model type and parameters
        config_str = f"{model_type}_{json.dumps(parameters, sort_keys=True)}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{model_type}_{timestamp}_{config_hash}"
    
    def _generate_record_id(self) -> str:
        """Generate unique performance record ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"perf_{timestamp}"
    
    def save_configuration(self, 
                          model_type: str,
                          parameters: Dict[str, Any],
                          description: Optional[str] = None,
                          tags: Optional[List[str]] = None,
                          parent_config_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a model configuration.
        
        Args:
            model_type: Type of model (e.g., 'transformer', 'cnn_lstm')
            parameters: Model parameters
            description: Optional description
            tags: Optional tags for categorization
            parent_config_id: ID of parent configuration (for versioning)
            metadata: Additional metadata
            
        Returns:
            Configuration ID
        """
        config_id = self._generate_config_id(model_type, parameters)
        
        config = ModelConfiguration(
            config_id=config_id,
            model_type=model_type,
            parameters=parameters.copy(),
            created_at=datetime.now(),
            description=description,
            tags=tags or [],
            parent_config_id=parent_config_id,
            metadata=metadata or {}
        )
        
        # Save to file
        config_file = self.configs_path / f"{config_id}.json"
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Update cache
        self._config_cache[config_id] = config
        
        logging.info(f"Saved configuration {config_id}")
        return config_id
    
    def load_configuration(self, config_id: str) -> Optional[ModelConfiguration]:
        """
        Load a configuration by ID.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            ModelConfiguration or None if not found
        """
        if config_id in self._config_cache:
            return self._config_cache[config_id]
        
        config_file = self.configs_path / f"{config_id}.json"
        if not config_file.exists():
            return None
        
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            config = ModelConfiguration.from_dict(data)
            self._config_cache[config_id] = config
            return config
            
        except Exception as e:
            logging.error(f"Error loading configuration {config_id}: {e}")
            return None
    
    def delete_configuration(self, config_id: str) -> bool:
        """
        Delete a configuration.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            True if deleted successfully
        """
        config_file = self.configs_path / f"{config_id}.json"
        
        if config_file.exists():
            config_file.unlink()
            
            # Remove from cache
            if config_id in self._config_cache:
                del self._config_cache[config_id]
            
            # Remove associated performance records
            if config_id in self._performance_cache:
                del self._performance_cache[config_id]
            
            logging.info(f"Deleted configuration {config_id}")
            return True
        
        return False
    
    def list_configurations(self, 
                           model_type: Optional[str] = None,
                           tags: Optional[List[str]] = None,
                           limit: Optional[int] = None) -> List[ModelConfiguration]:
        """
        List configurations with optional filtering.
        
        Args:
            model_type: Filter by model type
            tags: Filter by tags (must have all specified tags)
            limit: Maximum number of configurations to return
            
        Returns:
            List of configurations
        """
        configs = list(self._config_cache.values())
        
        # Apply filters
        if model_type:
            configs = [c for c in configs if c.model_type == model_type]
        
        if tags:
            configs = [c for c in configs if all(tag in c.tags for tag in tags)]
        
        # Sort by creation time (newest first)
        configs.sort(key=lambda c: c.created_at, reverse=True)
        
        # Apply limit
        if limit:
            configs = configs[:limit]
        
        return configs
    
    def record_performance(self,
                          config_id: str,
                          metrics: Dict[str, float],
                          dataset_info: Dict[str, Any],
                          training_time: Optional[float] = None,
                          inference_time: Optional[float] = None,
                          model_size: Optional[int] = None,
                          memory_usage: Optional[float] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record performance for a configuration.
        
        Args:
            config_id: Configuration ID
            metrics: Performance metrics (e.g., {'mse': 0.1, 'mae': 0.05})
            dataset_info: Information about the dataset used
            training_time: Training time in seconds
            inference_time: Inference time per sample in seconds
            model_size: Number of model parameters
            memory_usage: Peak memory usage in MB
            metadata: Additional metadata
            
        Returns:
            Performance record ID
        """
        record_id = self._generate_record_id()
        
        record = PerformanceRecord(
            record_id=record_id,
            config_id=config_id,
            metrics=metrics.copy(),
            dataset_info=dataset_info.copy(),
            timestamp=datetime.now(),
            training_time=training_time,
            inference_time=inference_time,
            model_size=model_size,
            memory_usage=memory_usage,
            metadata=metadata or {}
        )
        
        # Save to file
        record_file = self.performance_path / f"{record_id}.json"
        with open(record_file, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
        
        # Update cache
        self._performance_cache[config_id].append(record)
        
        logging.info(f"Recorded performance {record_id} for config {config_id}")
        return record_id
    
    def get_performance_history(self, config_id: str) -> List[PerformanceRecord]:
        """
        Get performance history for a configuration.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            List of performance records
        """
        return self._performance_cache.get(config_id, [])
    
    def compare_configurations(self, 
                              config_ids: List[str],
                              metric: str = 'mse') -> pd.DataFrame:
        """
        Compare configurations based on performance.
        
        Args:
            config_ids: List of configuration IDs to compare
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for config_id in config_ids:
            config = self.load_configuration(config_id)
            if not config:
                continue
            
            performance_records = self.get_performance_history(config_id)
            
            if performance_records:
                # Get best performance for this metric
                best_record = min(performance_records, 
                                key=lambda r: r.metrics.get(metric, float('inf')))
                
                row = {
                    'config_id': config_id,
                    'model_type': config.model_type,
                    'created_at': config.created_at,
                    'description': config.description or '',
                    f'best_{metric}': best_record.metrics.get(metric),
                    'training_time': best_record.training_time,
                    'inference_time': best_record.inference_time,
                    'model_size': best_record.model_size,
                    'memory_usage': best_record.memory_usage
                }
                
                # Add parameter values
                for param_name, param_value in config.parameters.items():
                    row[f'param_{param_name}'] = param_value
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def find_best_configuration(self, 
                               model_type: Optional[str] = None,
                               metric: str = 'mse',
                               tags: Optional[List[str]] = None) -> Optional[Tuple[ModelConfiguration, PerformanceRecord]]:
        """
        Find the best performing configuration.
        
        Args:
            model_type: Filter by model type
            metric: Metric to optimize (lower is better)
            tags: Filter by tags
            
        Returns:
            Tuple of (best_config, best_performance_record) or None
        """
        configs = self.list_configurations(model_type=model_type, tags=tags)
        
        best_config = None
        best_record = None
        best_value = float('inf')
        
        for config in configs:
            performance_records = self.get_performance_history(config.config_id)
            
            for record in performance_records:
                if metric in record.metrics:
                    value = record.metrics[metric]
                    if value < best_value:
                        best_value = value
                        best_config = config
                        best_record = record
        
        if best_config and best_record:
            return best_config, best_record
        
        return None
    
    def create_configuration_variant(self,
                                   base_config_id: str,
                                   parameter_updates: Dict[str, Any],
                                   description: Optional[str] = None) -> str:
        """
        Create a variant of an existing configuration.
        
        Args:
            base_config_id: ID of base configuration
            parameter_updates: Parameters to update
            description: Description of the variant
            
        Returns:
            New configuration ID
        """
        base_config = self.load_configuration(base_config_id)
        if not base_config:
            raise ValueError(f"Base configuration {base_config_id} not found")
        
        # Create new parameters by updating base parameters
        new_parameters = base_config.parameters.copy()
        new_parameters.update(parameter_updates)
        
        # Create variant description
        if not description:
            updated_params = list(parameter_updates.keys())
            description = f"Variant of {base_config_id} with updated: {', '.join(updated_params)}"
        
        # Save new configuration
        new_config_id = self.save_configuration(
            model_type=base_config.model_type,
            parameters=new_parameters,
            description=description,
            tags=base_config.tags + ['variant'],
            parent_config_id=base_config_id
        )
        
        return new_config_id
    
    def get_configuration_lineage(self, config_id: str) -> List[ModelConfiguration]:
        """
        Get the lineage (ancestry) of a configuration.
        
        Args:
            config_id: Configuration ID
            
        Returns:
            List of configurations from root to current
        """
        lineage = []
        current_id = config_id
        
        while current_id:
            config = self.load_configuration(current_id)
            if not config:
                break
            
            lineage.insert(0, config)  # Insert at beginning
            current_id = config.parent_config_id
        
        return lineage
    
    def export_configuration(self, config_id: str, export_path: str) -> bool:
        """
        Export a configuration and its performance history.
        
        Args:
            config_id: Configuration ID
            export_path: Path to export to
            
        Returns:
            True if exported successfully
        """
        try:
            config = self.load_configuration(config_id)
            if not config:
                return False
            
            performance_records = self.get_performance_history(config_id)
            
            export_data = {
                'configuration': config.to_dict(),
                'performance_history': [record.to_dict() for record in performance_records]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logging.info(f"Exported configuration {config_id} to {export_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting configuration {config_id}: {e}")
            return False
    
    def import_configuration(self, import_path: str) -> Optional[str]:
        """
        Import a configuration from file.
        
        Args:
            import_path: Path to import from
            
        Returns:
            New configuration ID or None if failed
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            config_data = import_data['configuration']
            performance_data = import_data.get('performance_history', [])
            
            # Create new configuration (will get new ID)
            config = ModelConfiguration.from_dict(config_data)
            new_config_id = self.save_configuration(
                model_type=config.model_type,
                parameters=config.parameters,
                description=f"Imported: {config.description or ''}",
                tags=config.tags + ['imported'],
                metadata=config.metadata
            )
            
            # Import performance records
            for record_data in performance_data:
                record = PerformanceRecord.from_dict(record_data)
                record.config_id = new_config_id  # Update to new config ID
                record.record_id = self._generate_record_id()  # Generate new record ID
                
                self.record_performance(
                    config_id=new_config_id,
                    metrics=record.metrics,
                    dataset_info=record.dataset_info,
                    training_time=record.training_time,
                    inference_time=record.inference_time,
                    model_size=record.model_size,
                    memory_usage=record.memory_usage,
                    metadata=record.metadata
                )
            
            logging.info(f"Imported configuration as {new_config_id}")
            return new_config_id
            
        except Exception as e:
            logging.error(f"Error importing configuration: {e}")
            return None
    
    def cleanup_old_configurations(self, days: int = 30, keep_best: int = 5) -> int:
        """
        Clean up old configurations while keeping the best performing ones.
        
        Args:
            days: Delete configurations older than this many days
            keep_best: Number of best configurations to keep regardless of age
            
        Returns:
            Number of configurations deleted
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        
        # Get all configurations
        all_configs = list(self._config_cache.values())
        
        # Find best configurations to keep
        configs_with_performance = []
        for config in all_configs:
            performance_records = self.get_performance_history(config.config_id)
            if performance_records:
                best_mse = min(record.metrics.get('mse', float('inf')) 
                              for record in performance_records)
                configs_with_performance.append((config, best_mse))
        
        # Sort by performance and keep best ones
        configs_with_performance.sort(key=lambda x: x[1])
        configs_to_keep = {config.config_id for config, _ in configs_with_performance[:keep_best]}
        
        # Delete old configurations (except best ones)
        deleted_count = 0
        for config in all_configs:
            if (config.created_at < cutoff_date and 
                config.config_id not in configs_to_keep):
                
                if self.delete_configuration(config.config_id):
                    deleted_count += 1
        
        logging.info(f"Cleaned up {deleted_count} old configurations")
        return deleted_count
    
    def _load_configurations(self):
        """Load all configurations from storage."""
        for config_file in self.configs_path.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                
                config = ModelConfiguration.from_dict(data)
                self._config_cache[config.config_id] = config
                
            except Exception as e:
                logging.error(f"Error loading configuration from {config_file}: {e}")
    
    def _load_performance_records(self):
        """Load all performance records from storage."""
        for record_file in self.performance_path.glob("*.json"):
            try:
                with open(record_file, 'r') as f:
                    data = json.load(f)
                
                record = PerformanceRecord.from_dict(data)
                self._performance_cache[record.config_id].append(record)
                
            except Exception as e:
                logging.error(f"Error loading performance record from {record_file}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored configurations and performance."""
        total_configs = len(self._config_cache)
        total_records = sum(len(records) for records in self._performance_cache.values())
        
        # Count by model type
        model_type_counts = defaultdict(int)
        for config in self._config_cache.values():
            model_type_counts[config.model_type] += 1
        
        # Storage usage
        storage_size = sum(f.stat().st_size for f in self.storage_path.rglob("*") if f.is_file())
        
        return {
            'total_configurations': total_configs,
            'total_performance_records': total_records,
            'configurations_by_type': dict(model_type_counts),
            'storage_size_mb': storage_size / (1024 * 1024),
            'storage_path': str(self.storage_path)
        }


# Utility functions
def create_config_manager(storage_path: str = "model_configs") -> ConfigurationManager:
    """Create a configuration manager."""
    return ConfigurationManager(storage_path)


def auto_save_model_config(model, model_type: str, parameters: Dict[str, Any],
                          config_manager: ConfigurationManager,
                          description: Optional[str] = None) -> str:
    """
    Automatically save model configuration.
    
    Args:
        model: Model instance
        model_type: Type of model
        parameters: Model parameters
        config_manager: Configuration manager
        description: Optional description
        
    Returns:
        Configuration ID
    """
    # Add model size to metadata
    if hasattr(model, 'parameters'):
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metadata = {'model_size': model_size}
    else:
        metadata = {}
    
    config_id = config_manager.save_configuration(
        model_type=model_type,
        parameters=parameters,
        description=description,
        metadata=metadata
    )
    
    return config_id
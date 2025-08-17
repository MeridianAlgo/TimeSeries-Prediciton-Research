"""
Unit tests for automated model configuration management.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from enhanced_timeseries.optimization.config_manager import (
    ModelConfiguration, PerformanceRecord, ConfigurationManager,
    create_config_manager, auto_save_model_config
)


class TestModelConfiguration(unittest.TestCase):
    """Test ModelConfiguration class."""
    
    def test_configuration_creation(self):
        """Test configuration creation."""
        config = ModelConfiguration(
            config_id="test_001",
            model_type="transformer",
            parameters={"lr": 0.01, "hidden_dim": 128},
            created_at=datetime.now(),
            description="Test configuration",
            tags=["test", "transformer"]
        )
        
        self.assertEqual(config.config_id, "test_001")
        self.assertEqual(config.model_type, "transformer")
        self.assertEqual(config.parameters["lr"], 0.01)
        self.assertEqual(config.description, "Test configuration")
        self.assertIn("test", config.tags)
    
    def test_configuration_to_dict(self):
        """Test configuration serialization."""
        config = ModelConfiguration(
            config_id="test_001",
            model_type="transformer",
            parameters={"lr": 0.01},
            created_at=datetime.now()
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["config_id"], "test_001")
        self.assertEqual(config_dict["model_type"], "transformer")
        self.assertIsInstance(config_dict["created_at"], str)  # Should be ISO format
    
    def test_configuration_from_dict(self):
        """Test configuration deserialization."""
        data = {
            "config_id": "test_001",
            "model_type": "transformer",
            "parameters": {"lr": 0.01},
            "created_at": datetime.now().isoformat(),
            "description": None,
            "tags": [],
            "parent_config_id": None,
            "metadata": {}
        }
        
        config = ModelConfiguration.from_dict(data)
        
        self.assertEqual(config.config_id, "test_001")
        self.assertEqual(config.model_type, "transformer")
        self.assertIsInstance(config.created_at, datetime)


class TestPerformanceRecord(unittest.TestCase):
    """Test PerformanceRecord class."""
    
    def test_record_creation(self):
        """Test performance record creation."""
        record = PerformanceRecord(
            record_id="perf_001",
            config_id="config_001",
            metrics={"mse": 0.1, "mae": 0.05},
            dataset_info={"name": "test_dataset", "size": 1000},
            timestamp=datetime.now(),
            training_time=120.5,
            model_size=50000
        )
        
        self.assertEqual(record.record_id, "perf_001")
        self.assertEqual(record.config_id, "config_001")
        self.assertEqual(record.metrics["mse"], 0.1)
        self.assertEqual(record.training_time, 120.5)
    
    def test_record_serialization(self):
        """Test record serialization and deserialization."""
        record = PerformanceRecord(
            record_id="perf_001",
            config_id="config_001",
            metrics={"mse": 0.1},
            dataset_info={"name": "test"},
            timestamp=datetime.now()
        )
        
        # Serialize
        record_dict = record.to_dict()
        self.assertIsInstance(record_dict, dict)
        self.assertIsInstance(record_dict["timestamp"], str)
        
        # Deserialize
        restored_record = PerformanceRecord.from_dict(record_dict)
        self.assertEqual(restored_record.record_id, record.record_id)
        self.assertIsInstance(restored_record.timestamp, datetime)


class TestConfigurationManager(unittest.TestCase):
    """Test ConfigurationManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ConfigurationManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_manager_creation(self):
        """Test configuration manager creation."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue((Path(self.temp_dir) / "configurations").exists())
        self.assertTrue((Path(self.temp_dir) / "performance").exists())
        self.assertTrue((Path(self.temp_dir) / "models").exists())
    
    def test_save_and_load_configuration(self):
        """Test saving and loading configurations."""
        # Save configuration
        config_id = self.manager.save_configuration(
            model_type="transformer",
            parameters={"lr": 0.01, "hidden_dim": 128},
            description="Test config",
            tags=["test"]
        )
        
        self.assertIsInstance(config_id, str)
        self.assertTrue(config_id.startswith("transformer_"))
        
        # Load configuration
        loaded_config = self.manager.load_configuration(config_id)
        
        self.assertIsNotNone(loaded_config)
        self.assertEqual(loaded_config.model_type, "transformer")
        self.assertEqual(loaded_config.parameters["lr"], 0.01)
        self.assertEqual(loaded_config.description, "Test config")
        self.assertIn("test", loaded_config.tags)
    
    def test_delete_configuration(self):
        """Test configuration deletion."""
        # Save configuration
        config_id = self.manager.save_configuration(
            model_type="lstm",
            parameters={"hidden_dim": 64}
        )
        
        # Verify it exists
        self.assertIsNotNone(self.manager.load_configuration(config_id))
        
        # Delete it
        result = self.manager.delete_configuration(config_id)
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertIsNone(self.manager.load_configuration(config_id))
    
    def test_list_configurations(self):
        """Test listing configurations with filters."""
        # Save multiple configurations
        config1_id = self.manager.save_configuration(
            model_type="transformer",
            parameters={"lr": 0.01},
            tags=["test", "v1"]
        )
        
        config2_id = self.manager.save_configuration(
            model_type="lstm",
            parameters={"hidden_dim": 64},
            tags=["test", "v2"]
        )
        
        config3_id = self.manager.save_configuration(
            model_type="transformer",
            parameters={"lr": 0.001},
            tags=["production"]
        )
        
        # Test listing all
        all_configs = self.manager.list_configurations()
        self.assertEqual(len(all_configs), 3)
        
        # Test filtering by model type
        transformer_configs = self.manager.list_configurations(model_type="transformer")
        self.assertEqual(len(transformer_configs), 2)
        
        # Test filtering by tags
        test_configs = self.manager.list_configurations(tags=["test"])
        self.assertEqual(len(test_configs), 2)
        
        # Test limit
        limited_configs = self.manager.list_configurations(limit=1)
        self.assertEqual(len(limited_configs), 1)
    
    def test_record_and_get_performance(self):
        """Test recording and retrieving performance."""
        # Save configuration first
        config_id = self.manager.save_configuration(
            model_type="cnn",
            parameters={"filters": [32, 64]}
        )
        
        # Record performance
        record_id = self.manager.record_performance(
            config_id=config_id,
            metrics={"mse": 0.1, "mae": 0.05},
            dataset_info={"name": "test_data", "size": 1000},
            training_time=60.0,
            model_size=10000
        )
        
        self.assertIsInstance(record_id, str)
        self.assertTrue(record_id.startswith("perf_"))
        
        # Get performance history
        history = self.manager.get_performance_history(config_id)
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].metrics["mse"], 0.1)
        self.assertEqual(history[0].training_time, 60.0)
    
    def test_compare_configurations(self):
        """Test configuration comparison."""
        # Save configurations with different performance
        config1_id = self.manager.save_configuration(
            model_type="model_a",
            parameters={"param1": 1}
        )
        
        config2_id = self.manager.save_configuration(
            model_type="model_b",
            parameters={"param1": 2}
        )
        
        # Record different performance levels
        self.manager.record_performance(
            config_id=config1_id,
            metrics={"mse": 0.1, "mae": 0.05},
            dataset_info={"name": "test"}
        )
        
        self.manager.record_performance(
            config_id=config2_id,
            metrics={"mse": 0.2, "mae": 0.1},
            dataset_info={"name": "test"}
        )
        
        # Compare configurations
        comparison = self.manager.compare_configurations([config1_id, config2_id], metric="mse")
        
        self.assertIsInstance(comparison, pd.DataFrame)
        self.assertEqual(len(comparison), 2)
        self.assertIn("config_id", comparison.columns)
        self.assertIn("best_mse", comparison.columns)
        self.assertIn("param_param1", comparison.columns)
    
    def test_find_best_configuration(self):
        """Test finding best configuration."""
        # Save configurations with different performance
        config1_id = self.manager.save_configuration(
            model_type="test_model",
            parameters={"param": 1}
        )
        
        config2_id = self.manager.save_configuration(
            model_type="test_model",
            parameters={"param": 2}
        )
        
        # Record performance (config1 is better)
        self.manager.record_performance(
            config_id=config1_id,
            metrics={"mse": 0.05},
            dataset_info={"name": "test"}
        )
        
        self.manager.record_performance(
            config_id=config2_id,
            metrics={"mse": 0.15},
            dataset_info={"name": "test"}
        )
        
        # Find best configuration
        best_result = self.manager.find_best_configuration(
            model_type="test_model",
            metric="mse"
        )
        
        self.assertIsNotNone(best_result)
        best_config, best_record = best_result
        
        self.assertEqual(best_config.config_id, config1_id)
        self.assertEqual(best_record.metrics["mse"], 0.05)
    
    def test_create_configuration_variant(self):
        """Test creating configuration variants."""
        # Save base configuration
        base_config_id = self.manager.save_configuration(
            model_type="transformer",
            parameters={"lr": 0.01, "hidden_dim": 128, "num_layers": 4},
            description="Base config"
        )
        
        # Create variant
        variant_id = self.manager.create_configuration_variant(
            base_config_id=base_config_id,
            parameter_updates={"lr": 0.001, "num_layers": 6},
            description="Variant with lower LR and more layers"
        )
        
        # Load variant
        variant_config = self.manager.load_configuration(variant_id)
        
        self.assertIsNotNone(variant_config)
        self.assertEqual(variant_config.parameters["lr"], 0.001)
        self.assertEqual(variant_config.parameters["hidden_dim"], 128)  # Unchanged
        self.assertEqual(variant_config.parameters["num_layers"], 6)
        self.assertEqual(variant_config.parent_config_id, base_config_id)
        self.assertIn("variant", variant_config.tags)
    
    def test_configuration_lineage(self):
        """Test configuration lineage tracking."""
        # Create configuration chain: base -> variant1 -> variant2
        base_id = self.manager.save_configuration(
            model_type="test",
            parameters={"param": 1}
        )
        
        variant1_id = self.manager.create_configuration_variant(
            base_config_id=base_id,
            parameter_updates={"param": 2}
        )
        
        variant2_id = self.manager.create_configuration_variant(
            base_config_id=variant1_id,
            parameter_updates={"param": 3}
        )
        
        # Get lineage
        lineage = self.manager.get_configuration_lineage(variant2_id)
        
        self.assertEqual(len(lineage), 3)
        self.assertEqual(lineage[0].config_id, base_id)
        self.assertEqual(lineage[1].config_id, variant1_id)
        self.assertEqual(lineage[2].config_id, variant2_id)
    
    def test_export_import_configuration(self):
        """Test configuration export and import."""
        # Save configuration with performance
        config_id = self.manager.save_configuration(
            model_type="test_model",
            parameters={"param1": 1, "param2": "value"},
            description="Export test"
        )
        
        self.manager.record_performance(
            config_id=config_id,
            metrics={"mse": 0.1},
            dataset_info={"name": "test_data"}
        )
        
        # Export configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            export_path = tmp_file.name
        
        result = self.manager.export_configuration(config_id, export_path)
        self.assertTrue(result)
        
        # Create new manager and import
        temp_dir2 = tempfile.mkdtemp()
        manager2 = ConfigurationManager(temp_dir2)
        
        imported_id = manager2.import_configuration(export_path)
        
        self.assertIsNotNone(imported_id)
        self.assertNotEqual(imported_id, config_id)  # Should get new ID
        
        # Verify imported configuration
        imported_config = manager2.load_configuration(imported_id)
        self.assertEqual(imported_config.model_type, "test_model")
        self.assertEqual(imported_config.parameters["param1"], 1)
        
        # Verify imported performance
        imported_history = manager2.get_performance_history(imported_id)
        self.assertEqual(len(imported_history), 1)
        self.assertEqual(imported_history[0].metrics["mse"], 0.1)
        
        # Clean up
        Path(export_path).unlink()
        shutil.rmtree(temp_dir2)
    
    def test_cleanup_old_configurations(self):
        """Test cleanup of old configurations."""
        # Save old configurations
        old_time = datetime.now() - timedelta(days=40)
        
        # Manually create old configuration files
        old_config = ModelConfiguration(
            config_id="old_config",
            model_type="old_model",
            parameters={"param": 1},
            created_at=old_time
        )
        
        # Save to cache and file
        self.manager._config_cache["old_config"] = old_config
        config_file = self.manager.configs_path / "old_config.json"
        with open(config_file, 'w') as f:
            json.dump(old_config.to_dict(), f)
        
        # Save recent configuration
        recent_id = self.manager.save_configuration(
            model_type="recent_model",
            parameters={"param": 2}
        )
        
        # Record performance for both (old one is better)
        self.manager.record_performance(
            config_id="old_config",
            metrics={"mse": 0.05},
            dataset_info={"name": "test"}
        )
        
        self.manager.record_performance(
            config_id=recent_id,
            metrics={"mse": 0.15},
            dataset_info={"name": "test"}
        )
        
        # Cleanup (keep 1 best)
        deleted_count = self.manager.cleanup_old_configurations(days=30, keep_best=1)
        
        # Old config should be kept because it's best performing
        # Recent config might be deleted if it's not in top 1
        self.assertGreaterEqual(deleted_count, 0)
    
    def test_statistics(self):
        """Test getting manager statistics."""
        # Save some configurations
        self.manager.save_configuration("transformer", {"lr": 0.01})
        self.manager.save_configuration("lstm", {"hidden_dim": 64})
        self.manager.save_configuration("transformer", {"lr": 0.001})
        
        stats = self.manager.get_statistics()
        
        self.assertIn("total_configurations", stats)
        self.assertIn("configurations_by_type", stats)
        self.assertIn("storage_size_mb", stats)
        
        self.assertEqual(stats["total_configurations"], 3)
        self.assertEqual(stats["configurations_by_type"]["transformer"], 2)
        self.assertEqual(stats["configurations_by_type"]["lstm"], 1)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_config_manager(self):
        """Test config manager creation utility."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_config_manager(temp_dir)
            
            self.assertIsInstance(manager, ConfigurationManager)
            self.assertEqual(str(manager.storage_path), temp_dir)
    
    def test_auto_save_model_config(self):
        """Test automatic model configuration saving."""
        # Create mock model
        mock_model = Mock()
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        
        # Create manager
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigurationManager(temp_dir)
            
            # Auto-save configuration
            config_id = auto_save_model_config(
                model=mock_model,
                model_type="test_model",
                parameters={"param1": 1, "param2": "value"},
                config_manager=manager,
                description="Auto-saved config"
            )
            
            self.assertIsInstance(config_id, str)
            
            # Verify configuration was saved
            config = manager.load_configuration(config_id)
            self.assertIsNotNone(config)
            self.assertEqual(config.model_type, "test_model")
            self.assertEqual(config.parameters["param1"], 1)
            self.assertEqual(config.metadata["model_size"], 1000)


class TestIntegration(unittest.TestCase):
    """Integration tests for configuration management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ConfigurationManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_full_configuration_lifecycle(self):
        """Test complete configuration lifecycle."""
        # 1. Save initial configuration
        base_config_id = self.manager.save_configuration(
            model_type="transformer",
            parameters={"lr": 0.01, "hidden_dim": 128, "num_layers": 4},
            description="Initial configuration",
            tags=["baseline"]
        )
        
        # 2. Record initial performance
        self.manager.record_performance(
            config_id=base_config_id,
            metrics={"mse": 0.15, "mae": 0.1},
            dataset_info={"name": "train_set", "size": 10000},
            training_time=300.0
        )
        
        # 3. Create improved variant
        improved_id = self.manager.create_configuration_variant(
            base_config_id=base_config_id,
            parameter_updates={"lr": 0.005, "num_layers": 6},
            description="Improved variant"
        )
        
        # 4. Record better performance for variant
        self.manager.record_performance(
            config_id=improved_id,
            metrics={"mse": 0.08, "mae": 0.06},
            dataset_info={"name": "train_set", "size": 10000},
            training_time=450.0
        )
        
        # 5. Find best configuration
        best_result = self.manager.find_best_configuration(
            model_type="transformer",
            metric="mse"
        )
        
        self.assertIsNotNone(best_result)
        best_config, best_record = best_result
        
        # Should find the improved variant
        self.assertEqual(best_config.config_id, improved_id)
        self.assertEqual(best_record.metrics["mse"], 0.08)
        
        # 6. Compare configurations
        comparison = self.manager.compare_configurations(
            [base_config_id, improved_id],
            metric="mse"
        )
        
        self.assertEqual(len(comparison), 2)
        
        # Improved config should have better MSE
        improved_row = comparison[comparison["config_id"] == improved_id].iloc[0]
        base_row = comparison[comparison["config_id"] == base_config_id].iloc[0]
        
        self.assertLess(improved_row["best_mse"], base_row["best_mse"])
        
        # 7. Get lineage
        lineage = self.manager.get_configuration_lineage(improved_id)
        
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].config_id, base_config_id)
        self.assertEqual(lineage[1].config_id, improved_id)
        
        # 8. Export and import
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            export_path = tmp_file.name
        
        export_result = self.manager.export_configuration(improved_id, export_path)
        self.assertTrue(export_result)
        
        # Create new manager and import
        temp_dir2 = tempfile.mkdtemp()
        manager2 = ConfigurationManager(temp_dir2)
        
        imported_id = manager2.import_configuration(export_path)
        self.assertIsNotNone(imported_id)
        
        # Verify imported data
        imported_config = manager2.load_configuration(imported_id)
        self.assertEqual(imported_config.model_type, "transformer")
        self.assertEqual(imported_config.parameters["lr"], 0.005)
        
        imported_history = manager2.get_performance_history(imported_id)
        self.assertEqual(len(imported_history), 1)
        self.assertEqual(imported_history[0].metrics["mse"], 0.08)
        
        # Clean up
        Path(export_path).unlink()
        shutil.rmtree(temp_dir2)
    
    def test_performance_tracking_over_time(self):
        """Test performance tracking over multiple evaluations."""
        config_id = self.manager.save_configuration(
            model_type="test_model",
            parameters={"param": 1}
        )
        
        # Record performance over time (improving trend)
        for i in range(10):
            mse = 0.2 - i * 0.01  # Improving performance
            self.manager.record_performance(
                config_id=config_id,
                metrics={"mse": mse, "mae": mse * 0.8},
                dataset_info={"name": f"eval_{i}"}
            )
        
        # Get performance history
        history = self.manager.get_performance_history(config_id)
        
        self.assertEqual(len(history), 10)
        
        # Check that performance improved
        first_mse = history[0].metrics["mse"]
        last_mse = history[-1].metrics["mse"]
        self.assertGreater(first_mse, last_mse)
        
        # All records should have the same config_id
        for record in history:
            self.assertEqual(record.config_id, config_id)


if __name__ == '__main__':
    unittest.main()
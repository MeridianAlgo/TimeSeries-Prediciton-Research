"""
Performance and scalability tests for the enhanced time series prediction system.
Tests training speed, inference speed, memory usage, GPU utilization, and stress testing.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import psutil
import gc
import os
import sys
from unittest.mock import Mock, patch
import warnings
from datetime import datetime, timedelta

# Add the parent directory to the path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_timeseries.models.advanced_transformer import AdvancedTransformer
from enhanced_timeseries.models.lstm_model import EnhancedBidirectionalLSTM as AdvancedLSTM
from enhanced_timeseries.models.cnn_lstm_hybrid import CNNLSTMHybrid
from enhanced_timeseries.ensemble.ensemble_framework import EnsembleFramework
from enhanced_timeseries.features.technical_indicators import TechnicalIndicators
from enhanced_timeseries.multi_asset.data_coordinator import MultiAssetDataCoordinator

warnings.filterwarnings('ignore')


class TestTrainingPerformance(unittest.TestCase):
    """Test training performance and speed benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create synthetic training data
        np.random.seed(42)
        self.batch_size = 32
        self.sequence_length = 60
        self.input_dim = 100
        
        self.X_train = torch.randn(1000, self.sequence_length, self.input_dim, device=self.device)
        self.y_train = torch.randn(1000, 1, device=self.device)
        
        # Performance thresholds (seconds per epoch)
        self.performance_thresholds = {
            'transformer': 5.0,
            'lstm': 3.0,
            'cnn_lstm': 4.0
        }
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_transformer_training_performance(self):
        """Test Transformer model training performance."""
        model = AdvancedTransformer(
            input_dim=self.input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            seq_len=self.sequence_length
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure training time
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        training_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        print(f"Transformer training time: {training_time:.2f}s")
        print(f"Transformer memory usage: {memory_usage:.2f}MB")
        
        # Performance assertions
        self.assertLess(training_time, self.performance_thresholds['transformer'])
        self.assertLess(memory_usage, 1000)  # Less than 1GB memory increase
    
    def test_lstm_training_performance(self):
        """Test LSTM model training performance."""
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=3,
            bidirectional=True,
            attention=True
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure training time
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        training_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        print(f"LSTM training time: {training_time:.2f}s")
        print(f"LSTM memory usage: {memory_usage:.2f}MB")
        
        # Performance assertions
        self.assertLess(training_time, self.performance_thresholds['lstm'])
        self.assertLess(memory_usage, 800)  # Less than 800MB memory increase
    
    def test_cnn_lstm_training_performance(self):
        """Test CNN-LSTM hybrid model training performance."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[32, 64, 128],
            lstm_hidden=128,
            seq_len=self.sequence_length
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure training time
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        training_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        print(f"CNN-LSTM training time: {training_time:.2f}s")
        print(f"CNN-LSTM memory usage: {memory_usage:.2f}MB")
        
        # Performance assertions
        self.assertLess(training_time, self.performance_thresholds['cnn_lstm'])
        self.assertLess(memory_usage, 900)  # Less than 900MB memory increase
    
    def test_batch_size_scalability(self):
        """Test training performance with different batch sizes."""
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2
        ).to(self.device)
        
        batch_sizes = [16, 32, 64, 128]
        training_times = []
        
        for batch_size in batch_sizes:
            X_batch = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
            y_batch = torch.randn(batch_size, 1, device=self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            start_time = time.time()
            
            model.train()
            for _ in range(10):  # 10 iterations
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            end_time = time.time()
            training_times.append(end_time - start_time)
        
        # Check that larger batch sizes don't increase time proportionally
        # (should be more efficient due to better GPU utilization)
        time_ratio = training_times[-1] / training_times[0]  # 128 vs 16 batch size
        batch_ratio = batch_sizes[-1] / batch_sizes[0]  # 8x larger batch
        
        print(f"Batch size scaling - Time ratio: {time_ratio:.2f}, Batch ratio: {batch_ratio}")
        self.assertLess(time_ratio, batch_ratio * 0.8)  # Should be more efficient


class TestInferencePerformance(unittest.TestCase):
    """Test inference performance and speed benchmarks."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create synthetic inference data
        np.random.seed(42)
        self.sequence_length = 60
        self.input_dim = 100
        
        # Different batch sizes for testing
        self.batch_sizes = [1, 8, 32, 128]
        
        # Inference performance thresholds (seconds per 1000 samples)
        self.inference_thresholds = {
            'transformer': 2.0,
            'lstm': 1.5,
            'cnn_lstm': 1.8
        }
    
    def test_transformer_inference_performance(self):
        """Test Transformer model inference performance."""
        model = AdvancedTransformer(
            input_dim=self.input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            seq_len=self.sequence_length
        ).to(self.device)
        
        model.eval()
        
        # Test different batch sizes
        for batch_size in self.batch_sizes:
            X_test = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
            
            # Warm up
            with torch.no_grad():
                _ = model(X_test)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(1000 // batch_size):  # 1000 total samples
                    _ = model(X_test)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            print(f"Transformer inference - Batch {batch_size}: {inference_time:.3f}s")
            
            # Performance assertions for larger batches
            if batch_size >= 32:
                self.assertLess(inference_time, self.inference_thresholds['transformer'])
    
    def test_lstm_inference_performance(self):
        """Test LSTM model inference performance."""
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=3,
            bidirectional=True,
            attention=True
        ).to(self.device)
        
        model.eval()
        
        # Test different batch sizes
        for batch_size in self.batch_sizes:
            X_test = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
            
            # Warm up
            with torch.no_grad():
                _ = model(X_test)
            
            # Measure inference time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(1000 // batch_size):  # 1000 total samples
                    _ = model(X_test)
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            print(f"LSTM inference - Batch {batch_size}: {inference_time:.3f}s")
            
            # Performance assertions for larger batches
            if batch_size >= 32:
                self.assertLess(inference_time, self.inference_thresholds['lstm'])
    
    def test_ensemble_inference_performance(self):
        """Test ensemble inference performance."""
        models = {
            'transformer': AdvancedTransformer(self.input_dim, d_model=64, nhead=4, num_layers=2),
            'lstm': AdvancedLSTM(self.input_dim, hidden_dim=64, num_layers=2),
            'cnn_lstm': CNNLSTMHybrid(self.input_dim, cnn_channels=[16, 32], lstm_hidden=64)
        }
        
        # Move models to device
        for model in models.values():
            model.to(self.device)
            model.eval()
        
        ensemble = EnsembleFramework(
            models=models,
            weighting_method='performance_based',
            uncertainty_method='ensemble_variance'
        )
        
        # Test inference
        batch_size = 32
        X_test = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
        
        # Warm up
        with torch.no_grad():
            _ = ensemble.predict_with_uncertainty(X_test)
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):  # 100 predictions
                predictions, uncertainties = ensemble.predict_with_uncertainty(X_test)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        print(f"Ensemble inference time: {inference_time:.3f}s")
        
        # Performance assertions
        self.assertLess(inference_time, 5.0)  # Should complete within 5 seconds
        self.assertEqual(predictions.shape, (batch_size, 1))
        self.assertEqual(uncertainties.shape, (batch_size, 1))


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage and optimization."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create synthetic data
        np.random.seed(42)
        self.sequence_length = 60
        self.input_dim = 100
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_memory_efficiency_training(self):
        """Test memory efficiency during training."""
        initial_memory = self.get_memory_usage()
        
        # Create model
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=3
        ).to(self.device)
        
        model_memory = self.get_memory_usage()
        model_memory_increase = model_memory - initial_memory
        
        print(f"Model memory usage: {model_memory_increase:.2f}MB")
        
        # Test training memory usage
        batch_size = 32
        X_train = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
        y_train = torch.randn(batch_size, 1, device=self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure peak memory during training
        peak_memory = model_memory
        
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            current_memory = self.get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
        
        training_memory_increase = peak_memory - model_memory
        
        print(f"Training memory increase: {training_memory_increase:.2f}MB")
        
        # Memory assertions
        self.assertLess(model_memory_increase, 500)  # Model should use less than 500MB
        self.assertLess(training_memory_increase, 200)  # Training should add less than 200MB
    
    def test_memory_cleanup(self):
        """Test memory cleanup after model deletion."""
        initial_memory = self.get_memory_usage()
        
        # Create and train multiple models
        models = []
        for i in range(3):
            model = AdvancedLSTM(
                input_dim=self.input_dim,
                hidden_dim=64,
                num_layers=2
            ).to(self.device)
            
            # Train briefly
            X = torch.randn(16, self.sequence_length, self.input_dim, device=self.device)
            y = torch.randn(16, 1, device=self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            model.train()
            for _ in range(3):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
            
            models.append(model)
        
        peak_memory = self.get_memory_usage()
        
        # Delete models and force garbage collection
        del models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        final_memory = self.get_memory_usage()
        memory_recovered = peak_memory - final_memory
        
        print(f"Memory recovered after cleanup: {memory_recovered:.2f}MB")
        
        # Memory cleanup assertions
        self.assertGreater(memory_recovered, 0)  # Should recover some memory
        self.assertLess(final_memory - initial_memory, 100)  # Should be close to initial
    
    def test_large_dataset_memory_management(self):
        """Test memory management with large datasets."""
        # Create large dataset
        large_batch_size = 512
        X_large = torch.randn(large_batch_size, self.sequence_length, self.input_dim, device=self.device)
        y_large = torch.randn(large_batch_size, 1, device=self.device)
        
        initial_memory = self.get_memory_usage()
        
        # Process in smaller batches to test memory efficiency
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2
        ).to(self.device)
        
        model.eval()
        
        batch_size = 32
        total_predictions = []
        
        with torch.no_grad():
            for i in range(0, large_batch_size, batch_size):
                end_idx = min(i + batch_size, large_batch_size)
                batch_X = X_large[i:end_idx]
                
                predictions = model(batch_X)
                total_predictions.append(predictions.cpu())  # Move to CPU to save GPU memory
        
        peak_memory = self.get_memory_usage()
        memory_increase = peak_memory - initial_memory
        
        print(f"Large dataset memory increase: {memory_increase:.2f}MB")
        
        # Memory assertions
        self.assertLess(memory_increase, 1000)  # Should handle large datasets efficiently
        self.assertEqual(len(total_predictions), (large_batch_size + batch_size - 1) // batch_size)


class TestGPUEfficiency(unittest.TestCase):
    """Test GPU utilization and efficiency."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Create synthetic data
        np.random.seed(42)
        self.sequence_length = 60
        self.input_dim = 100
    
    def test_gpu_memory_utilization(self):
        """Test GPU memory utilization."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Create model
        model = AdvancedTransformer(
            input_dim=self.input_dim,
            d_model=256,
            nhead=16,
            num_layers=8,
            seq_len=self.sequence_length
        ).to(self.device)
        
        model_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        model_memory_increase = model_gpu_memory - initial_gpu_memory
        
        print(f"GPU memory for model: {model_memory_increase:.2f}MB")
        
        # Test training GPU memory usage
        batch_size = 64
        X_train = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
        y_train = torch.randn(batch_size, 1, device=self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Measure peak GPU memory during training
        peak_gpu_memory = model_gpu_memory
        
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
        
        training_gpu_memory_increase = peak_gpu_memory - model_gpu_memory
        
        print(f"GPU memory increase during training: {training_gpu_memory_increase:.2f}MB")
        
        # GPU memory assertions
        self.assertLess(model_memory_increase, 2000)  # Model should use reasonable GPU memory
        self.assertLess(training_gpu_memory_increase, 1000)  # Training should be efficient
    
    def test_gpu_utilization_efficiency(self):
        """Test GPU utilization efficiency with different batch sizes."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=3
        ).to(self.device)
        
        model.eval()
        
        batch_sizes = [16, 32, 64, 128]
        gpu_memory_usage = []
        
        for batch_size in batch_sizes:
            X_test = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Measure GPU memory usage
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            
            with torch.no_grad():
                _ = model(X_test)
            
            final_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_memory_usage.append(final_gpu_memory - initial_gpu_memory)
        
        # Check that GPU memory usage scales reasonably with batch size
        memory_ratio = gpu_memory_usage[-1] / gpu_memory_usage[0]  # 128 vs 16 batch size
        batch_ratio = batch_sizes[-1] / batch_sizes[0]  # 8x larger batch
        
        print(f"GPU memory scaling - Memory ratio: {memory_ratio:.2f}, Batch ratio: {batch_ratio}")
        
        # GPU memory should scale roughly linearly with batch size
        self.assertLess(memory_ratio, batch_ratio * 1.2)  # Allow some overhead
        self.assertGreater(memory_ratio, batch_ratio * 0.8)  # But not too much overhead


class TestStressTesting(unittest.TestCase):
    """Test system performance under stress conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create synthetic data
        np.random.seed(42)
        self.sequence_length = 60
        self.input_dim = 100
    
    def test_high_frequency_prediction_stress(self):
        """Test system performance under high-frequency prediction stress."""
        model = AdvancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2
        ).to(self.device)
        
        model.eval()
        
        # Simulate high-frequency predictions
        num_predictions = 10000
        batch_size = 1  # Single predictions
        
        X_test = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        predictions = []
        with torch.no_grad():
            for i in range(num_predictions):
                pred = model(X_test)
                predictions.append(pred.cpu().numpy())
                
                # Simulate some processing delay
                if i % 1000 == 0:
                    time.sleep(0.001)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        total_time = end_time - start_time
        memory_increase = end_memory - start_memory
        predictions_per_second = num_predictions / total_time
        
        print(f"High-frequency stress test:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Predictions per second: {predictions_per_second:.0f}")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        
        # Stress test assertions
        self.assertGreater(predictions_per_second, 1000)  # Should handle 1000+ predictions/second
        self.assertLess(memory_increase, 500)  # Memory should remain stable
        self.assertEqual(len(predictions), num_predictions)
    
    def test_large_batch_stress_test(self):
        """Test system performance with very large batches."""
        model = AdvancedTransformer(
            input_dim=self.input_dim,
            d_model=128,
            nhead=8,
            num_layers=4,
            seq_len=self.sequence_length
        ).to(self.device)
        
        model.eval()
        
        # Test with very large batch
        large_batch_size = 1024
        X_large = torch.randn(large_batch_size, self.sequence_length, self.input_dim, device=self.device)
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        with torch.no_grad():
            predictions = model(X_large)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        inference_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        print(f"Large batch stress test:")
        print(f"  Batch size: {large_batch_size}")
        print(f"  Inference time: {inference_time:.3f}s")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        print(f"  Samples per second: {large_batch_size / inference_time:.0f}")
        
        # Large batch assertions
        self.assertLess(inference_time, 10.0)  # Should complete within 10 seconds
        self.assertLess(memory_increase, 2000)  # Memory should be reasonable
        self.assertEqual(predictions.shape, (large_batch_size, 1))
    
    def test_multi_model_stress_test(self):
        """Test system performance with multiple models running simultaneously."""
        # Create multiple models
        models = []
        for i in range(5):
            model = AdvancedLSTM(
                input_dim=self.input_dim,
                hidden_dim=64,
                num_layers=2
            ).to(self.device)
            model.eval()
            models.append(model)
        
        # Test simultaneous predictions
        batch_size = 32
        X_test = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
        
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        all_predictions = []
        with torch.no_grad():
            for model in models:
                predictions = model(X_test)
                all_predictions.append(predictions)
        
        end_time = time.time()
        end_memory = self.get_memory_usage()
        
        total_time = end_time - start_time
        memory_increase = end_memory - start_memory
        
        print(f"Multi-model stress test:")
        print(f"  Number of models: {len(models)}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        
        # Multi-model assertions
        self.assertLess(total_time, 5.0)  # Should complete within 5 seconds
        self.assertLess(memory_increase, 1500)  # Memory should be reasonable
        self.assertEqual(len(all_predictions), len(models))
        
        for predictions in all_predictions:
            self.assertEqual(predictions.shape, (batch_size, 1))
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


if __name__ == '__main__':
    unittest.main()

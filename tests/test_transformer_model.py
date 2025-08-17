"""
Unit tests for enhanced transformer model.
"""

import unittest
import torch
import numpy as np
from enhanced_timeseries.models.transformer_model import (
    EnhancedTimeSeriesTransformer, MultiScaleAttention, LearnablePositionalEncoding,
    AdaptiveDropout, EnhancedTransformerBlock, TransformerConfig, create_transformer_model
)


class TestTransformerModel(unittest.TestCase):
    """Test cases for enhanced transformer model."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 4
        self.seq_len = 60
        self.input_dim = 15
        self.d_model = 128
        self.n_heads = 8
        self.num_layers = 4
        
        # Create sample input data
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_multi_scale_attention(self):
        """Test multi-scale attention mechanism."""
        scales = [1, 2, 4]
        attention = MultiScaleAttention(self.d_model, self.n_heads, scales)
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = attention(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check that output is different from input (attention should transform)
        self.assertFalse(torch.allclose(output, x))
        
        # Test with mask
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[:, -10:] = False  # Mask last 10 positions
        
        output_masked = attention(x, mask)
        self.assertEqual(output_masked.shape, (self.batch_size, self.seq_len, self.d_model))
    
    def test_learnable_positional_encoding(self):
        """Test learnable positional encoding."""
        pos_encoding = LearnablePositionalEncoding(self.d_model, max_len=100)
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = pos_encoding(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that positional encoding was added
        self.assertFalse(torch.allclose(output, x))
        
        # Test with different sequence lengths
        x_short = torch.randn(self.batch_size, 30, self.d_model)
        output_short = pos_encoding(x_short)
        self.assertEqual(output_short.shape, x_short.shape)
    
    def test_adaptive_dropout(self):
        """Test adaptive dropout mechanism."""
        dropout = AdaptiveDropout(base_dropout=0.1, uncertainty_factor=0.1)
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test in training mode
        dropout.train()
        output_train = dropout(x)
        self.assertEqual(output_train.shape, x.shape)
        
        # Test in eval mode
        dropout.eval()
        output_eval = dropout(x)
        self.assertEqual(output_eval.shape, x.shape)
        torch.testing.assert_close(output_eval, x)  # Should be identical in eval mode
        
        # Test uncertainty update
        dropout.update_uncertainty(0.5)
        self.assertAlmostEqual(dropout.uncertainty_level.item(), 0.5)
    
    def test_enhanced_transformer_block(self):
        """Test enhanced transformer block."""
        block = EnhancedTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_model * 4,
            scales=[1, 2],
            dropout=0.1
        )
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Test uncertainty update
        block.update_uncertainty(0.3)
        
        # Test with mask
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        output_masked = block(x, mask)
        self.assertEqual(output_masked.shape, x.shape)
    
    def test_enhanced_transformer_model(self):
        """Test complete enhanced transformer model."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers,
            seq_len=self.seq_len,
            scales=[1, 2, 4],
            num_prediction_heads=3
        )
        
        # Test forward pass
        output = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test that model is trainable
        self.assertTrue(any(p.requires_grad for p in model.parameters()))
        
        # Test model info
        info = model.get_model_info()
        self.assertIn('d_model', info)
        self.assertIn('n_heads', info)
        self.assertIn('total_parameters', info)
    
    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=2,  # Smaller for faster testing
            seq_len=self.seq_len,
            num_prediction_heads=3
        )
        
        # Test uncertainty prediction
        prediction, uncertainty = model.predict_with_uncertainty(self.x)
        
        # Check shapes
        self.assertEqual(prediction.shape, (self.batch_size, 1))
        self.assertEqual(uncertainty.shape, (self.batch_size, 1))
        
        # Check that uncertainty is positive
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_monte_carlo_prediction(self):
        """Test Monte Carlo prediction."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,  # Smaller for faster testing
            n_heads=4,
            num_layers=2,
            seq_len=self.seq_len
        )
        
        # Test Monte Carlo prediction
        mean_pred, uncertainty = model.monte_carlo_predict(self.x, n_samples=10)
        
        # Check shapes
        self.assertEqual(mean_pred.shape, (self.batch_size, 1))
        self.assertEqual(uncertainty.shape, (self.batch_size, 1))
        
        # Check that uncertainty is positive
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_attention_weights(self):
        """Test attention weights extraction."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,
            n_heads=4,
            num_layers=2,
            seq_len=self.seq_len,
            scales=[1, 2]
        )
        
        # Get attention weights (placeholder implementation)
        attention_weights = model.get_attention_weights(self.x)
        
        # Check that we get weights for each scale
        self.assertIn('scale_1', attention_weights)
        self.assertIn('scale_2', attention_weights)
    
    def test_transformer_config(self):
        """Test transformer configuration."""
        config = TransformerConfig(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_layers=self.num_layers
        )
        
        # Test config to dict
        config_dict = config.to_dict()
        self.assertIn('input_dim', config_dict)
        self.assertIn('d_model', config_dict)
        
        # Test config from dict
        config2 = TransformerConfig.from_dict(config_dict)
        self.assertEqual(config.input_dim, config2.input_dim)
        self.assertEqual(config.d_model, config2.d_model)
        
        # Test model creation from config
        model = create_transformer_model(config)
        self.assertIsInstance(model, EnhancedTimeSeriesTransformer)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,
            n_heads=4,
            num_layers=2,
            seq_len=self.seq_len
        )
        
        # Create target
        target = torch.randn(self.batch_size, 1)
        
        # Forward pass
        output = model(self.x)
        
        # Calculate loss
        loss = torch.nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(torch.all(param.grad == 0), f"Zero gradient for {name}")
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,
            n_heads=4,
            num_layers=2,
            seq_len=self.seq_len
        )
        
        # Test with different batch sizes
        x_small = torch.randn(2, self.seq_len, self.input_dim)
        output_small = model(x_small)
        self.assertEqual(output_small.shape, (2, 1))
        
        x_large = torch.randn(8, self.seq_len, self.input_dim)
        output_large = model(x_large)
        self.assertEqual(output_large.shape, (8, 1))
        
        # Test with different sequence lengths (should work with positional encoding)
        x_short = torch.randn(self.batch_size, 30, self.input_dim)
        output_short = model(x_short)
        self.assertEqual(output_short.shape, (self.batch_size, 1))
    
    def test_model_modes(self):
        """Test model in training and evaluation modes."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,
            n_heads=4,
            num_layers=2,
            seq_len=self.seq_len
        )
        
        # Test training mode
        model.train()
        output_train = model(self.x)
        self.assertEqual(output_train.shape, (self.batch_size, 1))
        
        # Test evaluation mode
        model.eval()
        output_eval = model(self.x)
        self.assertEqual(output_eval.shape, (self.batch_size, 1))
        
        # Outputs might be different due to dropout
        # but shapes should be the same
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=64,
            n_heads=4,
            num_layers=2,
            seq_len=self.seq_len
        )
        
        # Check that parameters are initialized (not all zeros)
        for name, param in model.named_parameters():
            if param.dim() > 1:  # Skip bias terms
                self.assertFalse(torch.all(param == 0), f"Parameter {name} is all zeros")
    
    def test_memory_efficiency(self):
        """Test memory usage with different configurations."""
        # Small model
        small_model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=32,
            n_heads=2,
            num_layers=1,
            seq_len=self.seq_len
        )
        
        small_params = sum(p.numel() for p in small_model.parameters())
        
        # Large model
        large_model = EnhancedTimeSeriesTransformer(
            input_dim=self.input_dim,
            d_model=128,
            n_heads=8,
            num_layers=4,
            seq_len=self.seq_len
        )
        
        large_params = sum(p.numel() for p in large_model.parameters())
        
        # Large model should have more parameters
        self.assertGreater(large_params, small_params)
        
        # Both should work
        small_output = small_model(self.x)
        large_output = large_model(self.x)
        
        self.assertEqual(small_output.shape, (self.batch_size, 1))
        self.assertEqual(large_output.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()
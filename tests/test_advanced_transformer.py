"""
Unit tests for Advanced Transformer model.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np

from enhanced_timeseries.models.advanced_transformer import (
    LearnablePositionalEncoding, MultiScaleAttention, AdaptiveDropout,
    TransformerBlock, AdvancedTransformer, MultiHorizonTransformer,
    HierarchicalTransformer, create_advanced_transformer,
    count_transformer_parameters
)


class TestLearnablePositionalEncoding(unittest.TestCase):
    """Test LearnablePositionalEncoding class."""
    
    def setUp(self):
        """Set up test environment."""
        self.d_model = 128
        self.max_len = 1000
        self.pos_enc = LearnablePositionalEncoding(self.d_model, self.max_len)
    
    def test_pos_encoding_creation(self):
        """Test positional encoding creation."""
        self.assertEqual(self.pos_enc.d_model, self.d_model)
        self.assertEqual(self.pos_enc.pos_embedding.shape, (self.max_len, self.d_model))
    
    def test_forward_pass(self):
        """Test forward pass through positional encoding."""
        batch_size, seq_len = 4, 50
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        output = self.pos_enc(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        batch_size = 2
        
        for seq_len in [10, 50, 100, 500]:
            x = torch.randn(batch_size, seq_len, self.d_model)
            output = self.pos_enc(x)
            
            self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
    
    def test_sinusoidal_initialization(self):
        """Test that sinusoidal initialization works."""
        # Check that the embedding is not all zeros after initialization
        self.assertFalse(torch.allclose(self.pos_enc.pos_embedding, torch.zeros_like(self.pos_enc.pos_embedding)))
        
        # Check that even and odd positions have different patterns (sin vs cos)
        even_positions = self.pos_enc.pos_embedding[:, 0::2]
        odd_positions = self.pos_enc.pos_embedding[:, 1::2]
        
        # They should be different due to sin/cos initialization
        self.assertFalse(torch.allclose(even_positions, odd_positions, atol=1e-3))


class TestMultiScaleAttention(unittest.TestCase):
    """Test MultiScaleAttention class."""
    
    def setUp(self):
        """Set up test environment."""
        self.d_model = 256
        self.n_heads = 8
        self.scales = [1, 2, 4]
        self.attention = MultiScaleAttention(self.d_model, self.n_heads, self.scales)
    
    def test_attention_creation(self):
        """Test multi-scale attention creation."""
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.n_heads, self.n_heads)
        self.assertEqual(self.attention.scales, self.scales)
        self.assertEqual(len(self.attention.scale_attentions), len(self.scales))
    
    def test_forward_pass(self):
        """Test forward pass through multi-scale attention."""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        output, attention_weights = self.attention(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
        self.assertIsNotNone(attention_weights)
    
    def test_different_scales(self):
        """Test with different scale configurations."""
        scale_configs = [
            [1],
            [1, 2],
            [1, 2, 4, 8]
        ]
        
        for scales in scale_configs:
            # Adjust n_heads to be divisible by number of scales
            n_heads = len(scales) * 2
            attention = MultiScaleAttention(self.d_model, n_heads, scales)
            
            batch_size, seq_len = 2, 16
            x = torch.randn(batch_size, seq_len, self.d_model)
            
            output, weights = attention(x)
            self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
    
    def test_attention_mask(self):
        """Test attention with mask."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        # Create a simple mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        output, weights = self.attention(x, mask)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))


class TestAdaptiveDropout(unittest.TestCase):
    """Test AdaptiveDropout class."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_dropout = 0.1
        self.adaptive_dropout = AdaptiveDropout(self.base_dropout)
    
    def test_dropout_creation(self):
        """Test adaptive dropout creation."""
        self.assertEqual(self.adaptive_dropout.base_dropout, self.base_dropout)
        self.assertIsInstance(self.adaptive_dropout.uncertainty_estimate, nn.Parameter)
    
    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        self.adaptive_dropout.train()
        
        batch_size, seq_len, d_model = 4, 20, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = self.adaptive_dropout(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
    
    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode."""
        self.adaptive_dropout.eval()
        
        batch_size, seq_len, d_model = 4, 20, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = self.adaptive_dropout(x)
        
        # In eval mode, output should be identical to input
        torch.testing.assert_close(output, x)
    
    def test_uncertainty_based_dropout(self):
        """Test dropout with uncertainty input."""
        self.adaptive_dropout.train()
        
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        uncertainty = torch.tensor(0.5)  # High uncertainty
        
        output = self.adaptive_dropout(x, uncertainty)
        
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))


class TestTransformerBlock(unittest.TestCase):
    """Test TransformerBlock class."""
    
    def setUp(self):
        """Set up test environment."""
        self.d_model = 256
        self.n_heads = 8
        self.d_ff = 1024
        self.block = TransformerBlock(self.d_model, self.n_heads, self.d_ff)
    
    def test_block_creation(self):
        """Test Transformer block creation."""
        self.assertIsInstance(self.block.attention, MultiScaleAttention)
        self.assertIsInstance(self.block.feed_forward, nn.Sequential)
        self.assertIsInstance(self.block.norm1, nn.LayerNorm)
        self.assertIsInstance(self.block.norm2, nn.LayerNorm)
    
    def test_forward_pass(self):
        """Test forward pass through Transformer block."""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        output, attention_weights = self.block(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))
        self.assertIsNotNone(attention_weights)
    
    def test_residual_connections(self):
        """Test that residual connections work."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        self.block.eval()  # Reduce randomness
        
        output, _ = self.block(x)
        
        # Output should be different from input but related due to residual connections
        self.assertFalse(torch.allclose(x, output, atol=1e-6))
        
        # The magnitude should be similar due to residual connections
        input_norm = torch.norm(x)
        output_norm = torch.norm(output)
        self.assertLess(abs(input_norm - output_norm) / input_norm, 1.0)
    
    def test_with_uncertainty(self):
        """Test block with uncertainty input."""
        batch_size, seq_len = 2, 16
        x = torch.randn(batch_size, seq_len, self.d_model)
        uncertainty = torch.randn(batch_size, 1, 1) * 0.1
        
        output, weights = self.block(x, uncertainty=uncertainty)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))


class TestAdvancedTransformer(unittest.TestCase):
    """Test AdvancedTransformer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 10
        self.d_model = 128
        self.n_heads = 8
        self.n_layers = 4
        self.output_dim = 1
        
        self.model = AdvancedTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            output_dim=self.output_dim
        )
    
    def test_model_creation(self):
        """Test Advanced Transformer model creation."""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.d_model, self.d_model)
        self.assertEqual(self.model.output_dim, self.output_dim)
        
        self.assertIsInstance(self.model.input_projection, nn.Linear)
        self.assertIsInstance(self.model.pos_encoding, LearnablePositionalEncoding)
        self.assertEqual(len(self.model.transformer_layers), self.n_layers)
    
    def test_forward_pass(self):
        """Test forward pass through Advanced Transformer."""
        batch_size, seq_len = 4, 50
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_dim))
    
    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty estimation."""
        batch_size, seq_len = 2, 30
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        predictions, uncertainty = self.model(x, return_uncertainty=True)
        
        self.assertEqual(predictions.shape, (batch_size, self.output_dim))
        self.assertEqual(uncertainty.shape, (batch_size, 1))
        self.assertTrue(torch.all(uncertainty >= 0))  # Uncertainty should be non-negative
    
    def test_forward_with_attention(self):
        """Test forward pass with attention weights."""
        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        predictions, attention_weights = self.model(x, return_attention=True)
        
        self.assertEqual(predictions.shape, (batch_size, self.output_dim))
        self.assertEqual(attention_weights.shape[0], batch_size)
        self.assertEqual(attention_weights.shape[1], self.n_layers)
    
    def test_forward_with_both(self):
        """Test forward pass with both uncertainty and attention."""
        batch_size, seq_len = 2, 15
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        predictions, uncertainty, attention = self.model(
            x, return_uncertainty=True, return_attention=True
        )
        
        self.assertEqual(predictions.shape, (batch_size, self.output_dim))
        self.assertEqual(uncertainty.shape, (batch_size, 1))
        self.assertIsNotNone(attention)
    
    def test_predict_with_uncertainty(self):
        """Test Monte Carlo uncertainty prediction."""
        batch_size, seq_len = 2, 25
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        mean_pred, uncertainty = self.model.predict_with_uncertainty(x, n_samples=10)
        
        self.assertEqual(mean_pred.shape, (batch_size, self.output_dim))
        self.assertEqual(uncertainty.shape, (batch_size, self.output_dim))
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_get_attention_maps(self):
        """Test attention map extraction."""
        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        attention_maps = self.model.get_attention_maps(x)
        
        self.assertEqual(attention_maps.shape[0], batch_size)
        self.assertEqual(attention_maps.shape[1], self.n_layers)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        batch_size, seq_len = 2, 30
        x = torch.randn(batch_size, seq_len, self.input_dim, requires_grad=True)
        target = torch.randn(batch_size, self.output_dim)
        
        # Forward pass
        output = self.model(x)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in self.model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        batch_size = 2
        
        for seq_len in [10, 25, 50, 100]:
            x = torch.randn(batch_size, seq_len, self.input_dim)
            output = self.model(x)
            
            self.assertEqual(output.shape, (batch_size, self.output_dim))


class TestMultiHorizonTransformer(unittest.TestCase):
    """Test MultiHorizonTransformer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 8
        self.prediction_horizons = [1, 3, 5]
        
        self.model = MultiHorizonTransformer(
            input_dim=self.input_dim,
            prediction_horizons=self.prediction_horizons,
            d_model=128,
            n_heads=8,
            n_layers=2
        )
    
    def test_model_creation(self):
        """Test multi-horizon model creation."""
        self.assertEqual(self.model.prediction_horizons, self.prediction_horizons)
        self.assertEqual(len(self.model.prediction_heads), len(self.prediction_horizons))
        
        # Check that each horizon has a prediction head
        for horizon in self.prediction_horizons:
            self.assertIn(str(horizon), self.model.prediction_heads)
    
    def test_forward_all_horizons(self):
        """Test forward pass for all horizons."""
        batch_size, seq_len = 4, 40
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        predictions = self.model(x)
        
        self.assertIsInstance(predictions, dict)
        
        # Check predictions for each horizon
        for horizon in self.prediction_horizons:
            key = f'horizon_{horizon}'
            self.assertIn(key, predictions)
            self.assertEqual(predictions[key].shape, (batch_size, horizon))
    
    def test_forward_specific_horizon(self):
        """Test forward pass for specific horizon."""
        batch_size, seq_len = 2, 30
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        horizon = 3
        predictions = self.model(x, horizon=horizon)
        
        self.assertEqual(predictions.shape, (batch_size, horizon))
    
    def test_invalid_horizon(self):
        """Test with invalid horizon."""
        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        with self.assertRaises(ValueError):
            self.model(x, horizon=99)  # Invalid horizon


class TestHierarchicalTransformer(unittest.TestCase):
    """Test HierarchicalTransformer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 6
        self.d_model = 192  # Divisible by 3 scales
        self.scales = [1, 2, 4]
        
        self.model = HierarchicalTransformer(
            input_dim=self.input_dim,
            d_model=self.d_model,
            n_heads=6,  # Divisible by 3 scales
            scales=self.scales,
            output_dim=1
        )
    
    def test_model_creation(self):
        """Test hierarchical model creation."""
        self.assertEqual(len(self.model.scale_transformers), len(self.scales))
        self.assertIsInstance(self.model.cross_scale_attention, nn.MultiheadAttention)
    
    def test_forward_pass(self):
        """Test forward pass through hierarchical model."""
        batch_size, seq_len = 4, 32  # Use power of 2 for clean downsampling
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, 1))
    
    def test_different_scales(self):
        """Test with different scale configurations."""
        scale_configs = [
            [1, 2],
            [1, 4, 8]
        ]
        
        for scales in scale_configs:
            # Adjust d_model to be divisible by number of scales
            d_model = len(scales) * 64
            n_heads = len(scales) * 2
            
            model = HierarchicalTransformer(
                input_dim=self.input_dim,
                d_model=d_model,
                n_heads=n_heads,
                scales=scales,
                output_dim=1
            )
            
            batch_size, seq_len = 2, 32
            x = torch.randn(batch_size, seq_len, self.input_dim)
            
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 1))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_advanced_transformer(self):
        """Test creating Advanced Transformer from configuration."""
        config = {
            'input_dim': 15,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'output_dim': 2,
            'dropout': 0.1
        }
        
        model = create_advanced_transformer(config)
        
        self.assertIsInstance(model, AdvancedTransformer)
        self.assertEqual(model.input_dim, config['input_dim'])
        self.assertEqual(model.d_model, config['d_model'])
        self.assertEqual(model.output_dim, config['output_dim'])
    
    def test_create_transformer_with_defaults(self):
        """Test creating model with default configuration."""
        config = {
            'input_dim': 5
        }
        
        model = create_advanced_transformer(config)
        
        self.assertIsInstance(model, AdvancedTransformer)
        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.output_dim, 1)  # Default
    
    def test_count_transformer_parameters(self):
        """Test parameter counting function."""
        model = AdvancedTransformer(
            input_dim=10,
            d_model=128,
            n_heads=8,
            n_layers=4,
            output_dim=1
        )
        
        param_counts = count_transformer_parameters(model)
        
        self.assertIsInstance(param_counts, dict)
        self.assertIn('total', param_counts)
        self.assertIn('input_projection', param_counts)
        self.assertIn('pos_encoding', param_counts)
        self.assertIn('transformer_layers', param_counts)
        
        # Total should be positive
        self.assertGreater(param_counts['total'], 0)
    
    def test_model_training_mode(self):
        """Test model in training vs evaluation mode."""
        model = AdvancedTransformer(
            input_dim=5,
            d_model=64,
            n_heads=4,
            n_layers=2,
            output_dim=1
        )
        
        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, 5)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should have same shape
        self.assertEqual(output_train.shape, output_eval.shape)
        self.assertEqual(output_train.shape, (batch_size, 1))


class TestModelIntegration(unittest.TestCase):
    """Integration tests for Transformer models."""
    
    def test_transformer_ensemble_compatibility(self):
        """Test that Transformer works with ensemble framework."""
        from enhanced_timeseries.ensemble.ensemble_framework import ModelWrapper
        
        # Create Transformer model
        transformer = AdvancedTransformer(
            input_dim=5,
            d_model=64,
            n_heads=4,
            n_layers=2,
            output_dim=1
        )
        
        # Wrap for ensemble
        wrapped_model = ModelWrapper(transformer, "transformer_model")
        
        # Test prediction
        batch_size, seq_len = 3, 25
        x = torch.randn(batch_size, seq_len, 5)
        
        pred = wrapped_model.predict(x)
        self.assertEqual(pred.shape, (batch_size, 1))
        
        # Test uncertainty prediction
        pred, unc = wrapped_model.predict_with_uncertainty(x)
        self.assertEqual(pred.shape, (batch_size, 1))
        self.assertEqual(unc.shape, (batch_size, 1))
    
    def test_multi_scale_attention_scales(self):
        """Test that multi-scale attention handles different scales properly."""
        d_model = 256
        n_heads = 8
        
        # Test with various scale configurations
        scale_configs = [
            [1],
            [1, 2],
            [1, 2, 4],
            [1, 2, 4, 8]
        ]
        
        for scales in scale_configs:
            attention = MultiScaleAttention(d_model, n_heads, scales)
            
            batch_size, seq_len = 2, 32
            x = torch.randn(batch_size, seq_len, d_model)
            
            output, weights = attention(x)
            
            self.assertEqual(output.shape, (batch_size, seq_len, d_model))
            self.assertIsNotNone(weights)
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large sequences."""
        model = AdvancedTransformer(
            input_dim=10,
            d_model=128,
            n_heads=8,
            n_layers=2,
            output_dim=1
        )
        
        # Test with progressively larger sequences
        for seq_len in [50, 100, 200]:
            batch_size = max(1, 100 // seq_len)  # Adjust batch size
            x = torch.randn(batch_size, seq_len, 10)
            
            # Should not raise memory errors
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()
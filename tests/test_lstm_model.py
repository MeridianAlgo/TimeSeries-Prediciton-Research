"""
Unit tests for enhanced LSTM model.
"""

import unittest
import torch
import numpy as np
from enhanced_timeseries.models.lstm_model import (
    EnhancedBidirectionalLSTM, VariationalDropout, AttentionMechanism,
    SkipConnectionLSTM, LSTMConfig, create_lstm_model
)


class TestLSTMModel(unittest.TestCase):
    """Test cases for enhanced LSTM model."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 4
        self.seq_len = 60
        self.input_dim = 15
        self.hidden_dim = 128
        self.num_layers = 3
        
        # Create sample input data
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_variational_dropout(self):
        """Test variational dropout mechanism."""
        dropout = VariationalDropout(dropout_rate=0.5)
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Test in training mode
        dropout.train()
        output_train = dropout(x)
        self.assertEqual(output_train.shape, x.shape)
        
        # Test in eval mode (should be identical to input)
        dropout.eval()
        output_eval = dropout(x)
        self.assertEqual(output_eval.shape, x.shape)
        torch.testing.assert_close(output_eval, x)
        
        # Test with zero dropout
        dropout_zero = VariationalDropout(dropout_rate=0.0)
        dropout_zero.train()
        output_zero = dropout_zero(x)
        torch.testing.assert_close(output_zero, x)
    
    def test_attention_mechanism_additive(self):
        """Test additive attention mechanism."""
        attention = AttentionMechanism(self.hidden_dim, attention_type='additive')
        
        # Test input
        lstm_outputs = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Forward pass
        attended_output, attention_weights = attention(lstm_outputs)
        
        # Check output shapes
        self.assertEqual(attended_output.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
        
        # Check attention weights sum to 1
        torch.testing.assert_close(
            torch.sum(attention_weights, dim=1),
            torch.ones(self.batch_size),
            atol=1e-6, rtol=1e-6
        )
        
        # Test with mask
        mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[:, -10:] = False  # Mask last 10 positions
        
        attended_output_masked, attention_weights_masked = attention(lstm_outputs, mask)
        self.assertEqual(attended_output_masked.shape, (self.batch_size, self.hidden_dim))
        
        # Masked positions should have zero attention
        self.assertTrue(torch.all(attention_weights_masked[:, -10:] == 0))
    
    def test_attention_mechanism_multiplicative(self):
        """Test multiplicative attention mechanism."""
        attention = AttentionMechanism(self.hidden_dim, attention_type='multiplicative')
        
        # Test input
        lstm_outputs = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Forward pass
        attended_output, attention_weights = attention(lstm_outputs)
        
        # Check output shapes
        self.assertEqual(attended_output.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
        
        # Check attention weights sum to 1
        torch.testing.assert_close(
            torch.sum(attention_weights, dim=1),
            torch.ones(self.batch_size),
            atol=1e-6, rtol=1e-6
        )
    
    def test_attention_mechanism_scaled_dot_product(self):
        """Test scaled dot-product attention mechanism."""
        attention = AttentionMechanism(self.hidden_dim, attention_type='scaled_dot_product')
        
        # Test input
        lstm_outputs = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        
        # Forward pass
        attended_output, attention_weights = attention(lstm_outputs)
        
        # Check output shapes
        self.assertEqual(attended_output.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
        
        # Check attention weights sum to 1
        torch.testing.assert_close(
            torch.sum(attention_weights, dim=1),
            torch.ones(self.batch_size),
            atol=1e-6, rtol=1e-6
        )
    
    def test_skip_connection_lstm(self):
        """Test LSTM layer with skip connections."""
        lstm_layer = SkipConnectionLSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            dropout=0.1
        )
        
        # Test input
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Forward pass
        output, hidden = lstm_layer(x)
        
        # Check output shape (bidirectional LSTM doubles hidden size)
        expected_output_size = self.hidden_dim * 2
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, expected_output_size))
        
        # Check hidden state shapes
        h_n, c_n = hidden
        self.assertEqual(h_n.shape, (2, self.batch_size, self.hidden_dim))  # 2 for bidirectional
        self.assertEqual(c_n.shape, (2, self.batch_size, self.hidden_dim))
        
        # Test with matching input/output sizes (no projection needed)
        lstm_layer_no_proj = SkipConnectionLSTM(
            input_size=self.hidden_dim * 2,
            hidden_size=self.hidden_dim,
            dropout=0.1
        )
        
        x_large = torch.randn(self.batch_size, self.seq_len, self.hidden_dim * 2)
        output_no_proj, _ = lstm_layer_no_proj(x_large)
        self.assertEqual(output_no_proj.shape, (self.batch_size, self.seq_len, self.hidden_dim * 2))
    
    def test_enhanced_bidirectional_lstm(self):
        """Test complete enhanced bidirectional LSTM model."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.2,
            attention_type='additive',
            num_prediction_heads=3,
            use_skip_connections=True
        )
        
        # Test forward pass
        output = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test that model is trainable
        self.assertTrue(any(p.requires_grad for p in model.parameters()))
        
        # Test model info
        info = model.get_model_info()
        self.assertIn('hidden_dim', info)
        self.assertIn('num_layers', info)
        self.assertIn('attention_type', info)
        self.assertIn('total_parameters', info)
    
    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,  # Smaller for faster testing
            num_layers=2,
            num_prediction_heads=3
        )
        
        # Test uncertainty prediction
        prediction, uncertainty = model.predict_with_uncertainty(self.x)
        
        # Check shapes
        self.assertEqual(prediction.shape, (self.batch_size, 1))
        self.assertEqual(uncertainty.shape, (self.batch_size, 1))
        
        # Check that uncertainty is positive
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_attention_weights_extraction(self):
        """Test attention weights extraction."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2,
            attention_type='additive'
        )
        
        # Get attention weights
        attention_weights = model.get_attention_weights(self.x)
        
        # Check shape
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
        
        # Check that weights sum to 1
        torch.testing.assert_close(
            torch.sum(attention_weights, dim=1),
            torch.ones(self.batch_size),
            atol=1e-6, rtol=1e-6
        )
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=2
        )
        
        # Get feature importance
        feature_importance = model.get_feature_importance(self.x)
        
        # Check shape
        self.assertEqual(feature_importance.shape, (self.batch_size, self.input_dim))
        
        # Check that importance scores are non-negative
        self.assertTrue(torch.all(feature_importance >= 0))
    
    def test_monte_carlo_prediction(self):
        """Test Monte Carlo prediction."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,  # Small for faster testing
            num_layers=1,
            dropout=0.2
        )
        
        # Test Monte Carlo prediction
        mean_pred, uncertainty = model.monte_carlo_predict(self.x, n_samples=10)
        
        # Check shapes
        self.assertEqual(mean_pred.shape, (self.batch_size, 1))
        self.assertEqual(uncertainty.shape, (self.batch_size, 1))
        
        # Check that uncertainty is positive
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_hidden_states_extraction(self):
        """Test hidden states extraction."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=64,
            num_layers=3
        )
        
        # Get hidden states
        hidden_states = model.get_hidden_states(self.x)
        
        # Check that we get states from all layers
        self.assertEqual(len(hidden_states), 3)
        
        # Check shapes
        for i, hidden_state in enumerate(hidden_states):
            expected_size = 64 * 2  # bidirectional
            self.assertEqual(hidden_state.shape, (self.batch_size, self.seq_len, expected_size))
    
    def test_different_attention_types(self):
        """Test different attention mechanisms."""
        attention_types = ['additive', 'multiplicative', 'scaled_dot_product']
        
        for attention_type in attention_types:
            model = EnhancedBidirectionalLSTM(
                input_dim=self.input_dim,
                hidden_dim=32,
                num_layers=1,
                attention_type=attention_type
            )
            
            # Test forward pass
            output = model(self.x)
            self.assertEqual(output.shape, (self.batch_size, 1))
            
            # Test attention weights
            attention_weights = model.get_attention_weights(self.x)
            self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_len))
    
    def test_skip_connections_vs_regular(self):
        """Test skip connections vs regular LSTM."""
        # Model with skip connections
        model_skip = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2,
            use_skip_connections=True
        )
        
        # Model without skip connections
        model_regular = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2,
            use_skip_connections=False
        )
        
        # Both should work
        output_skip = model_skip(self.x)
        output_regular = model_regular(self.x)
        
        self.assertEqual(output_skip.shape, (self.batch_size, 1))
        self.assertEqual(output_regular.shape, (self.batch_size, 1))
        
        # Outputs should be different due to different architectures
        self.assertFalse(torch.allclose(output_skip, output_regular))
    
    def test_lstm_config(self):
        """Test LSTM configuration."""
        config = LSTMConfig(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            attention_type='multiplicative'
        )
        
        # Test config to dict
        config_dict = config.to_dict()
        self.assertIn('input_dim', config_dict)
        self.assertIn('hidden_dim', config_dict)
        self.assertIn('attention_type', config_dict)
        
        # Test config from dict
        config2 = LSTMConfig.from_dict(config_dict)
        self.assertEqual(config.input_dim, config2.input_dim)
        self.assertEqual(config.hidden_dim, config2.hidden_dim)
        self.assertEqual(config.attention_type, config2.attention_type)
        
        # Test model creation from config
        model = create_lstm_model(config)
        self.assertIsInstance(model, EnhancedBidirectionalLSTM)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2
        )
        
        # Create target
        target = torch.randn(self.batch_size, 1)
        
        # Forward pass
        output = model(self.x)
        
        # Calculate loss
        loss = torch.nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are non-zero
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                # Allow some parameters to have zero gradients (e.g., unused bias terms)
                if param.grad.numel() > 1:
                    self.assertFalse(torch.all(param.grad == 0), f"All gradients zero for {name}")
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2
        )
        
        # Test with different batch sizes
        x_small = torch.randn(2, self.seq_len, self.input_dim)
        output_small = model(x_small)
        self.assertEqual(output_small.shape, (2, 1))
        
        x_large = torch.randn(8, self.seq_len, self.input_dim)
        output_large = model(x_large)
        self.assertEqual(output_large.shape, (8, 1))
        
        # Test with different sequence lengths
        x_short = torch.randn(self.batch_size, 30, self.input_dim)
        output_short = model(x_short)
        self.assertEqual(output_short.shape, (self.batch_size, 1))
        
        x_long = torch.randn(self.batch_size, 100, self.input_dim)
        output_long = model(x_long)
        self.assertEqual(output_long.shape, (self.batch_size, 1))
    
    def test_model_modes(self):
        """Test model in training and evaluation modes."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2,
            dropout=0.2
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
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        model = EnhancedBidirectionalLSTM(
            input_dim=self.input_dim,
            hidden_dim=32,
            num_layers=2
        )
        
        # Check LSTM parameter initialization
        for name, param in model.named_parameters():
            if 'lstm' in name and 'weight' in name:
                # Weights should not be all zeros
                self.assertFalse(torch.all(param == 0), f"Parameter {name} is all zeros")
            elif 'lstm' in name and 'bias_ih' in name:
                # Forget gate bias should be initialized to 1
                hidden_size = param.size(0) // 4
                forget_bias = param[hidden_size:2*hidden_size]
                # Allow some tolerance for initialization
                self.assertTrue(torch.all(forget_bias > 0.5), f"Forget gate bias not properly initialized in {name}")


if __name__ == '__main__':
    unittest.main()
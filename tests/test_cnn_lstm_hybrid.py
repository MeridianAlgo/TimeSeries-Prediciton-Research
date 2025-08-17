"""
Unit tests for CNN-LSTM hybrid model.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from enhanced_timeseries.models.cnn_lstm_hybrid import (
    MultiScaleConv1D, ResidualConvBlock, AttentionPooling,
    CNNFeatureExtractor, EnhancedLSTM, FeatureFusionLayer,
    CNNLSTMHybrid, MultiHorizonCNNLSTM, create_cnn_lstm_model,
    count_model_parameters
)


class TestMultiScaleConv1D(unittest.TestCase):
    """Test MultiScaleConv1D class."""
    
    def setUp(self):
        """Set up test environment."""
        self.in_channels = 10
        self.out_channels = 32
        self.kernel_sizes = [3, 5, 7]
        self.dilation_rates = [1, 2]
        
        self.conv = MultiScaleConv1D(
            self.in_channels, self.out_channels, 
            self.kernel_sizes, self.dilation_rates
        )
    
    def test_conv_creation(self):
        """Test multi-scale convolution creation."""
        self.assertEqual(self.conv.kernel_sizes, self.kernel_sizes)
        self.assertEqual(self.conv.dilation_rates, self.dilation_rates)
        
        # Should have conv layers for each kernel size * dilation rate combination
        expected_layers = len(self.kernel_sizes) * len(self.dilation_rates)
        self.assertEqual(len(self.conv.conv_layers), expected_layers)
    
    def test_forward_pass(self):
        """Test forward pass through multi-scale convolution."""
        batch_size, seq_len = 4, 100
        x = torch.randn(batch_size, self.in_channels, seq_len)
        
        output = self.conv(x)
        
        expected_out_channels = self.out_channels * len(self.kernel_sizes)
        self.assertEqual(output.shape, (batch_size, expected_out_channels, seq_len))
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        batch_size = 2
        
        for seq_len in [50, 100, 200]:
            x = torch.randn(batch_size, self.in_channels, seq_len)
            output = self.conv(x)
            
            expected_out_channels = self.out_channels * len(self.kernel_sizes)
            self.assertEqual(output.shape, (batch_size, expected_out_channels, seq_len))


class TestResidualConvBlock(unittest.TestCase):
    """Test ResidualConvBlock class."""
    
    def setUp(self):
        """Set up test environment."""
        self.channels = 64
        self.block = ResidualConvBlock(self.channels)
    
    def test_block_creation(self):
        """Test residual block creation."""
        self.assertIsInstance(self.block.conv1, nn.Conv1d)
        self.assertIsInstance(self.block.conv2, nn.Conv1d)
        self.assertIsInstance(self.block.bn1, nn.BatchNorm1d)
        self.assertIsInstance(self.block.bn2, nn.BatchNorm1d)
    
    def test_forward_pass(self):
        """Test forward pass through residual block."""
        batch_size, seq_len = 4, 50
        x = torch.randn(batch_size, self.channels, seq_len)
        
        output = self.block(x)
        
        self.assertEqual(output.shape, (batch_size, self.channels, seq_len))
    
    def test_residual_connection(self):
        """Test that residual connection is working."""
        batch_size, seq_len = 2, 30
        x = torch.randn(batch_size, self.channels, seq_len)
        
        # Set to eval mode to reduce randomness
        self.block.eval()
        
        output = self.block(x)
        
        # Output should be different from input but related due to residual connection
        self.assertFalse(torch.allclose(x, output, atol=1e-6))
        
        # The magnitude should be similar due to residual connection
        input_norm = torch.norm(x)
        output_norm = torch.norm(output)
        self.assertLess(abs(input_norm - output_norm) / input_norm, 1.0)


class TestAttentionPooling(unittest.TestCase):
    """Test AttentionPooling class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 128
        self.pooling = AttentionPooling(self.input_dim)
    
    def test_pooling_creation(self):
        """Test attention pooling creation."""
        self.assertIsInstance(self.pooling.attention, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass through attention pooling."""
        batch_size, seq_len = 4, 20
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        output = self.pooling(x)
        
        self.assertEqual(output.shape, (batch_size, self.input_dim))
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        # Get attention weights manually
        attention_weights = self.pooling.attention(x)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Should sum to 1 along sequence dimension
        weight_sums = attention_weights.sum(dim=1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-6)


class TestCNNLSTMHybrid(unittest.TestCase):
    """Test CNNLSTMHybrid class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 15
        self.sequence_length = 100
        self.output_dim = 3
        
        self.model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            output_dim=self.output_dim,
            cnn_filters=[32, 64],
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            fusion_dim=128
        )
    
    def test_model_creation(self):
        """Test CNN-LSTM hybrid model creation."""
        self.assertEqual(self.model.input_dim, self.input_dim)
        self.assertEqual(self.model.sequence_length, self.sequence_length)
        self.assertEqual(self.model.output_dim, self.output_dim)
        
        self.assertIsInstance(self.model.cnn_extractor, CNNFeatureExtractor)
        self.assertIsInstance(self.model.lstm_modeler, EnhancedLSTM)
        self.assertIsInstance(self.model.feature_fusion, FeatureFusionLayer)
    
    def test_forward_pass(self):
        """Test forward pass through hybrid model."""
        batch_size = 4
        x = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_dim))
    
    def test_forward_with_features(self):
        """Test forward pass with feature extraction."""
        batch_size = 2
        x = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        predictions, features = self.model(x, return_features=True)
        
        self.assertEqual(predictions.shape, (batch_size, self.output_dim))
        
        # Check feature dictionary
        expected_keys = ['cnn_features', 'lstm_features', 'lstm_sequence', 'fused_features']
        for key in expected_keys:
            self.assertIn(key, features)
        
        # Check feature shapes
        self.assertEqual(features['cnn_features'].shape[0], batch_size)
        self.assertEqual(features['lstm_features'].shape[0], batch_size)
        self.assertEqual(features['lstm_sequence'].shape, (batch_size, self.sequence_length, self.model.lstm_modeler.output_dim))
    
    def test_different_fusion_methods(self):
        """Test different fusion methods."""
        fusion_methods = ['concat', 'attention']
        
        for method in fusion_methods:
            model = CNNLSTMHybrid(
                input_dim=self.input_dim,
                sequence_length=50,
                output_dim=1,
                cnn_filters=[32, 64],
                lstm_hidden_dim=64,
                fusion_method=method
            )
            
            batch_size = 2
            x = torch.randn(batch_size, 50, self.input_dim)
            
            output = model(x)
            self.assertEqual(output.shape, (batch_size, 1))
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        batch_size = 2
        x = torch.randn(batch_size, self.sequence_length, self.input_dim, requires_grad=True)
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
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        batch_size = 2
        x = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        importance = self.model.get_feature_importance(x)
        
        self.assertIsInstance(importance, dict)
        self.assertIn('cnn_importance', importance)
        self.assertIn('lstm_importance', importance)
        
        # Importance scores should sum to 1
        total_importance = importance['cnn_importance'] + importance['lstm_importance']
        self.assertAlmostEqual(total_importance, 1.0, places=6)


class TestEnhancedLSTM(unittest.TestCase):
    """Test EnhancedLSTM class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 10
        self.hidden_dim = 64
        self.num_layers = 2
        
        self.lstm = EnhancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            use_attention=True
        )
    
    def test_lstm_creation(self):
        """Test enhanced LSTM creation."""
        self.assertEqual(self.lstm.hidden_dim, self.hidden_dim)
        self.assertEqual(self.lstm.num_layers, self.num_layers)
        self.assertTrue(self.lstm.bidirectional)
        self.assertTrue(self.lstm.use_attention)
    
    def test_forward_pass(self):
        """Test forward pass through enhanced LSTM."""
        batch_size, seq_len = 4, 30
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        sequence_output, pooled_output = self.lstm(x)
        
        expected_lstm_dim = self.hidden_dim * 2  # Bidirectional
        self.assertEqual(sequence_output.shape, (batch_size, seq_len, expected_lstm_dim))
        self.assertEqual(pooled_output.shape, (batch_size, expected_lstm_dim))
    
    def test_unidirectional_lstm(self):
        """Test unidirectional LSTM."""
        lstm_uni = EnhancedLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=1,
            bidirectional=False,
            use_attention=False
        )
        
        batch_size, seq_len = 2, 20
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        sequence_output, pooled_output = lstm_uni(x)
        
        self.assertEqual(sequence_output.shape, (batch_size, seq_len, self.hidden_dim))
        self.assertEqual(pooled_output.shape, (batch_size, self.hidden_dim))


class TestFeatureFusionLayer(unittest.TestCase):
    """Test FeatureFusionLayer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.cnn_dim = 128
        self.lstm_dim = 256
        self.fusion_dim = 512
    
    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = FeatureFusionLayer(
            self.cnn_dim, self.lstm_dim, self.fusion_dim, 'concat'
        )
        
        batch_size = 4
        cnn_features = torch.randn(batch_size, self.cnn_dim)
        lstm_features = torch.randn(batch_size, self.lstm_dim)
        
        output = fusion(cnn_features, lstm_features)
        
        self.assertEqual(output.shape, (batch_size, fusion.output_dim))
    
    def test_attention_fusion(self):
        """Test attention-based fusion."""
        fusion = FeatureFusionLayer(
            self.cnn_dim, self.lstm_dim, self.fusion_dim, 'attention'
        )
        
        batch_size = 4
        cnn_features = torch.randn(batch_size, self.cnn_dim)
        lstm_features = torch.randn(batch_size, self.lstm_dim)
        
        output = fusion(cnn_features, lstm_features)
        
        self.assertEqual(output.shape, (batch_size, fusion.output_dim))
    
    def test_add_fusion(self):
        """Test addition fusion."""
        # For add fusion, dimensions must match
        fusion = FeatureFusionLayer(
            self.cnn_dim, self.cnn_dim, self.fusion_dim, 'add'
        )
        
        batch_size = 4
        cnn_features = torch.randn(batch_size, self.cnn_dim)
        lstm_features = torch.randn(batch_size, self.cnn_dim)
        
        output = fusion(cnn_features, lstm_features)
        
        self.assertEqual(output.shape, (batch_size, fusion.output_dim))
    
    def test_multiply_fusion(self):
        """Test multiplication fusion."""
        # For multiply fusion, dimensions must match
        fusion = FeatureFusionLayer(
            self.cnn_dim, self.cnn_dim, self.fusion_dim, 'multiply'
        )
        
        batch_size = 4
        cnn_features = torch.randn(batch_size, self.cnn_dim)
        lstm_features = torch.randn(batch_size, self.cnn_dim)
        
        output = fusion(cnn_features, lstm_features)
        
        self.assertEqual(output.shape, (batch_size, fusion.output_dim))


class TestMultiHorizonCNNLSTM(unittest.TestCase):
    """Test MultiHorizonCNNLSTM class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 10
        self.sequence_length = 50
        self.prediction_horizons = [1, 3, 5]
        
        self.model = MultiHorizonCNNLSTM(
            input_dim=self.input_dim,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
            cnn_filters=[32, 64],
            lstm_hidden_dim=32
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
        batch_size = 4
        x = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        predictions = self.model(x)
        
        self.assertIsInstance(predictions, dict)
        
        # Check predictions for each horizon
        for horizon in self.prediction_horizons:
            key = f'horizon_{horizon}'
            self.assertIn(key, predictions)
            self.assertEqual(predictions[key].shape, (batch_size, horizon))
    
    def test_forward_specific_horizon(self):
        """Test forward pass for specific horizon."""
        batch_size = 2
        x = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        horizon = 3
        predictions = self.model(x, horizon=horizon)
        
        self.assertEqual(predictions.shape, (batch_size, horizon))
    
    def test_invalid_horizon(self):
        """Test with invalid horizon."""
        batch_size = 2
        x = torch.randn(batch_size, self.sequence_length, self.input_dim)
        
        with self.assertRaises(ValueError):
            self.model(x, horizon=99)  # Invalid horizon


class TestCNNFeatureExtractor(unittest.TestCase):
    """Test CNNFeatureExtractor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.input_dim = 10
        self.cnn_filters = [32, 64, 128]
        
        self.extractor = CNNFeatureExtractor(
            input_dim=self.input_dim,
            cnn_filters=self.cnn_filters
        )
    
    def test_extractor_creation(self):
        """Test CNN feature extractor creation."""
        self.assertEqual(self.extractor.input_dim, self.input_dim)
        self.assertEqual(self.extractor.output_dim, self.cnn_filters[-1] * 2)  # *2 for avg+max pooling
    
    def test_forward_pass(self):
        """Test forward pass through CNN extractor."""
        batch_size, seq_len = 4, 50
        x = torch.randn(batch_size, seq_len, self.input_dim)
        
        output = self.extractor(x)
        
        self.assertEqual(output.shape, (batch_size, self.extractor.output_dim))
    
    def test_different_filter_configurations(self):
        """Test with different filter configurations."""
        filter_configs = [
            [32, 64],
            [64, 128, 256],
            [16, 32, 64, 128]
        ]
        
        for filters in filter_configs:
            extractor = CNNFeatureExtractor(
                input_dim=self.input_dim,
                cnn_filters=filters
            )
            
            batch_size, seq_len = 2, 30
            x = torch.randn(batch_size, seq_len, self.input_dim)
            
            output = extractor(x)
            expected_dim = filters[-1] * 2
            self.assertEqual(output.shape, (batch_size, expected_dim))








class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_create_cnn_lstm_model(self):
        """Test creating CNN-LSTM model from configuration."""
        config = {
            'input_dim': 20,
            'sequence_length': 100,
            'output_dim': 2,
            'cnn_filters': [64, 128],
            'lstm_hidden_dim': 128,
            'fusion_method': 'attention',
            'dropout': 0.2
        }
        
        model = create_cnn_lstm_model(config)
        
        self.assertIsInstance(model, CNNLSTMHybrid)
        self.assertEqual(model.input_dim, config['input_dim'])
        self.assertEqual(model.output_dim, config['output_dim'])
    
    def test_create_model_with_defaults(self):
        """Test creating model with default configuration."""
        config = {
            'input_dim': 5,
            'sequence_length': 50
        }
        
        model = create_cnn_lstm_model(config)
        
        self.assertIsInstance(model, CNNLSTMHybrid)
        self.assertEqual(model.input_dim, 5)
        self.assertEqual(model.output_dim, 1)  # Default
    
    def test_count_model_parameters(self):
        """Test parameter counting function."""
        model = CNNLSTMHybrid(
            input_dim=10,
            sequence_length=50,
            output_dim=1,
            cnn_filters=[32, 64],
            lstm_hidden_dim=64
        )
        
        param_counts = count_model_parameters(model)
        
        self.assertIsInstance(param_counts, dict)
        self.assertIn('total', param_counts)
        self.assertIn('cnn', param_counts)
        self.assertIn('lstm', param_counts)
        self.assertIn('fusion', param_counts)
        
        # Total should equal sum of components
        component_sum = (param_counts.get('cnn', 0) + 
                        param_counts.get('lstm', 0) + 
                        param_counts.get('fusion', 0) + 
                        param_counts.get('output', 0))
        
        self.assertEqual(param_counts['total'], component_sum)
    
    def test_model_training_mode(self):
        """Test model in training vs evaluation mode."""
        model = CNNLSTMHybrid(
            input_dim=5,
            sequence_length=30,
            output_dim=1
        )
        
        batch_size = 2
        x = torch.randn(batch_size, 30, 5)
        
        # Training mode
        model.train()
        output_train = model(x)
        
        # Evaluation mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should have same shape
        self.assertEqual(output_train.shape, output_eval.shape)
        self.assertEqual(output_train.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()
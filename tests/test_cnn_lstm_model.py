"""
Unit tests for CNN-LSTM hybrid model.
"""

import unittest
import torch
import numpy as np
from enhanced_timeseries.models.cnn_lstm_model import (
    CNNLSTMHybrid, MultiScaleConv1D, ResidualConvBlock, TemporalConvolutionNetwork,
    FeatureFusionModule, CNNLSTMConfig, create_cnn_lstm_model
)


class TestCNNLSTMModel(unittest.TestCase):
    """Test cases for CNN-LSTM hybrid model."""
    
    def setUp(self):
        """Set up test parameters."""
        self.batch_size = 4
        self.seq_len = 60
        self.input_dim = 15
        self.cnn_channels = [32, 64, 128]
        self.lstm_hidden = 64
        
        # Create sample input data
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
    
    def test_multi_scale_conv1d(self):
        """Test multi-scale 1D convolution."""
        conv = MultiScaleConv1D(
            in_channels=self.input_dim,
            out_channels=64,
            kernel_sizes=[3, 5, 7],
            dilation_rates=[1, 2]
        )
        
        # Test input (batch_size, channels, seq_len)
        x = torch.randn(self.batch_size, self.input_dim, self.seq_len)
        
        # Forward pass
        output = conv(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 64, self.seq_len))
        
        # Check that output is different from input
        self.assertFalse(torch.allclose(output.mean(), x.mean()))
    
    def test_residual_conv_block(self):
        """Test residual convolution block."""
        channels = 64
        block = ResidualConvBlock(channels, kernel_size=3)
        
        # Test input
        x = torch.randn(self.batch_size, channels, self.seq_len)
        
        # Forward pass
        output = block(x)
        
        # Check output shape (should be same as input)
        self.assertEqual(output.shape, x.shape)
        
        # Check that residual connection works (output should be different from input)
        self.assertFalse(torch.allclose(output, x))
    
    def test_temporal_convolution_network(self):
        """Test Temporal Convolution Network."""
        tcn = TemporalConvolutionNetwork(
            input_channels=64,
            num_channels=[32, 64, 128],
            kernel_size=3
        )
        
        # Test input
        x = torch.randn(self.batch_size, 64, self.seq_len)
        
        # Forward pass
        output = tcn(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 128, self.seq_len))
    
    def test_feature_fusion_concatenate(self):
        """Test feature fusion with concatenation."""
        fusion = FeatureFusionModule(
            cnn_features=128,
            lstm_features=256,
            output_features=512,
            fusion_type='concatenate'
        )
        
        # Test inputs
        cnn_features = torch.randn(self.batch_size, 128)
        lstm_features = torch.randn(self.batch_size, 256)
        
        # Forward pass
        output = fusion(cnn_features, lstm_features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 512))
    
    def test_feature_fusion_add(self):
        """Test feature fusion with addition."""
        fusion = FeatureFusionModule(
            cnn_features=128,
            lstm_features=256,
            output_features=512,
            fusion_type='add'
        )
        
        # Test inputs
        cnn_features = torch.randn(self.batch_size, 128)
        lstm_features = torch.randn(self.batch_size, 256)
        
        # Forward pass
        output = fusion(cnn_features, lstm_features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 512))
    
    def test_feature_fusion_attention(self):
        """Test feature fusion with attention."""
        fusion = FeatureFusionModule(
            cnn_features=128,
            lstm_features=256,
            output_features=512,
            fusion_type='attention'
        )
        
        # Test inputs
        cnn_features = torch.randn(self.batch_size, 128)
        lstm_features = torch.randn(self.batch_size, 256)
        
        # Forward pass
        output = fusion(cnn_features, lstm_features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 512))
    
    def test_cnn_lstm_hybrid_basic(self):
        """Test basic CNN-LSTM hybrid model."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=self.cnn_channels,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=2,
            num_prediction_heads=1
        )
        
        # Test forward pass
        output = model(self.x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Test that model is trainable
        self.assertTrue(any(p.requires_grad for p in model.parameters()))
        
        # Test model info
        info = model.get_model_info()
        self.assertIn('cnn_channels', info)
        self.assertIn('lstm_hidden', info)
        self.assertIn('fusion_type', info)
        self.assertIn('total_parameters', info)
    
    def test_uncertainty_prediction(self):
        """Test uncertainty prediction."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],  # Smaller for faster testing
            lstm_hidden=32,
            lstm_layers=1,
            num_prediction_heads=3
        )
        
        # Test uncertainty prediction
        prediction, uncertainty = model.predict_with_uncertainty(self.x)
        
        # Check shapes
        self.assertEqual(prediction.shape, (self.batch_size, 1))
        self.assertEqual(uncertainty.shape, (self.batch_size, 1))
        
        # Check that uncertainty is positive
        self.assertTrue(torch.all(uncertainty >= 0))
    
    def test_feature_maps_extraction(self):
        """Test feature maps extraction."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            lstm_layers=1
        )
        
        # Get feature maps
        feature_maps = model.get_feature_maps(self.x)
        
        # Check that we get expected feature maps
        expected_keys = ['cnn_layer_0', 'cnn_layer_1', 'cnn_features', 
                        'lstm_output', 'lstm_features', 'fused_features']
        
        for key in expected_keys:
            self.assertIn(key, feature_maps)
        
        # Check shapes
        self.assertEqual(feature_maps['cnn_features'].shape[0], self.batch_size)
        self.assertEqual(feature_maps['lstm_features'].shape[0], self.batch_size)
        self.assertEqual(feature_maps['fused_features'].shape[0], self.batch_size)
    
    def test_different_fusion_types(self):
        """Test different fusion types."""
        fusion_types = ['concatenate', 'add', 'attention']
        
        for fusion_type in fusion_types:
            model = CNNLSTMHybrid(
                input_dim=self.input_dim,
                cnn_channels=[16, 32],
                lstm_hidden=32,
                lstm_layers=1,
                fusion_type=fusion_type
            )
            
            # Test forward pass
            output = model(self.x)
            self.assertEqual(output.shape, (self.batch_size, 1))
            
            # Test uncertainty prediction
            prediction, uncertainty = model.predict_with_uncertainty(self.x)
            self.assertEqual(prediction.shape, (self.batch_size, 1))
            self.assertEqual(uncertainty.shape, (self.batch_size, 1))
    
    def test_with_tcn(self):
        """Test model with TCN enabled."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            tcn_channels=[16, 32],  # Enable TCN
            lstm_layers=1
        )
        
        # Test forward pass
        output = model(self.x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check that TCN feature map is available
        feature_maps = model.get_feature_maps(self.x)
        self.assertIn('tcn_output', feature_maps)
    
    def test_without_tcn(self):
        """Test model without TCN."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            tcn_channels=[],  # Disable TCN
            lstm_layers=1
        )
        
        # Test forward pass
        output = model(self.x)
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check that TCN feature map is not available
        feature_maps = model.get_feature_maps(self.x)
        self.assertNotIn('tcn_output', feature_maps)
    
    def test_cnn_lstm_config(self):
        """Test CNN-LSTM configuration."""
        config = CNNLSTMConfig(
            input_dim=self.input_dim,
            cnn_channels=self.cnn_channels,
            lstm_hidden=self.lstm_hidden,
            fusion_type='attention'
        )
        
        # Test config to dict
        config_dict = config.to_dict()
        self.assertIn('input_dim', config_dict)
        self.assertIn('cnn_channels', config_dict)
        self.assertIn('fusion_type', config_dict)
        
        # Test config from dict
        config2 = CNNLSTMConfig.from_dict(config_dict)
        self.assertEqual(config.input_dim, config2.input_dim)
        self.assertEqual(config.cnn_channels, config2.cnn_channels)
        self.assertEqual(config.fusion_type, config2.fusion_type)
        
        # Test model creation from config
        model = create_cnn_lstm_model(config)
        self.assertIsInstance(model, CNNLSTMHybrid)
    
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            lstm_layers=1
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
                # Allow some parameters to have zero gradients
                if param.grad.numel() > 1:
                    # Check that not all gradients are zero
                    self.assertFalse(torch.all(param.grad == 0), f"All gradients zero for {name}")
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            lstm_layers=1
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
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            lstm_layers=1,
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
        
        # Outputs might be different due to dropout and batch norm
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[16, 32],
            lstm_hidden=32,
            lstm_layers=1
        )
        
        # Check that parameters are initialized (not all zeros)
        for name, param in model.named_parameters():
            if param.dim() > 1:  # Skip bias terms
                self.assertFalse(torch.all(param == 0), f"Parameter {name} is all zeros")
        
        # Check LSTM forget gate bias initialization
        for name, param in model.named_parameters():
            if 'lstm' in name and 'bias_ih' in name:
                hidden_size = param.size(0) // 4
                forget_bias = param[hidden_size:2*hidden_size]
                # Should be initialized to 1
                self.assertTrue(torch.all(forget_bias > 0.5), 
                              f"Forget gate bias not properly initialized in {name}")
    
    def test_different_kernel_sizes(self):
        """Test model with different kernel sizes."""
        kernel_sizes_list = [
            [3, 5],
            [3, 5, 7],
            [3, 5, 7, 9]
        ]
        
        for kernel_sizes in kernel_sizes_list:
            model = CNNLSTMHybrid(
                input_dim=self.input_dim,
                cnn_channels=[16, 32],
                kernel_sizes=kernel_sizes,
                lstm_hidden=32,
                lstm_layers=1
            )
            
            # Test forward pass
            output = model(self.x)
            self.assertEqual(output.shape, (self.batch_size, 1))
    
    def test_memory_efficiency(self):
        """Test memory usage with different configurations."""
        # Small model
        small_model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[8, 16],
            lstm_hidden=16,
            lstm_layers=1
        )
        
        small_params = sum(p.numel() for p in small_model.parameters())
        
        # Large model
        large_model = CNNLSTMHybrid(
            input_dim=self.input_dim,
            cnn_channels=[32, 64, 128],
            lstm_hidden=128,
            lstm_layers=3
        )
        
        large_params = sum(p.numel() for p in large_model.parameters())
        
        # Large model should have more parameters
        self.assertGreater(large_params, small_params)
        
        # Both should work
        small_output = small_model(self.x)
        large_output = large_model(self.x)
        
        self.assertEqual(small_output.shape, (self.batch_size, 1))
        self.assertEqual(large_output.shape, (self.batch_size, 1))
    
    def test_feature_fusion_invalid_type(self):
        """Test feature fusion with invalid type."""
        with self.assertRaises(ValueError):
            FeatureFusionModule(
                cnn_features=128,
                lstm_features=256,
                output_features=512,
                fusion_type='invalid_type'
            )


if __name__ == '__main__':
    unittest.main()
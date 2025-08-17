"""
CNN-LSTM hybrid model for multi-scale pattern recognition in time series.
Implements 1D CNN layers for local pattern extraction and LSTM layers for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolution for pattern extraction at different scales."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_sizes: List[int] = [3, 5, 7, 9], 
                 dilation_rates: List[int] = [1, 2, 4, 8],
                 dropout: float = 0.1):
        """
        Initialize multi-scale convolution.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per scale
            kernel_sizes: List of kernel sizes for different scales
            dilation_rates: List of dilation rates for different scales
            dropout: Dropout rate
        """
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Create convolution layers for each scale
        self.conv_layers = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            for dilation in dilation_rates:
                padding = (kernel_size - 1) * dilation // 2
                conv_layer = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self.conv_layers.append(conv_layer)
        
        # Feature fusion layer
        total_channels = out_channels * len(kernel_sizes) * len(dilation_rates)
        self.fusion_layer = nn.Sequential(
            nn.Conv1d(total_channels, out_channels * len(kernel_sizes), 1),
            nn.BatchNorm1d(out_channels * len(kernel_sizes)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Multi-scale features of shape (batch_size, out_channels * len(kernel_sizes), seq_len)
        """
        # Apply all convolution layers
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(x)
            conv_outputs.append(conv_out)
        
        # Concatenate all outputs
        multi_scale_features = torch.cat(conv_outputs, dim=1)
        
        # Apply fusion layer
        fused_features = self.fusion_layer(multi_scale_features)
        
        return fused_features


class ResidualConvBlock(nn.Module):
    """Residual convolution block for deep CNN architectures."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        """
        Initialize residual convolution block.
        
        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Add residual connection
        out += residual
        out = F.relu(out)
        
        return out


class AttentionPooling(nn.Module):
    """Attention-based pooling for feature aggregation."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        """
        Initialize attention pooling.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention computation
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Pooled tensor of shape (batch_size, input_dim)
        """
        # Calculate attention weights
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        pooled = torch.sum(x * attention_weights, dim=1)  # (batch_size, input_dim)
        
        return pooled


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for local pattern recognition."""
    
    def __init__(self, input_dim: int, cnn_filters: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7], 
                 use_residual: bool = True, dropout: float = 0.1):
        """
        Initialize CNN feature extractor.
        
        Args:
            input_dim: Input feature dimension
            cnn_filters: List of filter sizes for each CNN layer
            kernel_sizes: List of kernel sizes for multi-scale convolution
            use_residual: Whether to use residual connections
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.use_residual = use_residual
        
        # Initial projection to match filter size
        self.input_projection = nn.Conv1d(input_dim, cnn_filters[0], 1)
        
        # Multi-scale convolution layers
        self.conv_layers = nn.ModuleList()
        
        for i, filters in enumerate(cnn_filters):
            if i == 0:
                # First layer uses multi-scale convolution
                conv_layer = MultiScaleConv1D(
                    filters, filters // len(kernel_sizes), 
                    kernel_sizes, dropout=dropout
                )
            else:
                # Subsequent layers use regular convolution
                conv_layer = nn.Sequential(
                    nn.Conv1d(cnn_filters[i-1], filters, 3, padding=1),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            
            self.conv_layers.append(conv_layer)
            
            # Add residual blocks if specified
            if use_residual and i > 0:
                residual_block = ResidualConvBlock(filters, dropout=dropout)
                self.conv_layers.append(residual_block)
        
        # Global pooling options
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Feature dimension after CNN
        self.output_dim = cnn_filters[-1] * 2  # *2 for avg + max pooling
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN features from input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            CNN features of shape (batch_size, output_dim)
        """
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Initial projection
        x = self.input_projection(x)
        
        # Pass through CNN layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Global pooling
        avg_pooled = self.global_avg_pool(x).squeeze(-1)  # (batch_size, filters)
        max_pooled = self.global_max_pool(x).squeeze(-1)  # (batch_size, filters)
        
        # Concatenate pooled features
        cnn_features = torch.cat([avg_pooled, max_pooled], dim=1)
        
        return cnn_features


class EnhancedLSTM(nn.Module):
    """Enhanced LSTM with attention and skip connections."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 dropout: float = 0.1, bidirectional: bool = True, 
                 use_attention: bool = True):
        """
        Initialize enhanced LSTM.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Attention mechanism
        if use_attention:
            self.attention_pooling = AttentionPooling(lstm_output_dim)
            self.output_dim = lstm_output_dim
        else:
            self.output_dim = lstm_output_dim
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through enhanced LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (sequence_output, pooled_output)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Get pooled representation
        if self.use_attention:
            pooled_output = self.attention_pooling(lstm_out)
        else:
            # Use last time step
            pooled_output = lstm_out[:, -1, :]
        
        return lstm_out, pooled_output


class FeatureFusionLayer(nn.Module):
    """Feature fusion layer for combining CNN and LSTM features."""
    
    def __init__(self, cnn_dim: int, lstm_dim: int, fusion_dim: int = 256,
                 fusion_method: str = 'concat', dropout: float = 0.1):
        """
        Initialize feature fusion layer.
        
        Args:
            cnn_dim: CNN feature dimension
            lstm_dim: LSTM feature dimension
            fusion_dim: Fusion layer dimension
            fusion_method: Fusion method ('concat', 'add', 'multiply', 'attention')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            input_dim = cnn_dim + lstm_dim
        elif fusion_method in ['add', 'multiply']:
            # Ensure dimensions match
            assert cnn_dim == lstm_dim, "CNN and LSTM dimensions must match for add/multiply fusion"
            input_dim = cnn_dim
        elif fusion_method == 'attention':
            input_dim = cnn_dim + lstm_dim
            # Attention weights for each modality
            self.cnn_attention = nn.Linear(cnn_dim, 1)
            self.lstm_attention = nn.Linear(lstm_dim, 1)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_dim = fusion_dim // 2
        
    def forward(self, cnn_features: torch.Tensor, lstm_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse CNN and LSTM features.
        
        Args:
            cnn_features: CNN features of shape (batch_size, cnn_dim)
            lstm_features: LSTM features of shape (batch_size, lstm_dim)
            
        Returns:
            Fused features of shape (batch_size, output_dim)
        """
        if self.fusion_method == 'concat':
            fused = torch.cat([cnn_features, lstm_features], dim=1)
        
        elif self.fusion_method == 'add':
            fused = cnn_features + lstm_features
        
        elif self.fusion_method == 'multiply':
            fused = cnn_features * lstm_features
        
        elif self.fusion_method == 'attention':
            # Calculate attention weights
            cnn_weight = torch.sigmoid(self.cnn_attention(cnn_features))
            lstm_weight = torch.sigmoid(self.lstm_attention(lstm_features))
            
            # Normalize weights
            total_weight = cnn_weight + lstm_weight
            cnn_weight = cnn_weight / total_weight
            lstm_weight = lstm_weight / total_weight
            
            # Apply attention weights and concatenate
            weighted_cnn = cnn_features * cnn_weight
            weighted_lstm = lstm_features * lstm_weight
            fused = torch.cat([weighted_cnn, weighted_lstm], dim=1)
        
        # Pass through fusion network
        output = self.fusion_network(fused)
        
        return output


class CNNLSTMHybrid(nn.Module):
    """CNN-LSTM hybrid model for multi-scale time series pattern recognition."""
    
    def __init__(self, 
                 input_dim: int,
                 sequence_length: int,
                 output_dim: int = 1,
                 cnn_filters: List[int] = [64, 128, 256],
                 cnn_kernel_sizes: List[int] = [3, 5, 7],
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 2,
                 fusion_dim: int = 256,
                 fusion_method: str = 'attention',
                 dropout: float = 0.1,
                 use_residual_cnn: bool = True,
                 use_bidirectional_lstm: bool = True,
                 use_lstm_attention: bool = True):
        """
        Initialize CNN-LSTM hybrid model.
        
        Args:
            input_dim: Input feature dimension
            sequence_length: Input sequence length
            output_dim: Output dimension
            cnn_filters: CNN filter sizes
            cnn_kernel_sizes: CNN kernel sizes
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            fusion_dim: Feature fusion dimension
            fusion_method: Feature fusion method
            dropout: Dropout rate
            use_residual_cnn: Whether to use residual CNN blocks
            use_bidirectional_lstm: Whether to use bidirectional LSTM
            use_lstm_attention: Whether to use LSTM attention
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor(
            input_dim=input_dim,
            cnn_filters=cnn_filters,
            kernel_sizes=cnn_kernel_sizes,
            use_residual=use_residual_cnn,
            dropout=dropout
        )
        
        # LSTM temporal modeler
        self.lstm_modeler = EnhancedLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=use_bidirectional_lstm,
            use_attention=use_lstm_attention
        )
        
        # Feature fusion
        self.feature_fusion = FeatureFusionLayer(
            cnn_dim=self.cnn_extractor.output_dim,
            lstm_dim=self.lstm_modeler.output_dim,
            fusion_dim=fusion_dim,
            fusion_method=fusion_method,
            dropout=dropout
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.feature_fusion.output_dim, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through CNN-LSTM hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_features: Whether to return intermediate features
            
        Returns:
            Output predictions, optionally with intermediate features
        """
        # Extract CNN features (local patterns)
        cnn_features = self.cnn_extractor(x)
        
        # Extract LSTM features (temporal dependencies)
        lstm_sequence, lstm_features = self.lstm_modeler(x)
        
        # Fuse features
        fused_features = self.feature_fusion(cnn_features, lstm_features)
        
        # Generate predictions
        predictions = self.output_layers(fused_features)
        
        if return_features:
            features = {
                'cnn_features': cnn_features,
                'lstm_features': lstm_features,
                'lstm_sequence': lstm_sequence,
                'fused_features': fused_features
            }
            return predictions, features
        
        return predictions
    
    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Calculate feature importance using gradient-based attribution.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with feature importance scores
        """
        x.requires_grad_(True)
        
        predictions, features = self.forward(x, return_features=True)
        
        # Calculate gradients
        loss = predictions.sum()
        loss.backward()
        
        # Calculate importance scores
        cnn_importance = torch.norm(features['cnn_features'].grad).item()
        lstm_importance = torch.norm(features['lstm_features'].grad).item()
        
        total_importance = cnn_importance + lstm_importance
        
        return {
            'cnn_importance': cnn_importance / total_importance,
            'lstm_importance': lstm_importance / total_importance,
            'cnn_raw': cnn_importance,
            'lstm_raw': lstm_importance
        }


class MultiHorizonCNNLSTM(nn.Module):
    """Multi-horizon CNN-LSTM model for different prediction horizons."""
    
    def __init__(self, 
                 input_dim: int,
                 sequence_length: int,
                 prediction_horizons: List[int] = [1, 3, 5, 10],
                 **kwargs):
        """
        Initialize multi-horizon CNN-LSTM model.
        
        Args:
            input_dim: Input feature dimension
            sequence_length: Input sequence length
            prediction_horizons: List of prediction horizons
            **kwargs: Additional arguments for CNNLSTMHybrid
        """
        super().__init__()
        
        self.prediction_horizons = prediction_horizons
        
        # Shared feature extraction
        self.shared_model = CNNLSTMHybrid(
            input_dim=input_dim,
            sequence_length=sequence_length,
            output_dim=1,  # Will be overridden by horizon-specific heads
            **kwargs
        )
        
        # Remove the output layer from shared model
        self.shared_model.output_layers = nn.Identity()
        
        # Horizon-specific prediction heads
        self.prediction_heads = nn.ModuleDict()
        
        for horizon in prediction_horizons:
            head = nn.Sequential(
                nn.Linear(self.shared_model.feature_fusion.output_dim, 128),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
                nn.Linear(128, horizon)
            )
            self.prediction_heads[str(horizon)] = head
    
    def forward(self, x: torch.Tensor, horizon: Optional[int] = None) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for multi-horizon prediction.
        
        Args:
            x: Input tensor
            horizon: Specific horizon to predict (if None, predicts all)
            
        Returns:
            Predictions for specified horizon or all horizons
        """
        # Get shared features
        shared_features = self.shared_model(x)
        
        if horizon is not None:
            # Predict for specific horizon
            if str(horizon) not in self.prediction_heads:
                raise ValueError(f"Horizon {horizon} not supported")
            
            return self.prediction_heads[str(horizon)](shared_features)
        
        else:
            # Predict for all horizons
            predictions = {}
            for h in self.prediction_horizons:
                predictions[f'horizon_{h}'] = self.prediction_heads[str(h)](shared_features)
            
            return predictions


# Utility functions
def create_cnn_lstm_model(config: Dict) -> CNNLSTMHybrid:
    """
    Create CNN-LSTM model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured CNN-LSTM model
    """
    return CNNLSTMHybrid(
        input_dim=config.get('input_dim', 1),
        sequence_length=config.get('sequence_length', 100),
        output_dim=config.get('output_dim', 1),
        cnn_filters=config.get('cnn_filters', [64, 128, 256]),
        cnn_kernel_sizes=config.get('cnn_kernel_sizes', [3, 5, 7]),
        lstm_hidden_dim=config.get('lstm_hidden_dim', 128),
        lstm_num_layers=config.get('lstm_num_layers', 2),
        fusion_dim=config.get('fusion_dim', 256),
        fusion_method=config.get('fusion_method', 'attention'),
        dropout=config.get('dropout', 0.1),
        use_residual_cnn=config.get('use_residual_cnn', True),
        use_bidirectional_lstm=config.get('use_bidirectional_lstm', True),
        use_lstm_attention=config.get('use_lstm_attention', True)
    )


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in different parts of the model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    component_params = {}
    
    if hasattr(model, 'cnn_extractor'):
        component_params['cnn'] = sum(p.numel() for p in model.cnn_extractor.parameters() if p.requires_grad)
    
    if hasattr(model, 'lstm_modeler'):
        component_params['lstm'] = sum(p.numel() for p in model.lstm_modeler.parameters() if p.requires_grad)
    
    if hasattr(model, 'feature_fusion'):
        component_params['fusion'] = sum(p.numel() for p in model.feature_fusion.parameters() if p.requires_grad)
    
    if hasattr(model, 'output_layers'):
        component_params['output'] = sum(p.numel() for p in model.output_layers.parameters() if p.requires_grad)
    
    component_params['total'] = total_params
    
    return component_params
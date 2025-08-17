"""
CNN-LSTM hybrid model for multi-scale pattern recognition in time series.
Combines 1D CNNs for local pattern extraction with LSTMs for temporal modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from ..core.interfaces import BaseModel


class MultiScaleConv1D(nn.Module):
    """Multi-scale 1D convolution for capturing patterns at different scales."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7, 9],
                 dilation_rates: List[int] = [1, 2, 4], dropout: float = 0.1):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates
        
        # Create convolution branches for each kernel size and dilation
        self.conv_branches = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            for dilation in dilation_rates:
                padding = (kernel_size - 1) * dilation // 2  # Same padding
                conv_branch = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels // (len(kernel_sizes) * len(dilation_rates)),
                             kernel_size=kernel_size, dilation=dilation, padding=padding),
                    nn.BatchNorm1d(out_channels // (len(kernel_sizes) * len(dilation_rates))),
                    nn.ReLU(),
                    nn.Dropout1d(dropout)
                )
                self.conv_branches.append(conv_branch)
        
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.ReLU()
        )
        
        # Output projection
        total_channels = out_channels + out_channels // 4
        self.output_projection = nn.Conv1d(total_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale convolution.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, seq_len)
        """
        # Apply all convolution branches
        branch_outputs = []
        for conv_branch in self.conv_branches:
            branch_output = conv_branch(x)
            branch_outputs.append(branch_output)
        
        # Concatenate branch outputs
        multi_scale_output = torch.cat(branch_outputs, dim=1)
        
        # Global pooling branch
        global_output = self.global_pool(x)
        global_output = global_output.expand(-1, -1, x.size(2))  # Expand to match sequence length
        
        # Combine all outputs
        combined_output = torch.cat([multi_scale_output, global_output], dim=1)
        
        # Final projection
        output = self.output_projection(combined_output)
        
        return output


class ResidualConvBlock(nn.Module):
    """Residual convolution block with skip connections."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
        self.dropout = nn.Dropout1d(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out = out + residual
        out = self.relu(out)
        
        return out


class TemporalConvolutionNetwork(nn.Module):
    """Temporal Convolution Network (TCN) for sequence modeling."""
    
    def __init__(self, input_channels: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                self._make_tcn_layer(in_channels, out_channels, kernel_size, dilation_size, dropout)
            )
        
        self.network = nn.Sequential(*layers)
        
    def _make_tcn_layer(self, in_channels: int, out_channels: int, kernel_size: int, 
                       dilation: int, dropout: float) -> nn.Module:
        """Create a single TCN layer."""
        padding = (kernel_size - 1) * dilation
        
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN."""
        return self.network(x)


class FeatureFusionModule(nn.Module):
    """Feature fusion module for combining CNN and LSTM features."""
    
    def __init__(self, cnn_features: int, lstm_features: int, output_features: int, 
                 fusion_type: str = 'concatenate'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == 'concatenate':
            self.fusion_layer = nn.Linear(cnn_features + lstm_features, output_features)
        elif fusion_type == 'add':
            # Project to same dimension before adding
            self.cnn_projection = nn.Linear(cnn_features, output_features)
            self.lstm_projection = nn.Linear(lstm_features, output_features)
        elif fusion_type == 'attention':
            # Attention-based fusion
            self.cnn_attention = nn.Linear(cnn_features, 1)
            self.lstm_attention = nn.Linear(lstm_features, 1)
            self.fusion_layer = nn.Linear(cnn_features + lstm_features, output_features)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.layer_norm = nn.LayerNorm(output_features)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, cnn_features: torch.Tensor, lstm_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse CNN and LSTM features.
        
        Args:
            cnn_features: CNN features of shape (batch_size, cnn_features)
            lstm_features: LSTM features of shape (batch_size, lstm_features)
            
        Returns:
            Fused features of shape (batch_size, output_features)
        """
        if self.fusion_type == 'concatenate':
            combined = torch.cat([cnn_features, lstm_features], dim=-1)
            output = self.fusion_layer(combined)
            
        elif self.fusion_type == 'add':
            cnn_proj = self.cnn_projection(cnn_features)
            lstm_proj = self.lstm_projection(lstm_features)
            output = cnn_proj + lstm_proj
            
        elif self.fusion_type == 'attention':
            # Calculate attention weights
            cnn_weight = torch.sigmoid(self.cnn_attention(cnn_features))
            lstm_weight = torch.sigmoid(self.lstm_attention(lstm_features))
            
            # Normalize weights
            total_weight = cnn_weight + lstm_weight
            cnn_weight = cnn_weight / total_weight
            lstm_weight = lstm_weight / total_weight
            
            # Apply attention weights
            weighted_cnn = cnn_features * cnn_weight
            weighted_lstm = lstm_features * lstm_weight
            
            # Combine and project
            combined = torch.cat([weighted_cnn, weighted_lstm], dim=-1)
            output = self.fusion_layer(combined)
        
        # Apply normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class CNNLSTMHybrid(BaseModel):
    """
    CNN-LSTM hybrid model that combines:
    - Multi-scale 1D CNNs for local pattern extraction
    - Temporal Convolution Network (TCN) for hierarchical features
    - Bidirectional LSTM for temporal dependency modeling
    - Feature fusion for combining CNN and LSTM representations
    """
    
    def __init__(self, input_dim: int, cnn_channels: List[int] = [64, 128, 256],
                 kernel_sizes: List[int] = [3, 5, 7], lstm_hidden: int = 256,
                 lstm_layers: int = 2, tcn_channels: List[int] = [64, 128],
                 fusion_type: str = 'attention', dropout: float = 0.1,
                 num_prediction_heads: int = 1):
        super().__init__(input_dim)
        
        self.cnn_channels = cnn_channels
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.fusion_type = fusion_type
        self.num_prediction_heads = num_prediction_heads
        
        # CNN Feature Extractor
        self.cnn_layers = nn.ModuleList()
        
        # First multi-scale convolution
        self.cnn_layers.append(
            MultiScaleConv1D(input_dim, cnn_channels[0], kernel_sizes, dropout=dropout)
        )
        
        # Additional CNN layers with residual blocks
        for i in range(1, len(cnn_channels)):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(cnn_channels[i-1], cnn_channels[i], kernel_size=3, padding=1),
                    nn.BatchNorm1d(cnn_channels[i]),
                    nn.ReLU(),
                    ResidualConvBlock(cnn_channels[i], dropout=dropout),
                    nn.MaxPool1d(kernel_size=2, stride=1, padding=0)  # Slight downsampling
                )
            )
        
        # Temporal Convolution Network
        if tcn_channels:
            self.tcn = TemporalConvolutionNetwork(cnn_channels[-1], tcn_channels, dropout=dropout)
            cnn_output_channels = tcn_channels[-1]
        else:
            self.tcn = None
            cnn_output_channels = cnn_channels[-1]
        
        # Global pooling for CNN features
        self.cnn_global_pool = nn.AdaptiveAvgPool1d(1)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Feature fusion
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        fusion_output_size = 512
        
        self.feature_fusion = FeatureFusionModule(
            cnn_features=cnn_output_channels,
            lstm_features=lstm_output_size,
            output_features=fusion_output_size,
            fusion_type=fusion_type
        )
        
        # Output layers
        self.output_dropout = nn.Dropout(dropout)
        
        # Multiple prediction heads for uncertainty
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_output_size, fusion_output_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_output_size // 2, fusion_output_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_output_size // 4, 1)
            )
            for _ in range(num_prediction_heads)
        ])
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(fusion_output_size, fusion_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_size // 4, 1)
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # CNN Feature Extraction
        # Transpose for 1D convolution: (batch_size, input_dim, seq_len)
        x_cnn = x.transpose(1, 2)
        
        # Pass through CNN layers
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        
        # Pass through TCN if available
        if self.tcn is not None:
            x_cnn = self.tcn(x_cnn)
        
        # Global pooling for CNN features
        cnn_features = self.cnn_global_pool(x_cnn).squeeze(-1)  # (batch_size, cnn_channels)
        
        # LSTM Feature Extraction
        lstm_output, _ = self.lstm(x)  # (batch_size, seq_len, lstm_hidden * 2)
        
        # Use last LSTM output
        lstm_features = lstm_output[:, -1, :]  # (batch_size, lstm_hidden * 2)
        
        # Feature Fusion
        fused_features = self.feature_fusion(cnn_features, lstm_features)
        
        # Apply dropout
        fused_features = self.output_dropout(fused_features)
        
        # Get prediction from first head
        prediction = self.prediction_heads[0](fused_features)
        
        return prediction
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # CNN Feature Extraction
        x_cnn = x.transpose(1, 2)
        
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn)
        
        if self.tcn is not None:
            x_cnn = self.tcn(x_cnn)
        
        cnn_features = self.cnn_global_pool(x_cnn).squeeze(-1)
        
        # LSTM Feature Extraction
        lstm_output, _ = self.lstm(x)
        lstm_features = lstm_output[:, -1, :]
        
        # Feature Fusion
        fused_features = self.feature_fusion(cnn_features, lstm_features)
        fused_features = self.output_dropout(fused_features)
        
        # Get predictions from all heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(fused_features)
            predictions.append(pred)
        
        # Calculate epistemic uncertainty
        if len(predictions) > 1:
            predictions_tensor = torch.stack(predictions, dim=0)
            mean_prediction = torch.mean(predictions_tensor, dim=0)
            epistemic_uncertainty = torch.var(predictions_tensor, dim=0)
        else:
            mean_prediction = predictions[0]
            epistemic_uncertainty = torch.zeros_like(mean_prediction)
        
        # Get aleatoric uncertainty
        log_var = self.uncertainty_head(fused_features)
        aleatoric_uncertainty = torch.exp(log_var)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_prediction, total_uncertainty
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of feature maps from different layers
        """
        feature_maps = {}
        
        # CNN features
        x_cnn = x.transpose(1, 2)
        
        for i, cnn_layer in enumerate(self.cnn_layers):
            x_cnn = cnn_layer(x_cnn)
            feature_maps[f'cnn_layer_{i}'] = x_cnn.detach()
        
        if self.tcn is not None:
            x_cnn = self.tcn(x_cnn)
            feature_maps['tcn_output'] = x_cnn.detach()
        
        cnn_features = self.cnn_global_pool(x_cnn).squeeze(-1)
        feature_maps['cnn_features'] = cnn_features.detach()
        
        # LSTM features
        lstm_output, _ = self.lstm(x)
        feature_maps['lstm_output'] = lstm_output.detach()
        
        lstm_features = lstm_output[:, -1, :]
        feature_maps['lstm_features'] = lstm_features.detach()
        
        # Fused features
        fused_features = self.feature_fusion(cnn_features, lstm_features)
        feature_maps['fused_features'] = fused_features.detach()
        
        return feature_maps
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        
        enhanced_info = {
            **base_info,
            'cnn_channels': self.cnn_channels,
            'kernel_sizes': self.kernel_sizes,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'fusion_type': self.fusion_type,
            'num_prediction_heads': self.num_prediction_heads,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        return enhanced_info


class CNNLSTMConfig:
    """Configuration class for CNN-LSTM hybrid model."""
    
    def __init__(self, **kwargs):
        # Default configuration
        self.input_dim = kwargs.get('input_dim', 15)
        self.cnn_channels = kwargs.get('cnn_channels', [64, 128, 256])
        self.kernel_sizes = kwargs.get('kernel_sizes', [3, 5, 7])
        self.lstm_hidden = kwargs.get('lstm_hidden', 256)
        self.lstm_layers = kwargs.get('lstm_layers', 2)
        self.tcn_channels = kwargs.get('tcn_channels', [64, 128])
        self.fusion_type = kwargs.get('fusion_type', 'attention')
        self.dropout = kwargs.get('dropout', 0.1)
        self.num_prediction_heads = kwargs.get('num_prediction_heads', 3)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_dim': self.input_dim,
            'cnn_channels': self.cnn_channels,
            'kernel_sizes': self.kernel_sizes,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'tcn_channels': self.tcn_channels,
            'fusion_type': self.fusion_type,
            'dropout': self.dropout,
            'num_prediction_heads': self.num_prediction_heads
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CNNLSTMConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_cnn_lstm_model(config: CNNLSTMConfig) -> CNNLSTMHybrid:
    """Factory function to create CNN-LSTM model from config."""
    return CNNLSTMHybrid(**config.to_dict())
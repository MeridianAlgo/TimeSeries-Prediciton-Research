"""
Advanced Transformer model for time series prediction with multi-scale attention,
learnable positional embeddings, and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
from ..core.interfaces import BaseModel


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism for capturing patterns at different time scales."""
    
    def __init__(self, d_model: int, n_heads: int, scales: list = [1, 2, 4], dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Separate attention heads for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads // len(scales), dropout=dropout, batch_first=True)
            for _ in scales
        ])
        
        # Projection layers for each scale
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model // len(scales))
            for _ in scales
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with multi-scale attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        
        for i, (scale, attention, projection) in enumerate(zip(self.scales, self.scale_attentions, self.scale_projections)):
            # Downsample for larger scales
            if scale > 1:
                # Average pooling for downsampling
                x_scaled = F.avg_pool1d(
                    x.transpose(1, 2), 
                    kernel_size=scale, 
                    stride=scale, 
                    padding=0
                ).transpose(1, 2)
                
                # Create mask for downsampled sequence
                if mask is not None:
                    mask_scaled = F.avg_pool1d(
                        mask.float().unsqueeze(1),
                        kernel_size=scale,
                        stride=scale,
                        padding=0
                    ).squeeze(1) > 0.5
                else:
                    mask_scaled = None
            else:
                x_scaled = x
                mask_scaled = mask
            
            # Apply attention at this scale
            attn_output, _ = attention(x_scaled, x_scaled, x_scaled, attn_mask=mask_scaled)
            
            # Upsample back to original length if needed
            if scale > 1:
                # Interpolate to original length
                attn_output = F.interpolate(
                    attn_output.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            # Project to reduced dimension
            scale_output = projection(attn_output)
            scale_outputs.append(scale_output)
        
        # Concatenate outputs from all scales
        multi_scale_output = torch.cat(scale_outputs, dim=-1)
        
        # Final projection
        output = self.output_projection(multi_scale_output)
        output = self.dropout(output)
        
        return output


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        
        # Optional: Add sinusoidal initialization
        self._init_sinusoidal()
        
    def _init_sinusoidal(self):
        """Initialize with sinusoidal patterns."""
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        with torch.no_grad():
            self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
            self.pos_embedding[:, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        pos_enc = self.pos_embedding[:seq_len, :].unsqueeze(0)
        return x + pos_enc


class AdaptiveDropout(nn.Module):
    """Adaptive dropout that varies based on uncertainty estimation."""
    
    def __init__(self, base_dropout: float = 0.1, uncertainty_factor: float = 0.1):
        super().__init__()
        self.base_dropout = base_dropout
        self.uncertainty_factor = uncertainty_factor
        self.register_buffer('uncertainty_level', torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive dropout."""
        if self.training:
            # Adjust dropout rate based on uncertainty
            adaptive_rate = self.base_dropout + self.uncertainty_factor * self.uncertainty_level
            adaptive_rate = torch.clamp(adaptive_rate, 0.0, 0.5)
            return F.dropout(x, p=adaptive_rate.item(), training=True)
        else:
            return x
    
    def update_uncertainty(self, uncertainty: float):
        """Update uncertainty level for adaptive dropout."""
        self.uncertainty_level.fill_(uncertainty)


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with multi-scale attention and improved normalization."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, scales: list = [1, 2, 4], 
                 dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        
        # Multi-scale attention
        self.multi_scale_attention = MultiScaleAttention(d_model, n_heads, scales, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (Pre-LN architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Adaptive dropout
        self.adaptive_dropout1 = AdaptiveDropout(dropout)
        self.adaptive_dropout2 = AdaptiveDropout(dropout)
        
        # Residual scaling (helps with deep networks)
        self.residual_scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through enhanced transformer block."""
        # Pre-LN multi-scale attention with residual connection
        norm_x = self.norm1(x)
        attn_output = self.multi_scale_attention(norm_x, mask)
        x = x + self.residual_scale * self.adaptive_dropout1(attn_output)
        
        # Pre-LN feed-forward with residual connection
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.residual_scale * self.adaptive_dropout2(ff_output)
        
        return x
    
    def update_uncertainty(self, uncertainty: float):
        """Update uncertainty for adaptive dropout."""
        self.adaptive_dropout1.update_uncertainty(uncertainty)
        self.adaptive_dropout2.update_uncertainty(uncertainty)


class EnhancedTimeSeriesTransformer(BaseModel):
    """
    Enhanced Transformer model for time series prediction with:
    - Multi-scale attention
    - Learnable positional encoding
    - Adaptive dropout for uncertainty
    - Residual scaling
    - Multiple prediction heads
    """
    
    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 16, 
                 num_layers: int = 8, d_ff: int = None, seq_len: int = 60,
                 scales: list = [1, 2, 4, 8], dropout: float = 0.1, 
                 activation: str = 'gelu', num_prediction_heads: int = 1):
        super().__init__(input_dim)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.scales = scales
        self.num_prediction_heads = num_prediction_heads
        
        if d_ff is None:
            d_ff = d_model * 4
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_len=seq_len * 2)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerBlock(d_model, n_heads, d_ff, scales, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        
        # Multiple prediction heads for uncertainty estimation
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 4, 1)
            )
            for _ in range(num_prediction_heads)
        ])
        
        # Uncertainty head (predicts log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Use last token for prediction (or could use mean/max pooling)
        x = x[:, -1, :]  # Shape: (batch_size, d_model)
        
        # Get prediction from first head
        prediction = self.prediction_heads[0](x)
        
        return prediction
    
    def predict_with_uncertainty(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Final normalization
        x = self.output_norm(x)
        
        # Use last token for prediction
        x = x[:, -1, :]  # Shape: (batch_size, d_model)
        
        # Get predictions from all heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(x)
            predictions.append(pred)
        
        # Calculate mean and variance across heads
        if len(predictions) > 1:
            predictions_tensor = torch.stack(predictions, dim=0)  # (num_heads, batch_size, 1)
            mean_prediction = torch.mean(predictions_tensor, dim=0)
            epistemic_uncertainty = torch.var(predictions_tensor, dim=0)
        else:
            mean_prediction = predictions[0]
            epistemic_uncertainty = torch.zeros_like(mean_prediction)
        
        # Get aleatoric uncertainty from uncertainty head
        log_var = self.uncertainty_head(x)
        aleatoric_uncertainty = torch.exp(log_var)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Update adaptive dropout based on uncertainty
        avg_uncertainty = torch.mean(total_uncertainty).item()
        for block in self.transformer_blocks:
            block.update_uncertainty(avg_uncertainty)
        
        return mean_prediction, total_uncertainty
    
    def monte_carlo_predict(self, x: torch.Tensor, n_samples: int = 100, 
                           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo prediction using dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            mask: Optional attention mask
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x, mask)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size, 1)
        
        mean_prediction = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_prediction, uncertainty
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from a specific layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (-1 for last layer)
            
        Returns:
            Dictionary of attention weights for each scale
        """
        # This is a simplified version - in practice, you'd need to modify
        # the attention modules to return weights
        self.eval()
        
        with torch.no_grad():
            # Forward pass up to the specified layer
            x = self.input_projection(x)
            x = self.pos_encoding(x)
            
            target_layer = self.transformer_blocks[layer_idx]
            
            # Get attention weights (would need to modify MultiScaleAttention)
            # This is a placeholder implementation
            attention_weights = {
                f'scale_{scale}': torch.randn(x.size(0), self.n_heads // len(self.scales), x.size(1), x.size(1))
                for scale in self.scales
            }
        
        return attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        
        enhanced_info = {
            **base_info,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'num_layers': self.num_layers,
            'seq_len': self.seq_len,
            'scales': self.scales,
            'num_prediction_heads': self.num_prediction_heads,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        return enhanced_info


class TransformerConfig:
    """Configuration class for Transformer model."""
    
    def __init__(self, **kwargs):
        # Default configuration
        self.input_dim = kwargs.get('input_dim', 15)
        self.d_model = kwargs.get('d_model', 256)
        self.n_heads = kwargs.get('n_heads', 16)
        self.num_layers = kwargs.get('num_layers', 8)
        self.d_ff = kwargs.get('d_ff', None)
        self.seq_len = kwargs.get('seq_len', 60)
        self.scales = kwargs.get('scales', [1, 2, 4, 8])
        self.dropout = kwargs.get('dropout', 0.1)
        self.activation = kwargs.get('activation', 'gelu')
        self.num_prediction_heads = kwargs.get('num_prediction_heads', 3)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'seq_len': self.seq_len,
            'scales': self.scales,
            'dropout': self.dropout,
            'activation': self.activation,
            'num_prediction_heads': self.num_prediction_heads
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_transformer_model(config: TransformerConfig) -> EnhancedTimeSeriesTransformer:
    """Factory function to create transformer model from config."""
    return EnhancedTimeSeriesTransformer(**config.to_dict())
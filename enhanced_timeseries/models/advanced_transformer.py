"""
Advanced Transformer model with multi-scale attention for time series forecasting.
Implements learnable positional embeddings, multi-head attention with different scales,
and adaptive dropout for uncertainty estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for time series data."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize learnable positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_len, d_model))
        
        # Initialize with sinusoidal pattern as starting point
        self._init_sinusoidal()
        
    def _init_sinusoidal(self):
        """Initialize with sinusoidal positional encoding pattern."""
        max_len, d_model = self.pos_embedding.shape
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        with torch.no_grad():
            self.pos_embedding[:, 0::2] = torch.sin(position * div_term)
            self.pos_embedding[:, 1::2] = torch.cos(position * div_term)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        pos_enc = self.pos_embedding[:seq_len, :].unsqueeze(0)
        
        x = x + pos_enc
        return self.dropout(x)


class MultiScaleAttention(nn.Module):
    """Multi-scale attention mechanism with different attention scales."""
    
    def __init__(self, d_model: int, n_heads: int, scales: List[int] = [1, 2, 4, 8],
                 dropout: float = 0.1):
        """
        Initialize multi-scale attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            scales: List of attention scales (dilation factors)
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.scales = scales
        self.d_k = d_model // n_heads
        
        # Separate attention heads for each scale
        self.scale_attentions = nn.ModuleList()
        
        for scale in scales:
            attention = nn.MultiheadAttention(
                embed_dim=d_model // len(scales),
                num_heads=n_heads // len(scales),
                dropout=dropout,
                batch_first=True
            )
            self.scale_attentions.append(attention)
        
        # Input projections for each scale
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model // len(scales))
            for _ in scales
        ])
        
        # Output fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-scale attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, d_model = x.shape
        
        scale_outputs = []
        attention_weights = []
        
        for i, (scale, projection, attention) in enumerate(zip(
            self.scales, self.scale_projections, self.scale_attentions
        )):
            # Project input for this scale
            x_scale = projection(x)
            
            # Apply dilated attention by subsampling
            if scale > 1:
                # Subsample sequence
                indices = torch.arange(0, seq_len, scale, device=x.device)
                x_sub = x_scale[:, indices, :]
                
                # Apply attention
                attn_out, attn_weights = attention(x_sub, x_sub, x_sub)
                
                # Upsample back to original length
                attn_out_full = torch.zeros_like(x_scale)
                attn_out_full[:, indices, :] = attn_out
                
                # Interpolate missing values
                if scale > 1:
                    for j in range(1, scale):
                        if indices[-1] + j < seq_len:
                            # Simple linear interpolation
                            left_idx = indices[indices < seq_len - j]
                            right_idx = left_idx + scale
                            right_idx = torch.clamp(right_idx, max=seq_len - 1)
                            
                            alpha = j / scale
                            attn_out_full[:, left_idx + j, :] = (
                                (1 - alpha) * attn_out_full[:, left_idx, :] +
                                alpha * attn_out_full[:, right_idx, :]
                            )
            else:
                # Regular attention for scale 1
                attn_out_full, attn_weights = attention(x_scale, x_scale, x_scale)
            
            scale_outputs.append(attn_out_full)
            attention_weights.append(attn_weights)
        
        # Concatenate outputs from all scales
        combined_output = torch.cat(scale_outputs, dim=-1)
        
        # Apply fusion layer
        fused_output = self.fusion_layer(combined_output)
        
        # Combine attention weights (handle different sizes)
        if attention_weights:
            # Use the first (full-scale) attention weights as representative
            combined_weights = attention_weights[0]
        else:
            combined_weights = None
        
        return fused_output, combined_weights


class AdaptiveDropout(nn.Module):
    """Adaptive dropout that adjusts dropout rate based on uncertainty."""
    
    def __init__(self, base_dropout: float = 0.1, uncertainty_threshold: float = 0.5):
        """
        Initialize adaptive dropout.
        
        Args:
            base_dropout: Base dropout rate
            uncertainty_threshold: Threshold for uncertainty-based adjustment
        """
        super().__init__()
        
        self.base_dropout = base_dropout
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_estimate = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adaptive dropout.
        
        Args:
            x: Input tensor
            uncertainty: Current uncertainty estimate
            
        Returns:
            Output with adaptive dropout applied
        """
        if not self.training:
            return x
        
        # Calculate adaptive dropout rate
        if uncertainty is not None:
            # Increase dropout when uncertainty is high
            uncertainty_factor = torch.clamp(uncertainty.mean(), 0, 1)
            adaptive_rate = self.base_dropout * (1 + uncertainty_factor)
        else:
            # Use learned uncertainty estimate
            uncertainty_factor = torch.sigmoid(self.uncertainty_estimate)
            adaptive_rate = self.base_dropout * (1 + uncertainty_factor)
        
        # Apply dropout
        return F.dropout(x, p=adaptive_rate.item(), training=self.training)


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with multi-scale attention and residual connections."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 scales: List[int] = [1, 2, 4], dropout: float = 0.1):
        """
        Initialize Transformer block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            scales: Attention scales
            dropout: Dropout rate
        """
        super().__init__()
        
        # Multi-scale attention
        self.attention = MultiScaleAttention(d_model, n_heads, scales, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Adaptive dropout
        self.adaptive_dropout1 = AdaptiveDropout(dropout)
        self.adaptive_dropout2 = AdaptiveDropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Transformer block.
        
        Args:
            x: Input tensor
            mask: Attention mask
            uncertainty: Current uncertainty estimate
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Multi-scale attention with residual connection
        attn_out, attn_weights = self.attention(self.norm1(x), mask)
        x = x + self.adaptive_dropout1(attn_out, uncertainty)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.adaptive_dropout2(ff_out, uncertainty)
        
        return x, attn_weights


class AdvancedTransformer(nn.Module):
    """Advanced Transformer model for time series forecasting."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 max_seq_len: int = 1000,
                 output_dim: int = 1,
                 scales: List[int] = [1, 2, 4, 8],
                 dropout: float = 0.1):
        """
        Initialize Advanced Transformer model.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of Transformer layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            output_dim: Output dimension
            scales: Multi-scale attention scales
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, scales, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Softplus()  # Ensure positive uncertainty
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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention: bool = False, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through Advanced Transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Attention mask
            return_attention: Whether to return attention weights
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions, optionally with attention weights and uncertainty
        """
        # Input projection and positional encoding
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Pass through Transformer layers
        attention_weights = []
        uncertainty_estimate = None
        
        for layer in self.transformer_layers:
            x, attn_weights = layer(x, mask, uncertainty_estimate)
            attention_weights.append(attn_weights)
            
            # Update uncertainty estimate for adaptive dropout
            if return_uncertainty:
                uncertainty_estimate = self.uncertainty_head(x.mean(dim=1, keepdim=True))
        
        # Output normalization
        x = self.output_norm(x)
        
        # Use last time step for prediction (or mean pooling)
        if x.size(1) > 1:
            # Use last time step
            output_features = x[:, -1, :]
        else:
            # Single time step
            output_features = x.squeeze(1)
        
        # Generate predictions
        predictions = self.output_projection(output_features)
        
        # Prepare outputs
        outputs = [predictions]
        
        if return_uncertainty:
            uncertainty = self.uncertainty_head(output_features)
            outputs.append(uncertainty)
        
        if return_attention:
            # Stack attention weights from all layers
            stacked_attention = torch.stack(attention_weights, dim=1)  # (batch, layers, heads, seq, seq)
            outputs.append(stacked_attention)
        
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with Monte Carlo dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Tuple of (mean_predictions, uncertainty_estimates)
        """
        self.train()  # Enable dropout
        
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred, unc = self.forward(x, return_uncertainty=True)
                predictions.append(pred)
                uncertainties.append(unc)
        
        predictions = torch.stack(predictions)  # (n_samples, batch_size, output_dim)
        uncertainties = torch.stack(uncertainties)  # (n_samples, batch_size, 1)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)
        aleatoric_uncertainty = uncertainties.mean(dim=0)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        self.eval()  # Disable dropout
        
        return mean_pred, total_uncertainty
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention maps from all layers
        """
        self.eval()
        with torch.no_grad():
            _, _, attention_maps = self.forward(x, return_attention=True)
        
        return attention_maps


class MultiHorizonTransformer(nn.Module):
    """Multi-horizon Transformer for different prediction horizons."""
    
    def __init__(self, 
                 input_dim: int,
                 prediction_horizons: List[int],
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize multi-horizon Transformer.
        
        Args:
            input_dim: Input feature dimension
            prediction_horizons: List of prediction horizons
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.prediction_horizons = prediction_horizons
        
        # Shared Transformer backbone
        self.backbone = AdvancedTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            output_dim=d_model,  # Output features, not predictions
            dropout=dropout
        )
        
        # Remove the output projection from backbone
        self.backbone.output_projection = nn.Identity()
        
        # Separate heads for each horizon
        self.prediction_heads = nn.ModuleDict()
        for horizon in prediction_horizons:
            self.prediction_heads[str(horizon)] = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, horizon)
            )
    
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
        shared_features = self.backbone(x)
        
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


class HierarchicalTransformer(nn.Module):
    """Hierarchical Transformer with multiple time scales."""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 scales: List[int] = [1, 4, 16],
                 output_dim: int = 1,
                 dropout: float = 0.1):
        """
        Initialize hierarchical Transformer.
        
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            n_heads: Number of attention heads
            scales: Hierarchical time scales
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.scales = scales
        
        # Separate Transformers for each scale
        self.scale_transformers = nn.ModuleList()
        
        for scale in scales:
            transformer = AdvancedTransformer(
                input_dim=input_dim,
                d_model=d_model // len(scales),
                n_heads=n_heads // len(scales),
                n_layers=2,
                output_dim=d_model // len(scales),
                dropout=dropout
            )
            # Remove output projection
            transformer.output_projection = nn.Identity()
            self.scale_transformers.append(transformer)
        
        # Cross-scale attention
        self.cross_scale_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final output layers
        self.output_layers = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through hierarchical Transformer.
        
        Args:
            x: Input tensor
            
        Returns:
            Hierarchical predictions
        """
        scale_features = []
        
        # Process each scale
        for scale, transformer in zip(self.scales, self.scale_transformers):
            if scale > 1:
                # Downsample for higher scales
                x_scale = x[:, ::scale, :]
            else:
                x_scale = x
            
            features = transformer(x_scale)
            
            # Upsample back if needed
            if scale > 1 and features.size(1) < x.size(1):
                # Simple repeat upsampling
                repeat_factor = x.size(1) // features.size(1)
                features = features.repeat_interleave(repeat_factor, dim=1)
                
                # Trim to exact length
                if features.size(1) > x.size(1):
                    features = features[:, :x.size(1), :]
            
            scale_features.append(features)
        
        # Concatenate features from all scales
        combined_features = torch.cat(scale_features, dim=-1)
        
        # Apply cross-scale attention
        attended_features, _ = self.cross_scale_attention(
            combined_features, combined_features, combined_features
        )
        
        # Use last time step for prediction
        output_features = attended_features[:, -1, :]
        
        # Generate final predictions
        predictions = self.output_layers(output_features)
        
        return predictions


# Utility functions
def create_advanced_transformer(config: Dict) -> AdvancedTransformer:
    """
    Create Advanced Transformer from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured Advanced Transformer model
    """
    return AdvancedTransformer(
        input_dim=config.get('input_dim', 1),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        max_seq_len=config.get('max_seq_len', 1000),
        output_dim=config.get('output_dim', 1),
        scales=config.get('scales', [1, 2, 4, 8]),
        dropout=config.get('dropout', 0.1)
    )


def count_transformer_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in different parts of the Transformer model."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    component_params = {}
    
    if hasattr(model, 'input_projection'):
        component_params['input_projection'] = sum(p.numel() for p in model.input_projection.parameters() if p.requires_grad)
    
    if hasattr(model, 'pos_encoding'):
        component_params['pos_encoding'] = sum(p.numel() for p in model.pos_encoding.parameters() if p.requires_grad)
    
    if hasattr(model, 'transformer_layers'):
        component_params['transformer_layers'] = sum(p.numel() for p in model.transformer_layers.parameters() if p.requires_grad)
    
    if hasattr(model, 'output_projection'):
        component_params['output_projection'] = sum(p.numel() for p in model.output_projection.parameters() if p.requires_grad)
    
    component_params['total'] = total_params
    
    return component_params
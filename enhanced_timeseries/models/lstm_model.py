"""
Advanced bidirectional LSTM model with attention mechanism, skip connections,
and variational dropout for time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from ..core.interfaces import BaseModel


class VariationalDropout(nn.Module):
    """Variational dropout that maintains the same dropout mask across time steps."""
    
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply variational dropout.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Tensor with variational dropout applied
        """
        if not self.training or self.dropout_rate == 0:
            return x
        
        # Create dropout mask for the feature dimension only
        # Same mask is applied across all time steps
        batch_size, seq_len, hidden_size = x.shape
        
        # Create mask for hidden dimension
        mask = torch.bernoulli(torch.full((batch_size, 1, hidden_size), 
                                        1 - self.dropout_rate, 
                                        device=x.device, dtype=x.dtype))
        
        # Scale by dropout probability
        mask = mask / (1 - self.dropout_rate)
        
        # Apply mask across all time steps
        return x * mask
    
    def extra_repr(self) -> str:
        return f'dropout_rate={self.dropout_rate}'


class AttentionMechanism(nn.Module):
    """Attention mechanism for LSTM outputs."""
    
    def __init__(self, hidden_size: int, attention_type: str = 'additive'):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        
        if attention_type == 'additive':
            # Additive (Bahdanau) attention
            self.attention_linear = nn.Linear(hidden_size, hidden_size)
            self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        elif attention_type == 'multiplicative':
            # Multiplicative (Luong) attention
            self.attention_linear = nn.Linear(hidden_size, hidden_size)
        elif attention_type == 'scaled_dot_product':
            # Scaled dot-product attention
            self.query_linear = nn.Linear(hidden_size, hidden_size)
            self.key_linear = nn.Linear(hidden_size, hidden_size)
            self.value_linear = nn.Linear(hidden_size, hidden_size)
            self.scale = np.sqrt(hidden_size)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, lstm_outputs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism.
        
        Args:
            lstm_outputs: LSTM outputs of shape (batch_size, seq_len, hidden_size)
            mask: Optional mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, hidden_size = lstm_outputs.shape
        
        if self.attention_type == 'additive':
            # Additive attention
            # Transform each hidden state
            transformed = torch.tanh(self.attention_linear(lstm_outputs))
            
            # Calculate attention scores
            attention_scores = self.context_vector(transformed).squeeze(-1)  # (batch_size, seq_len)
            
        elif self.attention_type == 'multiplicative':
            # Multiplicative attention
            # Use last hidden state as query
            query = lstm_outputs[:, -1:, :]  # (batch_size, 1, hidden_size)
            
            # Transform keys
            keys = self.attention_linear(lstm_outputs)  # (batch_size, seq_len, hidden_size)
            
            # Calculate attention scores
            attention_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # (batch_size, seq_len)
            
        elif self.attention_type == 'scaled_dot_product':
            # Scaled dot-product attention
            queries = self.query_linear(lstm_outputs[:, -1:, :])  # Use last state as query
            keys = self.key_linear(lstm_outputs)
            values = self.value_linear(lstm_outputs)
            
            # Calculate attention scores
            attention_scores = torch.bmm(queries, keys.transpose(1, 2)).squeeze(1) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_scores)  # (batch_size, seq_len)
        
        # Calculate weighted sum
        if self.attention_type == 'scaled_dot_product':
            attended_output = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        else:
            attended_output = torch.bmm(attention_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        
        return attended_output, attention_weights


class SkipConnectionLSTM(nn.Module):
    """LSTM layer with skip connections."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Skip connection projection (if input and output sizes differ)
        if input_size != hidden_size * 2:  # *2 for bidirectional
            self.skip_projection = nn.Linear(input_size, hidden_size * 2)
        else:
            self.skip_projection = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Variational dropout
        self.variational_dropout = VariationalDropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            Tuple of (output, (h_n, c_n))
        """
        # LSTM forward pass
        lstm_out, hidden_state = self.lstm(x, hidden)
        
        # Apply skip connection if input size matches
        if self.skip_projection is not None:
            # Project input to match LSTM output size
            skip_connection = self.skip_projection(x)
        else:
            skip_connection = x
        
        # Add skip connection
        output = lstm_out + skip_connection
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Apply variational dropout
        output = self.variational_dropout(output)
        
        return output, hidden_state


class EnhancedBidirectionalLSTM(BaseModel):
    """
    Enhanced bidirectional LSTM model with:
    - Attention mechanism
    - Skip connections between layers
    - Variational dropout
    - Multiple prediction heads for uncertainty
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4,
                 dropout: float = 0.2, attention_type: str = 'additive',
                 num_prediction_heads: int = 1, use_skip_connections: bool = True):
        super().__init__(input_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_type = attention_type
        self.num_prediction_heads = num_prediction_heads
        self.use_skip_connections = use_skip_connections
        
        # LSTM layers with skip connections
        self.lstm_layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                input_size = input_dim
            else:
                input_size = hidden_dim * 2  # *2 for bidirectional
            
            if use_skip_connections:
                lstm_layer = SkipConnectionLSTM(input_size, hidden_dim, dropout)
            else:
                lstm_layer = nn.LSTM(input_size, hidden_dim, batch_first=True, 
                                   bidirectional=True, dropout=dropout if i < num_layers - 1 else 0)
            
            self.lstm_layers.append(lstm_layer)
        
        # Attention mechanism
        self.attention = AttentionMechanism(hidden_dim * 2, attention_type)
        
        # Output layers
        self.output_dropout = nn.Dropout(dropout)
        
        # Multiple prediction heads for uncertainty estimation
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in range(num_prediction_heads)
        ])
        
        # Uncertainty head (predicts log variance)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Feature importance weights (for interpretability)
        self.feature_attention = nn.Linear(input_dim, 1)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with gradient flow)
                if 'bias_ih' in name:
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask for attention
            
        Returns:
            Prediction tensor of shape (batch_size, 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pass through LSTM layers
        lstm_output = x
        hidden_states = []
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            if self.use_skip_connections:
                lstm_output, hidden = lstm_layer(lstm_output)
            else:
                lstm_output, hidden = lstm_layer(lstm_output)
            
            hidden_states.append(lstm_output)
        
        # Apply attention mechanism
        attended_output, attention_weights = self.attention(lstm_output, mask)
        
        # Apply dropout
        attended_output = self.output_dropout(attended_output)
        
        # Get prediction from first head
        prediction = self.prediction_heads[0](attended_output)
        
        return prediction
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make prediction with uncertainty estimation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask for attention
            
        Returns:
            Tuple of (prediction, uncertainty)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pass through LSTM layers
        lstm_output = x
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            if self.use_skip_connections:
                lstm_output, _ = lstm_layer(lstm_output)
            else:
                lstm_output, _ = lstm_layer(lstm_output)
        
        # Apply attention mechanism
        attended_output, attention_weights = self.attention(lstm_output, mask)
        
        # Apply dropout
        attended_output = self.output_dropout(attended_output)
        
        # Get predictions from all heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(attended_output)
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
        log_var = self.uncertainty_head(attended_output)
        aleatoric_uncertainty = torch.exp(log_var)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_prediction, total_uncertainty
    
    def get_attention_weights(self, x: torch.Tensor, 
                            mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            mask: Optional mask
            
        Returns:
            Attention weights of shape (batch_size, seq_len)
        """
        self.eval()
        
        with torch.no_grad():
            # Pass through LSTM layers
            lstm_output = x
            
            for lstm_layer in self.lstm_layers:
                if self.use_skip_connections:
                    lstm_output, _ = lstm_layer(lstm_output)
                else:
                    lstm_output, _ = lstm_layer(lstm_output)
            
            # Get attention weights
            _, attention_weights = self.attention(lstm_output, mask)
        
        return attention_weights
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Feature importance scores of shape (batch_size, input_dim)
        """
        # Calculate feature attention scores
        feature_scores = self.feature_attention(x)  # (batch_size, seq_len, 1)
        
        # Average across time steps
        feature_importance = torch.mean(torch.abs(feature_scores), dim=1).squeeze(-1)
        
        return feature_importance
    
    def monte_carlo_predict(self, x: torch.Tensor, n_samples: int = 100,
                           mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo prediction using dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            mask: Optional mask
            
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
    
    def get_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get hidden states from all LSTM layers.
        
        Args:
            x: Input tensor
            
        Returns:
            List of hidden states from each layer
        """
        hidden_states = []
        lstm_output = x
        
        for lstm_layer in self.lstm_layers:
            if self.use_skip_connections:
                lstm_output, _ = lstm_layer(lstm_output)
            else:
                lstm_output, _ = lstm_layer(lstm_output)
            
            hidden_states.append(lstm_output.detach())
        
        return hidden_states
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        
        enhanced_info = {
            **base_info,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'attention_type': self.attention_type,
            'num_prediction_heads': self.num_prediction_heads,
            'use_skip_connections': self.use_skip_connections,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        return enhanced_info


class LSTMConfig:
    """Configuration class for LSTM model."""
    
    def __init__(self, **kwargs):
        # Default configuration
        self.input_dim = kwargs.get('input_dim', 15)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.num_layers = kwargs.get('num_layers', 4)
        self.dropout = kwargs.get('dropout', 0.2)
        self.attention_type = kwargs.get('attention_type', 'additive')
        self.num_prediction_heads = kwargs.get('num_prediction_heads', 3)
        self.use_skip_connections = kwargs.get('use_skip_connections', True)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'attention_type': self.attention_type,
            'num_prediction_heads': self.num_prediction_heads,
            'use_skip_connections': self.use_skip_connections
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LSTMConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def create_lstm_model(config: LSTMConfig) -> EnhancedBidirectionalLSTM:
    """Factory function to create LSTM model from config."""
    return EnhancedBidirectionalLSTM(**config.to_dict())
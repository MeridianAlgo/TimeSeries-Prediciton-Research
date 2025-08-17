"""
Monte Carlo dropout uncertainty estimation for time series models.
Implements various uncertainty quantification methods including MC dropout,
deep ensembles, and Bayesian neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Callable, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class MCDropout(nn.Module):
    """Monte Carlo Dropout layer that can be enabled during inference."""
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout regardless of training mode when MC sampling."""
        return F.dropout(x, self.p, training=True, inplace=self.inplace)
    
    def extra_repr(self) -> str:
        return f'p={self.p}, inplace={self.inplace}'


class ConcreteDropout(nn.Module):
    """
    Concrete Dropout for automatic dropout rate learning.
    Based on "Concrete Dropout" by Gal, Hron, and Kendall.
    """
    
    def __init__(self, layer: nn.Module, input_dim: int, weight_regularizer: float = 1e-6,
                 dropout_regularizer: float = 1e-5, init_min: float = 0.1, init_max: float = 0.1):
        super().__init__()
        
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        # Initialize dropout probability parameters
        init_min = np.log(init_min) - np.log(1.0 - init_min)
        init_max = np.log(init_max) - np.log(1.0 - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(input_dim).uniform_(init_min, init_max))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with concrete dropout."""
        # Convert logits to probabilities
        p = torch.sigmoid(self.p_logit)
        
        # Apply concrete dropout
        eps = 1e-7
        temp = 0.1
        
        # Sample from concrete distribution
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) + 
                    torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        
        # Apply dropout mask
        random_tensor = 1.0 - drop_prob
        retain_prob = 1.0 - p
        x = x * random_tensor / retain_prob
        
        # Pass through layer
        output = self.layer(x)
        
        return output
    
    def regularization_loss(self) -> torch.Tensor:
        """Calculate regularization loss for concrete dropout."""
        p = torch.sigmoid(self.p_logit)
        
        # Weight regularization
        weight_reg = self.weight_regularizer * torch.sum(torch.square(self.layer.weight)) / (1 - p)
        
        # Dropout regularization
        dropout_reg = p * torch.log(p)
        dropout_reg += (1 - p) * torch.log(1 - p)
        dropout_reg = self.dropout_regularizer * torch.sum(dropout_reg)
        
        return weight_reg + dropout_reg


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling."""
        # Sample weights
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_std * weight_eps
        
        # Sample bias
        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias_eps = torch.randn_like(self.bias_mu)
        bias = self.bias_mu + bias_std * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence from prior."""
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2 + 
            weight_var / self.prior_std**2 - 
            self.weight_logvar + 
            np.log(self.prior_std**2) - 1
        )
        
        # KL divergence for bias
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std**2 + 
            bias_var / self.prior_std**2 - 
            self.bias_logvar + 
            np.log(self.prior_std**2) - 1
        )
        
        return weight_kl + bias_kl


class UncertaintyQuantifier:
    """
    Comprehensive uncertainty quantification using multiple methods.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def monte_carlo_predict(self, x: torch.Tensor, n_samples: int = 100, 
                           enable_dropout: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo prediction using dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            enable_dropout: Whether to enable dropout during inference
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        if enable_dropout:
            self.model.train()  # Enable dropout
        else:
            self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, batch_size, output_dim)
        
        # Calculate statistics
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        return mean_pred, var_pred
    
    def deep_ensemble_predict(self, x: torch.Tensor, models: List[nn.Module]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deep ensemble prediction using multiple models.
        
        Args:
            x: Input tensor
            models: List of trained models
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        return mean_pred, var_pred
    
    def bayesian_predict(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian prediction for models with Bayesian layers.
        
        Args:
            x: Input tensor
            n_samples: Number of samples from posterior
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.model.train()  # Enable sampling
        
        predictions = []
        
        for _ in range(n_samples):
            pred = self.model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        return mean_pred, var_pred
    
    def temperature_scaling_predict(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Temperature scaling for calibrated uncertainty.
        
        Args:
            x: Input tensor
            temperature: Temperature parameter for scaling
            
        Returns:
            Tuple of (prediction, calibrated_uncertainty)
        """
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(x)
            
            # Apply temperature scaling
            scaled_logits = logits / temperature
            
            # For regression, we can use the scaled output directly
            # For classification, you'd apply softmax
            prediction = scaled_logits
            
            # Estimate uncertainty based on temperature
            # Higher temperature = higher uncertainty
            uncertainty = torch.full_like(prediction, temperature - 1.0)
            uncertainty = torch.clamp(uncertainty, min=0.0)
        
        return prediction, uncertainty
    
    def quantile_predict(self, x: torch.Tensor, quantiles: List[float] = [0.025, 0.5, 0.975]) -> Dict[str, torch.Tensor]:
        """
        Quantile prediction for uncertainty intervals.
        
        Args:
            x: Input tensor
            quantiles: List of quantiles to predict
            
        Returns:
            Dictionary mapping quantile names to predictions
        """
        # This assumes the model has been trained with quantile loss
        # For demonstration, we'll use MC dropout to estimate quantiles
        
        mean_pred, var_pred = self.monte_carlo_predict(x, n_samples=100)
        std_pred = torch.sqrt(var_pred)
        
        results = {}
        
        for q in quantiles:
            # Use normal approximation for quantiles
            z_score = stats.norm.ppf(q)
            quantile_pred = mean_pred + z_score * std_pred
            results[f'quantile_{q}'] = quantile_pred
        
        return results
    
    def epistemic_aleatoric_split(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split uncertainty into epistemic and aleatoric components.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_prediction, epistemic_uncertainty, aleatoric_uncertainty)
        """
        # This requires a model that outputs both mean and variance
        # For demonstration, we'll estimate both components
        
        self.model.train()
        predictions = []
        variances = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)
                
                if output.shape[-1] == 2:  # Model outputs mean and log_var
                    mean_out = output[..., 0:1]
                    log_var_out = output[..., 1:2]
                    var_out = torch.exp(log_var_out)
                else:  # Model outputs only mean
                    mean_out = output
                    var_out = torch.zeros_like(mean_out)
                
                predictions.append(mean_out)
                variances.append(var_out)
        
        predictions = torch.stack(predictions, dim=0)
        variances = torch.stack(variances, dim=0)
        
        # Mean prediction
        mean_pred = torch.mean(predictions, dim=0)
        
        # Epistemic uncertainty (variance of means)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Aleatoric uncertainty (mean of variances)
        aleatoric_uncertainty = torch.mean(variances, dim=0)
        
        return mean_pred, epistemic_uncertainty, aleatoric_uncertainty


class UncertaintyCalibration:
    """Calibration methods for uncertainty estimates."""
    
    def __init__(self):
        self.calibration_curve = None
        self.is_calibrated = False
        
    def fit_calibration(self, uncertainties: np.ndarray, errors: np.ndarray, 
                       method: str = 'isotonic') -> 'UncertaintyCalibration':
        """
        Fit calibration curve.
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors
            method: Calibration method ('isotonic' or 'platt')
            
        Returns:
            Self for chaining
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        if method == 'isotonic':
            self.calibration_curve = IsotonicRegression(out_of_bounds='clip')
            self.calibration_curve.fit(uncertainties, errors)
        elif method == 'platt':
            # For Platt scaling, we need to convert to binary classification
            # This is a simplified version
            threshold = np.median(errors)
            binary_errors = (errors > threshold).astype(int)
            
            self.calibration_curve = LogisticRegression()
            self.calibration_curve.fit(uncertainties.reshape(-1, 1), binary_errors)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        self.is_calibrated = True
        return self
    
    def calibrate_uncertainty(self, uncertainties: np.ndarray) -> np.ndarray:
        """
        Calibrate uncertainty estimates.
        
        Args:
            uncertainties: Raw uncertainty estimates
            
        Returns:
            Calibrated uncertainties
        """
        if not self.is_calibrated:
            raise ValueError("Calibration curve must be fitted first")
        
        if hasattr(self.calibration_curve, 'predict'):
            # For Platt scaling
            calibrated = self.calibration_curve.predict_proba(uncertainties.reshape(-1, 1))[:, 1]
        else:
            # For isotonic regression
            calibrated = self.calibration_curve.predict(uncertainties)
        
        return calibrated


class UncertaintyMetrics:
    """Metrics for evaluating uncertainty quality."""
    
    @staticmethod
    def calibration_error(uncertainties: np.ndarray, errors: np.ndarray, 
                         n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors
            n_bins: Number of bins for calibration
            
        Returns:
            Expected calibration error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Average confidence in this bin
                confidence_in_bin = uncertainties[in_bin].mean()
                
                # Average accuracy in this bin
                accuracy_in_bin = (errors[in_bin] <= confidence_in_bin).mean()
                
                # Add to ECE
                ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def reliability_diagram(uncertainties: np.ndarray, errors: np.ndarray, 
                           n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate reliability diagram data.
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual errors
            n_bins: Number of bins
            
        Returns:
            Tuple of (bin_centers, accuracies, confidences)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        accuracies = []
        confidences = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            
            if in_bin.sum() > 0:
                confidence_in_bin = uncertainties[in_bin].mean()
                accuracy_in_bin = (errors[in_bin] <= confidence_in_bin).mean()
            else:
                confidence_in_bin = bin_centers[i]
                accuracy_in_bin = 0.0
            
            confidences.append(confidence_in_bin)
            accuracies.append(accuracy_in_bin)
        
        return bin_centers, np.array(accuracies), np.array(confidences)
    
    @staticmethod
    def sharpness(uncertainties: np.ndarray) -> float:
        """
        Calculate sharpness (average uncertainty).
        
        Args:
            uncertainties: Predicted uncertainties
            
        Returns:
            Sharpness score (lower is better)
        """
        return np.mean(uncertainties)
    
    @staticmethod
    def coverage_probability(predictions: np.ndarray, uncertainties: np.ndarray, 
                           actuals: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Calculate coverage probability for prediction intervals.
        
        Args:
            predictions: Point predictions
            uncertainties: Uncertainty estimates (standard deviations)
            actuals: Actual values
            confidence_level: Desired confidence level
            
        Returns:
            Actual coverage probability
        """
        # Calculate prediction intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bounds = predictions - z_score * uncertainties
        upper_bounds = predictions + z_score * uncertainties
        
        # Check coverage
        covered = (actuals >= lower_bounds) & (actuals <= upper_bounds)
        
        return np.mean(covered)


def create_mc_dropout_model(base_model: nn.Module, dropout_rate: float = 0.1) -> nn.Module:
    """
    Convert a regular model to use MC Dropout.
    
    Args:
        base_model: Base model to convert
        dropout_rate: Dropout rate for MC sampling
        
    Returns:
        Model with MC Dropout layers
    """
    def replace_dropout(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                setattr(module, name, MCDropout(p=dropout_rate))
            else:
                replace_dropout(child)
    
    # Create a copy of the model
    mc_model = type(base_model)(base_model.input_dim)
    mc_model.load_state_dict(base_model.state_dict())
    
    # Replace dropout layers
    replace_dropout(mc_model)
    
    return mc_model
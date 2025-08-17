"""Simple predictor for testing."""

import numpy as np
import pandas as pd
import logging


class SimplePredictor:
    """Simple predictor for testing."""
    
    def __init__(self):
        """Initialize simple predictor."""
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        
    def train(self, data: pd.DataFrame):
        """Train the predictor."""
        self.logger.info("Training simple predictor")
        self.is_trained = True
        return {'status': 'completed'}
        
    def predict(self, data: pd.DataFrame):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Simple prediction: small random changes
        n_samples = len(data)
        predictions = np.random.normal(0, 0.01, n_samples)
        return predictions
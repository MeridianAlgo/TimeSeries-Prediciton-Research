"""Ultra-Precision Stock Price Predictor

A sophisticated machine learning system designed to achieve stock price 
prediction errors consistently under 0.5%.
"""

__version__ = "1.0.0"
__author__ = "Ultra-Precision Team"

from .core.config import UltraPrecisionConfig

# Import predictor only if needed to avoid circular imports
def get_predictor():
    from .predictor import UltraPrecisionPredictor
    return UltraPrecisionPredictor

__all__ = ["get_predictor", "UltraPrecisionConfig"]
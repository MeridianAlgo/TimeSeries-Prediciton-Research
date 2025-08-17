"""Data models and structures for stock price prediction."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class StockData:
    """Represents a single stock data point with OHLC values."""
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: float


@dataclass
class PredictionResult:
    """Represents a prediction result with confidence intervals."""
    date: datetime
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    model_contributions: Dict[str, float]


@dataclass
class ModelPerformance:
    """Represents model performance metrics."""
    model_name: str
    mae: float
    rmse: float
    directional_accuracy: float
    training_time: float
    last_updated: datetime
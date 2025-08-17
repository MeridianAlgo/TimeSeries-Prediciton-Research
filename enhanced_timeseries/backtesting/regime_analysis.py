"""
Market regime detection and performance tracking for backtesting.
Implements statistical methods for regime classification and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


class VolatilityRegime(Enum):
    """Volatility regime types."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Trend regime types."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    SIDEWAYS = "sideways"
    WEAK_DOWNTREND = "weak_downtrend"
    STRONG_DOWNTREND = "strong_downtrend"


@dataclass
class RegimeClassification:
    """Classification of market regime at a point in time."""
    timestamp: pd.Timestamp
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    confidence: float
    regime_features: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'market_regime': self.market_regime.value,
            'volatility_regime': self.volatility_regime.value,
            'trend_regime': self.trend_regime.value,
            'confidence': self.confidence,
            'regime_features': self.regime_features
        }


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    regime: MarketRegime
    n_periods: int
    total_predictions: int
    mae: float
    rmse: float
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_return: float
    volatility: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MarketRegimeDetector:
    """
    Market regime detection using multiple statistical methods.
    """
    
    def __init__(self, lookback_window: int = 252, min_regime_length: int = 21):
        self.lookback_window = lookback_window
        self.min_regime_length = min_regime_length
        self.regime_history = []
        
    def detect_regimes(self, data: pd.DataFrame, price_column: str = 'Close') -> List[RegimeClassification]:
        """
        Detect market regimes for the entire time series.
        
        Args:
            data: Time series data with datetime index
            price_column: Name of price column
            
        Returns:
            List of regime classifications
        """
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        prices = data[price_column]
        returns = prices.pct_change().fillna(0)
        
        regime_classifications = []
        
        for i, timestamp in enumerate(data.index):
            if i < self.lookback_window:
                # Not enough data for regime detection
                classification = RegimeClassification(
                    timestamp=timestamp,
                    market_regime=MarketRegime.UNKNOWN,
                    volatility_regime=VolatilityRegime.NORMAL,
                    trend_regime=TrendRegime.SIDEWAYS,
                    confidence=0.0,
                    regime_features={}
                )
            else:
                # Get window data
                window_prices = prices.iloc[i-self.lookback_window:i+1]
                window_returns = returns.iloc[i-self.lookback_window:i+1]
                
                # Detect regimes
                market_regime, market_confidence = self._detect_market_regime(window_prices, window_returns)
                volatility_regime = self._detect_volatility_regime(window_returns)
                trend_regime = self._detect_trend_regime(window_prices)
                
                # Calculate regime features
                regime_features = self._calculate_regime_features(window_prices, window_returns)
                
                classification = RegimeClassification(
                    timestamp=timestamp,
                    market_regime=market_regime,
                    volatility_regime=volatility_regime,
                    trend_regime=trend_regime,
                    confidence=market_confidence,
                    regime_features=regime_features
                )
            
            regime_classifications.append(classification)
        
        # Smooth regime transitions
        regime_classifications = self._smooth_regime_transitions(regime_classifications)
        
        return regime_classifications
    
    def _detect_market_regime(self, prices: pd.Series, returns: pd.Series) -> Tuple[MarketRegime, float]:
        """Detect overall market regime."""
        # Calculate key statistics
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # VIX-like fear index (rolling volatility percentile)
        rolling_vol = returns.rolling(21).std()
        vol_percentile = rolling_vol.iloc[-1] / rolling_vol.quantile(0.95) if rolling_vol.quantile(0.95) > 0 else 1
        
        confidence = 0.7  # Base confidence
        
        # Crisis detection (high volatility + large drawdown)
        if volatility > 0.4 and max_drawdown > 0.2:
            return MarketRegime.CRISIS, min(0.9, confidence + 0.2)
        
        # High volatility regime
        elif volatility > 0.3 or vol_percentile > 1.5:
            return MarketRegime.HIGH_VOLATILITY, min(0.8, confidence + 0.1)
        
        # Bull market (positive returns + reasonable volatility)
        elif total_return > 0.1 and volatility < 0.25 and sharpe > 0.5:
            return MarketRegime.BULL, min(0.9, confidence + 0.2)
        
        # Bear market (negative returns)
        elif total_return < -0.1 and max_drawdown > 0.1:
            return MarketRegime.BEAR, min(0.9, confidence + 0.2)
        
        # Recovery (positive momentum after drawdown)
        elif total_return > 0.05 and max_drawdown > 0.15:
            return MarketRegime.RECOVERY, min(0.8, confidence + 0.1)
        
        # Low volatility regime
        elif volatility < 0.15:
            return MarketRegime.LOW_VOLATILITY, min(0.8, confidence + 0.1)
        
        # Default to sideways
        else:
            return MarketRegime.SIDEWAYS, confidence
    
    def _detect_volatility_regime(self, returns: pd.Series) -> VolatilityRegime:
        """Detect volatility regime."""
        # Calculate rolling volatility
        rolling_vol = returns.rolling(21).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Historical percentiles
        vol_25 = rolling_vol.quantile(0.25)
        vol_75 = rolling_vol.quantile(0.75)
        vol_95 = rolling_vol.quantile(0.95)
        
        if current_vol > vol_95:
            return VolatilityRegime.EXTREME
        elif current_vol > vol_75:
            return VolatilityRegime.HIGH
        elif current_vol < vol_25:
            return VolatilityRegime.LOW
        else:
            return VolatilityRegime.NORMAL
    
    def _detect_trend_regime(self, prices: pd.Series) -> TrendRegime:
        """Detect trend regime using linear regression."""
        # Linear regression on log prices
        log_prices = np.log(prices)
        x = np.arange(len(log_prices))
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_prices)
        
        # Annualize slope
        annual_slope = slope * 252
        r_squared = r_value ** 2
        
        # Classify trend based on slope and R-squared
        if r_squared > 0.5:  # Strong trend
            if annual_slope > 0.2:
                return TrendRegime.STRONG_UPTREND
            elif annual_slope < -0.2:
                return TrendRegime.STRONG_DOWNTREND
            elif annual_slope > 0.05:
                return TrendRegime.WEAK_UPTREND
            elif annual_slope < -0.05:
                return TrendRegime.WEAK_DOWNTREND
        
        return TrendRegime.SIDEWAYS
    
    def _calculate_regime_features(self, prices: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate features that characterize the current regime."""
        features = {}
        
        # Return statistics
        features['total_return'] = (prices.iloc[-1] / prices.iloc[0]) - 1
        features['annualized_return'] = features['total_return'] * (252 / len(returns))
        features['volatility'] = returns.std() * np.sqrt(252)
        features['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        features['max_drawdown'] = abs(drawdown.min())
        features['current_drawdown'] = abs(drawdown.iloc[-1])
        
        # Skewness and kurtosis
        features['skewness'] = returns.skew()
        features['kurtosis'] = returns.kurtosis()
        
        # Momentum indicators
        features['momentum_1m'] = (prices.iloc[-1] / prices.iloc[-21]) - 1 if len(prices) >= 21 else 0
        features['momentum_3m'] = (prices.iloc[-1] / prices.iloc[-63]) - 1 if len(prices) >= 63 else 0
        features['momentum_6m'] = (prices.iloc[-1] / prices.iloc[-126]) - 1 if len(prices) >= 126 else 0
        
        # Volatility percentile
        rolling_vol = returns.rolling(21).std()
        if len(rolling_vol.dropna()) > 0:
            features['volatility_percentile'] = rolling_vol.iloc[-1] / rolling_vol.quantile(0.95)
        else:
            features['volatility_percentile'] = 1.0
        
        # Trend strength (R-squared of linear regression)
        if len(prices) > 10:
            log_prices = np.log(prices)
            x = np.arange(len(log_prices))
            _, _, r_value, _, _ = stats.linregress(x, log_prices)
            features['trend_strength'] = r_value ** 2
        else:
            features['trend_strength'] = 0.0
        
        return features
    
    def _smooth_regime_transitions(self, classifications: List[RegimeClassification]) -> List[RegimeClassification]:
        """Smooth regime transitions to avoid excessive switching."""
        if len(classifications) < self.min_regime_length:
            return classifications
        
        smoothed = classifications.copy()
        
        # Smooth market regimes
        for i in range(self.min_regime_length, len(classifications)):
            current_regime = classifications[i].market_regime
            
            # Check if we have a consistent regime in the recent past
            recent_regimes = [c.market_regime for c in classifications[i-self.min_regime_length:i]]
            
            if len(set(recent_regimes)) == 1 and recent_regimes[0] != current_regime:
                # If recent history is consistent but different from current, keep recent
                smoothed[i] = RegimeClassification(
                    timestamp=classifications[i].timestamp,
                    market_regime=recent_regimes[0],
                    volatility_regime=classifications[i].volatility_regime,
                    trend_regime=classifications[i].trend_regime,
                    confidence=classifications[i].confidence * 0.8,  # Reduce confidence for smoothed
                    regime_features=classifications[i].regime_features
                )
        
        return smoothed


class RegimePerformanceTracker:
    """
    Track model performance across different market regimes.
    """
    
    def __init__(self):
        self.regime_data = {}
        self.performance_history = []
        
    def add_prediction(self, timestamp: pd.Timestamp, prediction: float, actual: float,
                      regime_classification: RegimeClassification):
        """Add a prediction with its regime classification."""
        regime = regime_classification.market_regime
        
        if regime not in self.regime_data:
            self.regime_data[regime] = {
                'predictions': [],
                'actuals': [],
                'timestamps': [],
                'classifications': []
            }
        
        self.regime_data[regime]['predictions'].append(prediction)
        self.regime_data[regime]['actuals'].append(actual)
        self.regime_data[regime]['timestamps'].append(timestamp)
        self.regime_data[regime]['classifications'].append(regime_classification)
        
        # Add to overall history
        self.performance_history.append({
            'timestamp': timestamp,
            'prediction': prediction,
            'actual': actual,
            'regime': regime,
            'classification': regime_classification
        })
    
    def calculate_regime_performance(self) -> Dict[MarketRegime, RegimePerformance]:
        """Calculate performance metrics for each regime."""
        regime_performance = {}
        
        for regime, data in self.regime_data.items():
            if len(data['predictions']) == 0:
                continue
            
            predictions = np.array(data['predictions'])
            actuals = np.array(data['actuals'])
            
            # Basic metrics
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            
            # Directional accuracy
            directional_accuracy = np.mean(np.sign(predictions) == np.sign(actuals)) * 100
            
            # Financial metrics (treating predictions as returns)
            returns = predictions
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # Win rate
            win_rate = np.mean(returns > 0)
            
            # Average return and volatility
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)
            
            regime_performance[regime] = RegimePerformance(
                regime=regime,
                n_periods=len(set(pd.to_datetime([ts for ts in data['timestamps']]).date)),
                total_predictions=len(predictions),
                mae=mae,
                rmse=rmse,
                directional_accuracy=directional_accuracy,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                avg_return=avg_return,
                volatility=volatility
            )
        
        return regime_performance
    
    def get_regime_distribution(self) -> Dict[str, float]:
        """Get distribution of predictions across regimes."""
        total_predictions = len(self.performance_history)
        
        if total_predictions == 0:
            return {}
        
        regime_counts = {}
        for entry in self.performance_history:
            regime = entry['regime'].value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Convert to percentages
        regime_distribution = {
            regime: count / total_predictions * 100
            for regime, count in regime_counts.items()
        }
        
        return regime_distribution
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        regime_performance = self.calculate_regime_performance()
        regime_distribution = self.get_regime_distribution()
        
        # Overall statistics
        if self.performance_history:
            all_predictions = [entry['prediction'] for entry in self.performance_history]
            all_actuals = [entry['actual'] for entry in self.performance_history]
            
            overall_mae = np.mean(np.abs(np.array(all_predictions) - np.array(all_actuals)))
            overall_directional_accuracy = np.mean(
                np.sign(all_predictions) == np.sign(all_actuals)
            ) * 100
        else:
            overall_mae = float('inf')
            overall_directional_accuracy = 0.0
        
        # Best and worst performing regimes
        if regime_performance:
            best_regime = min(regime_performance.items(), key=lambda x: x[1].mae)
            worst_regime = max(regime_performance.items(), key=lambda x: x[1].mae)
        else:
            best_regime = None
            worst_regime = None
        
        return {
            'regime_performance': {regime.value: perf.to_dict() 
                                 for regime, perf in regime_performance.items()},
            'regime_distribution': regime_distribution,
            'overall_mae': overall_mae,
            'overall_directional_accuracy': overall_directional_accuracy,
            'total_predictions': len(self.performance_history),
            'n_regimes_encountered': len(regime_performance),
            'best_regime': {
                'regime': best_regime[0].value,
                'mae': best_regime[1].mae
            } if best_regime else None,
            'worst_regime': {
                'regime': worst_regime[0].value,
                'mae': worst_regime[1].mae
            } if worst_regime else None
        }
    
    def get_regime_transitions(self) -> List[Dict[str, Any]]:
        """Analyze regime transitions and their impact on performance."""
        if len(self.performance_history) < 2:
            return []
        
        transitions = []
        
        for i in range(1, len(self.performance_history)):
            prev_regime = self.performance_history[i-1]['regime']
            curr_regime = self.performance_history[i]['regime']
            
            if prev_regime != curr_regime:
                # Calculate performance around transition
                window_size = 10
                start_idx = max(0, i - window_size)
                end_idx = min(len(self.performance_history), i + window_size)
                
                window_data = self.performance_history[start_idx:end_idx]
                
                pre_transition = [entry for entry in window_data[:window_size] 
                                if entry['regime'] == prev_regime]
                post_transition = [entry for entry in window_data[window_size:] 
                                 if entry['regime'] == curr_regime]
                
                if pre_transition and post_transition:
                    pre_mae = np.mean([abs(entry['prediction'] - entry['actual']) 
                                     for entry in pre_transition])
                    post_mae = np.mean([abs(entry['prediction'] - entry['actual']) 
                                      for entry in post_transition])
                    
                    transitions.append({
                        'timestamp': self.performance_history[i]['timestamp'].isoformat(),
                        'from_regime': prev_regime.value,
                        'to_regime': curr_regime.value,
                        'pre_transition_mae': pre_mae,
                        'post_transition_mae': post_mae,
                        'performance_change': post_mae - pre_mae
                    })
        
        return transitions


class AdaptiveRegimeModel:
    """
    Model that adapts its behavior based on detected market regimes.
    """
    
    def __init__(self, base_predictor_factory, regime_detector: MarketRegimeDetector):
        self.base_predictor_factory = base_predictor_factory
        self.regime_detector = regime_detector
        self.regime_models = {}
        self.current_regime = MarketRegime.UNKNOWN
        
    def train_regime_specific_models(self, data: pd.DataFrame, regime_classifications: List[RegimeClassification]):
        """Train separate models for each regime."""
        # Group data by regime
        regime_data = {}
        
        for i, classification in enumerate(regime_classifications):
            regime = classification.market_regime
            
            if regime not in regime_data:
                regime_data[regime] = []
            
            if i < len(data):
                regime_data[regime].append(data.iloc[i])
        
        # Train models for each regime with sufficient data
        min_samples = 50
        
        for regime, regime_samples in regime_data.items():
            if len(regime_samples) >= min_samples:
                regime_df = pd.DataFrame(regime_samples)
                
                # Create and train regime-specific model
                model = self.base_predictor_factory()
                
                try:
                    model.train(regime_df)
                    self.regime_models[regime] = model
                    print(f"Trained model for {regime.value} regime with {len(regime_samples)} samples")
                except Exception as e:
                    print(f"Failed to train model for {regime.value} regime: {e}")
    
    def predict_with_regime_adaptation(self, data: pd.DataFrame, 
                                     current_regime: MarketRegime) -> Tuple[float, float]:
        """Make prediction using regime-specific model if available."""
        self.current_regime = current_regime
        
        # Use regime-specific model if available
        if current_regime in self.regime_models:
            model = self.regime_models[current_regime]
            
            try:
                if hasattr(model, 'predict_with_uncertainty'):
                    prediction, uncertainty = model.predict_with_uncertainty(data)
                else:
                    prediction = model.predict(data)
                    uncertainty = 0.1  # Default uncertainty
                
                return prediction, uncertainty
            except Exception as e:
                print(f"Error using regime-specific model for {current_regime.value}: {e}")
        
        # Fallback to general model
        if MarketRegime.SIDEWAYS in self.regime_models:
            model = self.regime_models[MarketRegime.SIDEWAYS]
        elif self.regime_models:
            model = list(self.regime_models.values())[0]
        else:
            # Create default model
            model = self.base_predictor_factory()
            model.train(data)
        
        try:
            if hasattr(model, 'predict_with_uncertainty'):
                prediction, uncertainty = model.predict_with_uncertainty(data)
            else:
                prediction = model.predict(data)
                uncertainty = 0.15  # Higher uncertainty for fallback
            
            return prediction, uncertainty
        except Exception as e:
            print(f"Error with fallback model: {e}")
            return 0.0, 0.5  # Conservative fallback
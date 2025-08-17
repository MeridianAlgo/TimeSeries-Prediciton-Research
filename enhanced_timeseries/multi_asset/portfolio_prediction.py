"""
Portfolio-level prediction and asset ranking system.
Implements portfolio optimization, asset ranking, and correlation-aware position sizing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None
from collections import defaultdict
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class AssetPrediction:
    """Individual asset prediction with confidence metrics."""
    symbol: str
    expected_return: float
    return_std: float
    confidence_score: float
    prediction_horizon: int  # days
    model_ensemble_agreement: float
    factor_exposure: Dict[str, float]
    sector: Optional[str] = None
    market_cap_rank: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'expected_return': self.expected_return,
            'return_std': self.return_std,
            'confidence_score': self.confidence_score,
            'prediction_horizon': self.prediction_horizon,
            'model_ensemble_agreement': self.model_ensemble_agreement,
            'factor_exposure': self.factor_exposure,
            'sector': self.sector,
            'market_cap_rank': self.market_cap_rank
        }


@dataclass
class PortfolioPrediction:
    """Portfolio-level prediction results."""
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk
    portfolio_beta: float
    diversification_ratio: float
    concentration_risk: float
    prediction_confidence: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'expected_return': self.expected_return,
            'expected_volatility': self.expected_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown_estimate': self.max_drawdown_estimate,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'portfolio_beta': self.portfolio_beta,
            'diversification_ratio': self.diversification_ratio,
            'concentration_risk': self.concentration_risk,
            'prediction_confidence': self.prediction_confidence
        }


@dataclass
class AssetRanking:
    """Asset ranking with multiple criteria."""
    symbol: str
    overall_rank: int
    return_rank: int
    risk_adjusted_rank: int
    confidence_rank: int
    momentum_rank: int
    value_rank: int
    quality_rank: int
    composite_score: float
    ranking_factors: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'overall_rank': self.overall_rank,
            'return_rank': self.return_rank,
            'risk_adjusted_rank': self.risk_adjusted_rank,
            'confidence_rank': self.confidence_rank,
            'momentum_rank': self.momentum_rank,
            'value_rank': self.value_rank,
            'quality_rank': self.quality_rank,
            'composite_score': self.composite_score,
            'ranking_factors': self.ranking_factors
        }


class CovarianceEstimator:
    """Advanced covariance matrix estimation with shrinkage and robust methods."""
    
    def __init__(self, method: str = 'ledoit_wolf', lookback_window: int = 252):
        """
        Initialize covariance estimator.
        
        Args:
            method: Estimation method ('empirical', 'ledoit_wolf', 'oas', 'mcd')
            lookback_window: Lookback window for covariance estimation
        """
        self.method = method
        self.lookback_window = lookback_window
        
    def estimate_covariance(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate covariance matrix using specified method.
        
        Args:
            returns_data: DataFrame with asset returns
            
        Returns:
            Covariance matrix
        """
        if returns_data.empty:
            return pd.DataFrame()
        
        # Use recent data for covariance estimation
        recent_data = returns_data.tail(self.lookback_window).dropna()
        
        if len(recent_data) < 30:  # Need minimum observations
            logger.warning(f"Insufficient data for covariance estimation: {len(recent_data)} observations")
            return pd.DataFrame()
        
        try:
            if self.method == 'empirical':
                cov_matrix = recent_data.cov()
            elif self.method == 'ledoit_wolf':
                lw = LedoitWolf()
                cov_array = lw.fit(recent_data.values).covariance_
                cov_matrix = pd.DataFrame(cov_array, index=recent_data.columns, columns=recent_data.columns)
            elif self.method == 'oas':
                # Oracle Approximating Shrinkage
                from sklearn.covariance import OAS
                oas = OAS()
                cov_array = oas.fit(recent_data.values).covariance_
                cov_matrix = pd.DataFrame(cov_array, index=recent_data.columns, columns=recent_data.columns)
            else:
                # Default to empirical
                cov_matrix = recent_data.cov()
            
            return cov_matrix.fillna(0)
            
        except Exception as e:
            logger.error(f"Covariance estimation failed: {e}")
            return recent_data.cov().fillna(0)
    
    def estimate_correlation_matrix(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Estimate correlation matrix."""
        cov_matrix = self.estimate_covariance(returns_data)
        
        if cov_matrix.empty:
            return pd.DataFrame()
        
        # Convert covariance to correlation
        std_devs = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        
        return corr_matrix.fillna(0)


class PortfolioOptimizer:
    """Portfolio optimization with multiple objective functions and constraints."""
    
    def __init__(self, 
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.2,
                 min_weight: float = 0.0,
                 transaction_cost: float = 0.001):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_aversion: Risk aversion parameter (higher = more conservative)
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            transaction_cost: Transaction cost as fraction of trade value
        """
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.transaction_cost = transaction_cost
        
    def optimize_portfolio(self, 
                          expected_returns: pd.Series,
                          covariance_matrix: pd.DataFrame,
                          current_weights: Optional[pd.Series] = None,
                          objective: str = 'max_sharpe') -> Dict[str, Any]:
        """
        Optimize portfolio weights.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            current_weights: Current portfolio weights (for transaction costs)
            objective: Optimization objective ('max_sharpe', 'min_variance', 'max_return')
            
        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        if expected_returns.empty or covariance_matrix.empty:
            return {}
        
        # Align data
        common_assets = expected_returns.index.intersection(covariance_matrix.index)
        if len(common_assets) == 0:
            return {}
        
        mu = expected_returns.loc[common_assets].values
        Sigma = covariance_matrix.loc[common_assets, common_assets].values
        n_assets = len(common_assets)
        
        # Ensure covariance matrix is positive definite
        try:
            np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            # Add regularization
            Sigma += np.eye(n_assets) * 1e-6
        
        try:
            if objective == 'max_sharpe':
                result = self._optimize_max_sharpe(mu, Sigma, n_assets, current_weights, common_assets)
            elif objective == 'min_variance':
                result = self._optimize_min_variance(mu, Sigma, n_assets, current_weights, common_assets)
            elif objective == 'max_return':
                result = self._optimize_max_return(mu, Sigma, n_assets, current_weights, common_assets)
            else:
                raise ValueError(f"Unknown objective: {objective}")
            
            if result['success']:
                weights = pd.Series(result['weights'], index=common_assets)
                portfolio_metrics = self._calculate_portfolio_metrics(weights, mu, Sigma)
                
                return {
                    'weights': weights,
                    'expected_return': portfolio_metrics['return'],
                    'expected_volatility': portfolio_metrics['volatility'],
                    'sharpe_ratio': portfolio_metrics['sharpe'],
                    'optimization_success': True,
                    'objective': objective
                }
            else:
                logger.warning(f"Portfolio optimization failed: {result.get('message', 'Unknown error')}")
                return {'optimization_success': False}
                
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {'optimization_success': False}
    
    def _optimize_max_sharpe(self, mu: np.ndarray, Sigma: np.ndarray, 
                           n_assets: int, current_weights: Optional[pd.Series],
                           asset_names: List[str]) -> Dict[str, Any]:
        """Optimize for maximum Sharpe ratio using cvxpy or scipy fallback."""
        if CVXPY_AVAILABLE:
            try:
                # Define variables
                w = cp.Variable(n_assets)
                
                # Portfolio return and risk
                portfolio_return = mu.T @ w
                portfolio_risk = cp.quad_form(w, Sigma)
                
                # Constraints
                constraints = [
                    cp.sum(w) == 1,  # Fully invested
                    w >= self.min_weight,  # Minimum weight
                    w <= self.max_weight   # Maximum weight
                ]
                
                # Add transaction costs if current weights provided
                if current_weights is not None:
                    current_w = current_weights.reindex(asset_names, fill_value=0).values
                    transaction_costs = cp.sum(cp.abs(w - current_w)) * self.transaction_cost
                    portfolio_return -= transaction_costs
                
                # Maximize Sharpe ratio (minimize negative Sharpe)
                # Use auxiliary variable for fractional programming
                kappa = cp.Variable()
                
                # Reformulated constraints for Sharpe ratio maximization
                constraints.extend([
                    mu.T @ w == 1,  # Normalize expected return
                    cp.quad_form(w, Sigma) <= kappa,  # Risk constraint
                    kappa >= 0
                ])
                
                # Minimize risk for unit expected return
                objective = cp.Minimize(kappa)
                
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    weights = w.value
                    # Renormalize weights
                    weights = weights / np.sum(weights)
                    return {'success': True, 'weights': weights}
                else:
                    return {'success': False, 'message': f"Solver status: {problem.status}"}
                    
            except Exception as e:
                logger.warning(f"CVXPY optimization failed, falling back to scipy: {e}")
                return self._optimize_max_sharpe_scipy(mu, Sigma, n_assets, current_weights)
        else:
            # Use scipy fallback
            return self._optimize_max_sharpe_scipy(mu, Sigma, n_assets, current_weights)
    
    def _optimize_max_sharpe_scipy(self, mu: np.ndarray, Sigma: np.ndarray, 
                                 n_assets: int, current_weights: Optional[pd.Series]) -> Dict[str, Any]:
        """Fallback Sharpe ratio optimization using scipy."""
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            
            # Add transaction costs
            if current_weights is not None:
                current_w = current_weights.values if hasattr(current_weights, 'values') else current_weights
                if len(current_w) == len(weights):
                    transaction_costs = np.sum(np.abs(weights - current_w)) * self.transaction_cost
                    portfolio_return -= transaction_costs
            
            return -(portfolio_return / portfolio_vol) if portfolio_vol > 0 else -portfolio_return
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(negative_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {'success': result.success, 'weights': result.x, 'message': result.message}
    
    def _optimize_min_variance_scipy(self, mu: np.ndarray, Sigma: np.ndarray, 
                                   n_assets: int, current_weights: Optional[pd.Series]) -> Dict[str, Any]:
        """Fallback minimum variance optimization using scipy."""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(Sigma, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {'success': result.success, 'weights': result.x, 'message': result.message}
    
    def _optimize_max_return_scipy(self, mu: np.ndarray, Sigma: np.ndarray, 
                                 n_assets: int, current_weights: Optional[pd.Series]) -> Dict[str, Any]:
        """Fallback maximum return optimization using scipy."""
        def negative_return(weights):
            return -np.dot(weights, mu)
        
        def portfolio_variance_constraint(weights):
            # Constraint: portfolio variance <= max_variance
            max_variance = 0.2 ** 2  # 20% volatility limit
            return max_variance - np.dot(weights.T, np.dot(Sigma, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
            {'type': 'ineq', 'fun': portfolio_variance_constraint}  # Risk constraint
        ]
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(negative_return, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return {'success': result.success, 'weights': result.x, 'message': result.message}
    
    def _optimize_min_variance(self, mu: np.ndarray, Sigma: np.ndarray, 
                             n_assets: int, current_weights: Optional[pd.Series],
                             asset_names: List[str]) -> Dict[str, Any]:
        """Optimize for minimum variance."""
        if CVXPY_AVAILABLE:
            try:
                w = cp.Variable(n_assets)
                
                # Minimize portfolio variance
                objective = cp.Minimize(cp.quad_form(w, Sigma))
                
                constraints = [
                    cp.sum(w) == 1,
                    w >= self.min_weight,
                    w <= self.max_weight
                ]
                
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    return {'success': True, 'weights': w.value}
                else:
                    return {'success': False, 'message': f"Solver status: {problem.status}"}
                    
            except Exception as e:
                logger.error(f"CVXPY min variance optimization failed, using scipy: {e}")
                return self._optimize_min_variance_scipy(mu, Sigma, n_assets, current_weights)
        else:
            # Use scipy fallback
            return self._optimize_min_variance_scipy(mu, Sigma, n_assets, current_weights)
    
    def _optimize_max_return(self, mu: np.ndarray, Sigma: np.ndarray, 
                           n_assets: int, current_weights: Optional[pd.Series],
                           asset_names: List[str]) -> Dict[str, Any]:
        """Optimize for maximum return with risk constraint."""
        if CVXPY_AVAILABLE:
            try:
                w = cp.Variable(n_assets)
                
                # Maximize expected return
                objective = cp.Maximize(mu.T @ w)
                
                # Risk constraint (portfolio volatility <= some threshold)
                max_volatility = 0.2  # 20% annual volatility limit
                
                constraints = [
                    cp.sum(w) == 1,
                    w >= self.min_weight,
                    w <= self.max_weight,
                    cp.quad_form(w, Sigma) <= max_volatility**2
                ]
                
                problem = cp.Problem(objective, constraints)
                problem.solve(solver=cp.ECOS, verbose=False)
                
                if problem.status == cp.OPTIMAL:
                    return {'success': True, 'weights': w.value}
                else:
                    return {'success': False, 'message': f"Solver status: {problem.status}"}
                    
            except Exception as e:
                logger.error(f"CVXPY max return optimization failed, using scipy: {e}")
                return self._optimize_max_return_scipy(mu, Sigma, n_assets, current_weights)
        else:
            # Use scipy fallback
            return self._optimize_max_return_scipy(mu, Sigma, n_assets, current_weights)
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, 
                                   mu: np.ndarray, Sigma: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        w = weights.values
        
        portfolio_return = np.dot(w, mu)
        portfolio_variance = np.dot(w.T, np.dot(Sigma, w))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'return': float(portfolio_return),
            'volatility': float(portfolio_volatility),
            'sharpe': float(sharpe_ratio)
        }


class AssetRankingSystem:
    """Multi-factor asset ranking system."""
    
    def __init__(self, 
                 ranking_factors: Dict[str, float] = None,
                 lookback_window: int = 252):
        """
        Initialize asset ranking system.
        
        Args:
            ranking_factors: Weights for different ranking factors
            lookback_window: Lookback window for factor calculation
        """
        self.ranking_factors = ranking_factors or {
            'expected_return': 0.3,
            'risk_adjusted_return': 0.25,
            'confidence': 0.2,
            'momentum': 0.15,
            'quality': 0.1
        }
        self.lookback_window = lookback_window
        
    def rank_assets(self, 
                   asset_predictions: Dict[str, AssetPrediction],
                   historical_data: pd.DataFrame,
                   market_data: Optional[Dict[str, Any]] = None) -> List[AssetRanking]:
        """
        Rank assets based on multiple factors.
        
        Args:
            asset_predictions: Dictionary of asset predictions
            historical_data: Historical return data
            market_data: Additional market data (market cap, fundamentals, etc.)
            
        Returns:
            List of asset rankings sorted by composite score
        """
        if not asset_predictions:
            return []
        
        # Calculate individual ranking factors
        factor_scores = {}
        
        # 1. Expected Return Ranking
        return_scores = self._calculate_return_scores(asset_predictions)
        
        # 2. Risk-Adjusted Return Ranking
        risk_adj_scores = self._calculate_risk_adjusted_scores(asset_predictions)
        
        # 3. Confidence Ranking
        confidence_scores = self._calculate_confidence_scores(asset_predictions)
        
        # 4. Momentum Ranking
        momentum_scores = self._calculate_momentum_scores(asset_predictions, historical_data)
        
        # 5. Quality Ranking
        quality_scores = self._calculate_quality_scores(asset_predictions, historical_data, market_data)
        
        # Combine all factor scores
        all_assets = set(asset_predictions.keys())
        rankings = []
        
        for asset in all_assets:
            # Get individual factor scores
            return_score = return_scores.get(asset, 0.5)
            risk_adj_score = risk_adj_scores.get(asset, 0.5)
            confidence_score = confidence_scores.get(asset, 0.5)
            momentum_score = momentum_scores.get(asset, 0.5)
            quality_score = quality_scores.get(asset, 0.5)
            
            # Calculate composite score
            composite_score = (
                self.ranking_factors['expected_return'] * return_score +
                self.ranking_factors['risk_adjusted_return'] * risk_adj_score +
                self.ranking_factors['confidence'] * confidence_score +
                self.ranking_factors['momentum'] * momentum_score +
                self.ranking_factors['quality'] * quality_score
            )
            
            ranking_factors_dict = {
                'return_score': return_score,
                'risk_adjusted_score': risk_adj_score,
                'confidence_score': confidence_score,
                'momentum_score': momentum_score,
                'quality_score': quality_score
            }
            
            rankings.append({
                'symbol': asset,
                'composite_score': composite_score,
                'ranking_factors': ranking_factors_dict,
                'return_score': return_score,
                'risk_adj_score': risk_adj_score,
                'confidence_score': confidence_score,
                'momentum_score': momentum_score,
                'quality_score': quality_score
            })
        
        # Sort by composite score (descending)
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Create AssetRanking objects with ranks
        asset_rankings = []
        
        # Calculate individual factor ranks
        return_ranks = self._assign_ranks([r['return_score'] for r in rankings])
        risk_adj_ranks = self._assign_ranks([r['risk_adj_score'] for r in rankings])
        confidence_ranks = self._assign_ranks([r['confidence_score'] for r in rankings])
        momentum_ranks = self._assign_ranks([r['momentum_score'] for r in rankings])
        quality_ranks = self._assign_ranks([r['quality_score'] for r in rankings])
        
        for i, ranking in enumerate(rankings):
            asset_ranking = AssetRanking(
                symbol=ranking['symbol'],
                overall_rank=i + 1,
                return_rank=return_ranks[i],
                risk_adjusted_rank=risk_adj_ranks[i],
                confidence_rank=confidence_ranks[i],
                momentum_rank=momentum_ranks[i],
                value_rank=quality_ranks[i],  # Using quality as value proxy
                quality_rank=quality_ranks[i],
                composite_score=ranking['composite_score'],
                ranking_factors=ranking['ranking_factors']
            )
            asset_rankings.append(asset_ranking)
        
        return asset_rankings
    
    def _calculate_return_scores(self, asset_predictions: Dict[str, AssetPrediction]) -> Dict[str, float]:
        """Calculate normalized return scores."""
        returns = [pred.expected_return for pred in asset_predictions.values()]
        
        if len(returns) <= 1:
            return {asset: 0.5 for asset in asset_predictions.keys()}
        
        # Normalize to 0-1 scale
        min_return = min(returns)
        max_return = max(returns)
        
        if max_return == min_return:
            return {asset: 0.5 for asset in asset_predictions.keys()}
        
        scores = {}
        for asset, pred in asset_predictions.items():
            normalized_score = (pred.expected_return - min_return) / (max_return - min_return)
            scores[asset] = max(0.0, min(1.0, normalized_score))
        
        return scores
    
    def _calculate_risk_adjusted_scores(self, asset_predictions: Dict[str, AssetPrediction]) -> Dict[str, float]:
        """Calculate risk-adjusted return scores (Sharpe-like ratios)."""
        sharpe_ratios = []
        
        for pred in asset_predictions.values():
            if pred.return_std > 0:
                sharpe = pred.expected_return / pred.return_std
            else:
                sharpe = pred.expected_return
            sharpe_ratios.append(sharpe)
        
        if len(sharpe_ratios) <= 1:
            return {asset: 0.5 for asset in asset_predictions.keys()}
        
        # Normalize Sharpe ratios
        min_sharpe = min(sharpe_ratios)
        max_sharpe = max(sharpe_ratios)
        
        if max_sharpe == min_sharpe:
            return {asset: 0.5 for asset in asset_predictions.keys()}
        
        scores = {}
        for asset, pred in asset_predictions.items():
            if pred.return_std > 0:
                sharpe = pred.expected_return / pred.return_std
            else:
                sharpe = pred.expected_return
            
            normalized_score = (sharpe - min_sharpe) / (max_sharpe - min_sharpe)
            scores[asset] = max(0.0, min(1.0, normalized_score))
        
        return scores
    
    def _calculate_confidence_scores(self, asset_predictions: Dict[str, AssetPrediction]) -> Dict[str, float]:
        """Calculate confidence scores."""
        return {asset: pred.confidence_score for asset, pred in asset_predictions.items()}
    
    def _calculate_momentum_scores(self, 
                                 asset_predictions: Dict[str, AssetPrediction],
                                 historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum scores based on historical performance."""
        scores = {}
        
        # Calculate momentum for different time horizons
        momentum_windows = [21, 63, 126]  # 1, 3, 6 months
        
        for asset in asset_predictions.keys():
            if asset not in historical_data.columns:
                scores[asset] = 0.5
                continue
            
            asset_data = historical_data[asset].dropna()
            
            if len(asset_data) < max(momentum_windows):
                scores[asset] = 0.5
                continue
            
            # Calculate momentum for each window
            momentum_values = []
            
            for window in momentum_windows:
                if len(asset_data) >= window:
                    recent_return = (asset_data.iloc[-1] / asset_data.iloc[-window] - 1)
                    momentum_values.append(recent_return)
            
            if momentum_values:
                # Average momentum across time horizons
                avg_momentum = np.mean(momentum_values)
                # Convert to 0-1 score (assuming momentum range of -50% to +50%)
                momentum_score = max(0.0, min(1.0, (avg_momentum + 0.5) / 1.0))
                scores[asset] = momentum_score
            else:
                scores[asset] = 0.5
        
        return scores
    
    def _calculate_quality_scores(self, 
                                asset_predictions: Dict[str, AssetPrediction],
                                historical_data: pd.DataFrame,
                                market_data: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate quality scores based on return stability and other factors."""
        scores = {}
        
        for asset in asset_predictions.keys():
            if asset not in historical_data.columns:
                scores[asset] = 0.5
                continue
            
            asset_returns = historical_data[asset].pct_change().dropna()
            
            if len(asset_returns) < 63:  # Need sufficient data
                scores[asset] = 0.5
                continue
            
            # Quality metrics
            quality_factors = []
            
            # 1. Return stability (inverse of volatility)
            volatility = asset_returns.std()
            stability_score = max(0.0, min(1.0, 1 - (volatility / 0.05)))  # Normalize by 5% vol
            quality_factors.append(stability_score)
            
            # 2. Consistency (low drawdowns)
            cumulative_returns = (1 + asset_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min())
            consistency_score = max(0.0, min(1.0, 1 - (max_drawdown / 0.3)))  # Normalize by 30% DD
            quality_factors.append(consistency_score)
            
            # 3. Positive return frequency
            positive_return_freq = (asset_returns > 0).mean()
            quality_factors.append(positive_return_freq)
            
            # 4. Skewness (prefer positive skew)
            skewness = asset_returns.skew()
            skew_score = max(0.0, min(1.0, (skewness + 2) / 4))  # Normalize skew range
            quality_factors.append(skew_score)
            
            # Average quality factors
            quality_score = np.mean(quality_factors)
            scores[asset] = quality_score
        
        return scores
    
    def _assign_ranks(self, scores: List[float]) -> List[int]:
        """Assign ranks based on scores (higher score = better rank)."""
        # Create list of (score, original_index) pairs
        score_index_pairs = [(score, i) for i, score in enumerate(scores)]
        
        # Sort by score (descending)
        score_index_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Assign ranks
        ranks = [0] * len(scores)
        for rank, (score, original_index) in enumerate(score_index_pairs):
            ranks[original_index] = rank + 1
        
        return ranks


class PortfolioPredictionSystem:
    """
    Main system for portfolio-level prediction and asset ranking.
    Integrates individual asset predictions with portfolio optimization and ranking.
    """
    
    def __init__(self,
                 covariance_method: str = 'ledoit_wolf',
                 optimization_objective: str = 'max_sharpe',
                 max_position_size: float = 0.15,
                 min_position_size: float = 0.01,
                 rebalancing_threshold: float = 0.05):
        """
        Initialize portfolio prediction system.
        
        Args:
            covariance_method: Method for covariance estimation
            optimization_objective: Portfolio optimization objective
            max_position_size: Maximum position size per asset
            min_position_size: Minimum position size per asset
            rebalancing_threshold: Threshold for triggering rebalancing
        """
        self.covariance_estimator = CovarianceEstimator(method=covariance_method)
        self.portfolio_optimizer = PortfolioOptimizer(
            max_weight=max_position_size,
            min_weight=min_position_size
        )
        self.ranking_system = AssetRankingSystem()
        
        self.optimization_objective = optimization_objective
        self.rebalancing_threshold = rebalancing_threshold
        
        self.current_portfolio = {}
        self.prediction_history = []
        
    def generate_portfolio_prediction(self,
                                    asset_predictions: Dict[str, AssetPrediction],
                                    historical_data: pd.DataFrame,
                                    current_weights: Optional[pd.Series] = None,
                                    market_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive portfolio prediction.
        
        Args:
            asset_predictions: Individual asset predictions
            historical_data: Historical return data
            current_weights: Current portfolio weights
            market_data: Additional market data
            
        Returns:
            Dictionary with portfolio prediction results
        """
        if not asset_predictions:
            return {'error': 'No asset predictions provided'}
        
        logger.info(f"Generating portfolio prediction for {len(asset_predictions)} assets")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_assets': len(asset_predictions),
            'optimization_objective': self.optimization_objective
        }
        
        try:
            # 1. Asset Ranking
            logger.info("Ranking assets...")
            asset_rankings = self.ranking_system.rank_assets(
                asset_predictions, historical_data, market_data
            )
            
            results['asset_rankings'] = [ranking.to_dict() for ranking in asset_rankings]
            
            # 2. Portfolio Optimization
            logger.info("Optimizing portfolio...")
            
            # Prepare data for optimization
            expected_returns = pd.Series({
                asset: pred.expected_return 
                for asset, pred in asset_predictions.items()
            })
            
            # Estimate covariance matrix
            returns_data = historical_data.pct_change().dropna()
            covariance_matrix = self.covariance_estimator.estimate_covariance(returns_data)
            
            if not covariance_matrix.empty:
                # Optimize portfolio
                optimization_result = self.portfolio_optimizer.optimize_portfolio(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    current_weights=current_weights,
                    objective=self.optimization_objective
                )
                
                if optimization_result.get('optimization_success', False):
                    optimal_weights = optimization_result['weights']
                    
                    # 3. Portfolio-Level Prediction
                    portfolio_prediction = self._calculate_portfolio_prediction(
                        optimal_weights, asset_predictions, covariance_matrix, historical_data
                    )
                    
                    results['portfolio_optimization'] = {
                        'optimal_weights': optimal_weights.to_dict(),
                        'expected_return': optimization_result['expected_return'],
                        'expected_volatility': optimization_result['expected_volatility'],
                        'sharpe_ratio': optimization_result['sharpe_ratio']
                    }
                    
                    results['portfolio_prediction'] = portfolio_prediction.to_dict()
                    
                    # 4. Position Sizing Recommendations
                    position_sizing = self._calculate_position_sizing(
                        optimal_weights, asset_predictions, covariance_matrix
                    )
                    
                    results['position_sizing'] = position_sizing
                    
                    # 5. Risk Analysis
                    risk_analysis = self._calculate_portfolio_risk_metrics(
                        optimal_weights, asset_predictions, covariance_matrix, historical_data
                    )
                    
                    results['risk_analysis'] = risk_analysis
                    
                else:
                    results['error'] = 'Portfolio optimization failed'
                    logger.error("Portfolio optimization failed")
            else:
                results['error'] = 'Covariance matrix estimation failed'
                logger.error("Covariance matrix estimation failed")
        
        except Exception as e:
            logger.error(f"Portfolio prediction failed: {e}")
            results['error'] = str(e)
        
        # Store prediction history
        self.prediction_history.append(results)
        
        # Keep only recent history
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
        
        return results
    
    def _calculate_portfolio_prediction(self,
                                      weights: pd.Series,
                                      asset_predictions: Dict[str, AssetPrediction],
                                      covariance_matrix: pd.DataFrame,
                                      historical_data: pd.DataFrame) -> PortfolioPrediction:
        """Calculate portfolio-level prediction metrics."""
        
        # Portfolio expected return
        portfolio_return = sum(
            weights.get(asset, 0) * pred.expected_return 
            for asset, pred in asset_predictions.items()
        )
        
        # Portfolio volatility
        aligned_weights = weights.reindex(covariance_matrix.index, fill_value=0)
        portfolio_variance = np.dot(aligned_weights.values.T, 
                                  np.dot(covariance_matrix.values, aligned_weights.values))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Estimate maximum drawdown using historical simulation
        returns_data = historical_data.pct_change().dropna()
        if not returns_data.empty and len(weights) > 0:
            # Calculate historical portfolio returns
            common_assets = weights.index.intersection(returns_data.columns)
            if len(common_assets) > 0:
                portfolio_weights = weights.reindex(common_assets, fill_value=0)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()  # Renormalize
                
                historical_portfolio_returns = (returns_data[common_assets] * portfolio_weights).sum(axis=1)
                cumulative_returns = (1 + historical_portfolio_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = abs(drawdowns.min())
            else:
                max_drawdown = 0.1  # Default estimate
        else:
            max_drawdown = 0.1  # Default estimate
        
        # Value at Risk (VaR) and Conditional VaR
        if portfolio_volatility > 0:
            # Assuming normal distribution for simplicity
            var_95 = -stats.norm.ppf(0.05) * portfolio_volatility  # 95% VaR
            cvar_95 = portfolio_volatility * stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05  # Expected shortfall
        else:
            var_95 = 0
            cvar_95 = 0
        
        # Portfolio beta (vs equal-weighted portfolio)
        if not returns_data.empty:
            market_return = returns_data.mean(axis=1)
            common_assets = weights.index.intersection(returns_data.columns)
            if len(common_assets) > 0:
                portfolio_weights = weights.reindex(common_assets, fill_value=0)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
                portfolio_returns = (returns_data[common_assets] * portfolio_weights).sum(axis=1)
                
                if len(portfolio_returns) > 30 and market_return.std() > 0:
                    covariance = np.cov(portfolio_returns, market_return)[0, 1]
                    market_variance = market_return.var()
                    portfolio_beta = covariance / market_variance if market_variance > 0 else 1.0
                else:
                    portfolio_beta = 1.0
            else:
                portfolio_beta = 1.0
        else:
            portfolio_beta = 1.0
        
        # Diversification ratio
        individual_volatilities = []
        for asset in weights.index:
            if asset in asset_predictions:
                individual_volatilities.append(asset_predictions[asset].return_std)
            else:
                individual_volatilities.append(0.02)  # Default volatility
        
        weighted_avg_volatility = np.dot(weights.values, individual_volatilities)
        diversification_ratio = weighted_avg_volatility / portfolio_volatility if portfolio_volatility > 0 else 1.0
        
        # Concentration risk (Herfindahl index)
        concentration_risk = np.sum(weights.values ** 2)
        
        # Prediction confidence (weighted average of individual confidences)
        prediction_confidence = sum(
            weights.get(asset, 0) * pred.confidence_score 
            for asset, pred in asset_predictions.items()
        )
        
        return PortfolioPrediction(
            expected_return=float(portfolio_return),
            expected_volatility=float(portfolio_volatility),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown_estimate=float(max_drawdown),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            portfolio_beta=float(portfolio_beta),
            diversification_ratio=float(diversification_ratio),
            concentration_risk=float(concentration_risk),
            prediction_confidence=float(prediction_confidence)
        )
    
    def _calculate_position_sizing(self,
                                 weights: pd.Series,
                                 asset_predictions: Dict[str, AssetPrediction],
                                 covariance_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation-aware position sizing recommendations."""
        
        position_sizing = {
            'recommended_weights': weights.to_dict(),
            'position_adjustments': {},
            'correlation_adjustments': {},
            'confidence_adjustments': {}
        }
        
        # Adjust positions based on correlation
        correlation_matrix = self.covariance_estimator.estimate_correlation_matrix(
            pd.DataFrame()  # Will use cached covariance matrix
        )
        
        if not correlation_matrix.empty:
            for asset in weights.index:
                if asset in correlation_matrix.index:
                    # Calculate average correlation with other holdings
                    other_assets = [a for a in weights.index if a != asset and weights[a] > 0]
                    
                    if other_assets:
                        correlations = [
                            abs(correlation_matrix.loc[asset, other_asset]) 
                            for other_asset in other_assets 
                            if other_asset in correlation_matrix.columns
                        ]
                        
                        if correlations:
                            avg_correlation = np.mean(correlations)
                            
                            # Reduce position if highly correlated with other holdings
                            correlation_adjustment = 1.0 - (avg_correlation * 0.3)  # Max 30% reduction
                            position_sizing['correlation_adjustments'][asset] = correlation_adjustment
        
        # Adjust positions based on confidence
        for asset, prediction in asset_predictions.items():
            if asset in weights.index:
                confidence_adjustment = 0.5 + (prediction.confidence_score * 0.5)  # 0.5 to 1.0 range
                position_sizing['confidence_adjustments'][asset] = confidence_adjustment
        
        return position_sizing
    
    def _calculate_portfolio_risk_metrics(self,
                                        weights: pd.Series,
                                        asset_predictions: Dict[str, AssetPrediction],
                                        covariance_matrix: pd.DataFrame,
                                        historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics."""
        
        risk_metrics = {}
        
        # Component risk contributions
        aligned_weights = weights.reindex(covariance_matrix.index, fill_value=0)
        portfolio_variance = np.dot(aligned_weights.values.T, 
                                  np.dot(covariance_matrix.values, aligned_weights.values))
        
        if portfolio_variance > 0:
            # Marginal risk contributions
            marginal_contributions = np.dot(covariance_matrix.values, aligned_weights.values)
            risk_contributions = aligned_weights.values * marginal_contributions / portfolio_variance
            
            risk_metrics['risk_contributions'] = dict(zip(aligned_weights.index, risk_contributions))
        
        # Sector concentration
        sector_exposure = defaultdict(float)
        for asset, weight in weights.items():
            if asset in asset_predictions:
                sector = asset_predictions[asset].sector or 'Unknown'
                sector_exposure[sector] += weight
        
        risk_metrics['sector_exposure'] = dict(sector_exposure)
        
        # Tail risk metrics
        returns_data = historical_data.pct_change().dropna()
        if not returns_data.empty:
            common_assets = weights.index.intersection(returns_data.columns)
            if len(common_assets) > 0:
                portfolio_weights = weights.reindex(common_assets, fill_value=0)
                portfolio_weights = portfolio_weights / portfolio_weights.sum()
                
                portfolio_returns = (returns_data[common_assets] * portfolio_weights).sum(axis=1)
                
                # Calculate tail risk metrics
                risk_metrics['skewness'] = float(portfolio_returns.skew())
                risk_metrics['kurtosis'] = float(portfolio_returns.kurt())
                risk_metrics['worst_day'] = float(portfolio_returns.min())
                risk_metrics['best_day'] = float(portfolio_returns.max())
                
                # Downside deviation
                downside_returns = portfolio_returns[portfolio_returns < 0]
                risk_metrics['downside_deviation'] = float(downside_returns.std()) if len(downside_returns) > 0 else 0.0
        
        return risk_metrics
    
    def get_rebalancing_recommendations(self,
                                     current_weights: pd.Series,
                                     target_weights: pd.Series) -> Dict[str, Any]:
        """Generate rebalancing recommendations."""
        
        # Calculate weight differences
        all_assets = current_weights.index.union(target_weights.index)
        current_aligned = current_weights.reindex(all_assets, fill_value=0)
        target_aligned = target_weights.reindex(all_assets, fill_value=0)
        
        weight_differences = target_aligned - current_aligned
        
        # Identify assets that need rebalancing
        rebalancing_needed = abs(weight_differences) > self.rebalancing_threshold
        
        recommendations = {
            'rebalancing_needed': rebalancing_needed.any(),
            'weight_differences': weight_differences.to_dict(),
            'assets_to_buy': weight_differences[weight_differences > self.rebalancing_threshold].to_dict(),
            'assets_to_sell': weight_differences[weight_differences < -self.rebalancing_threshold].to_dict(),
            'total_turnover': abs(weight_differences).sum() / 2  # Divide by 2 to avoid double counting
        }
        
        return recommendations
    
    def export_portfolio_analysis(self, filepath: str):
        """Export portfolio analysis to file."""
        if not self.prediction_history:
            logger.warning("No portfolio predictions to export")
            return
        
        export_data = {
            'prediction_history': self.prediction_history,
            'system_configuration': {
                'covariance_method': self.covariance_estimator.method,
                'optimization_objective': self.optimization_objective,
                'max_position_size': self.portfolio_optimizer.max_weight,
                'min_position_size': self.portfolio_optimizer.min_weight,
                'rebalancing_threshold': self.rebalancing_threshold
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Portfolio analysis exported to {filepath}")
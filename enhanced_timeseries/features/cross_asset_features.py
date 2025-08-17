"""
Cross-asset and regime detection features for enhanced time series prediction.
Implements correlation analysis, sector momentum, market regime detection,
and cross-asset relationship modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class CrossAssetFeatures:
    """Cross-asset relationship and regime detection features."""
    
    def __init__(self):
        self.correlation_cache = {}
        self.regime_cache = {}
        self.sector_mappings = {
            # Technology
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 
            'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
            'TSLA': 'Technology', 'NFLX': 'Technology', 'ADBE': 'Technology',
            
            # Financial
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'GS': 'Financial', 'MS': 'Financial', 'C': 'Financial',
            
            # Healthcare
            'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
            'ABBV': 'Healthcare', 'MRK': 'Healthcare', 'TMO': 'Healthcare',
            
            # Energy
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'EOG': 'Energy', 'SLB': 'Energy', 'PSX': 'Energy',
            
            # Consumer
            'WMT': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer',
            'PEP': 'Consumer', 'COST': 'Consumer', 'HD': 'Consumer',
            
            # Industrial
            'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial',
            'MMM': 'Industrial', 'HON': 'Industrial', 'UPS': 'Industrial'
        }
        
    def calculate_cross_asset_features(self, data_dict: Dict[str, pd.DataFrame], 
                                     target_symbol: str,
                                     lookback_periods: List[int] = [5, 10, 20, 50]) -> Dict[str, np.ndarray]:
        """Calculate cross-asset features for a target symbol."""
        if target_symbol not in data_dict:
            raise ValueError(f"Target symbol {target_symbol} not found in data")
            
        features = {}
        target_data = data_dict[target_symbol]
        
        # Correlation features
        features.update(self._correlation_features(data_dict, target_symbol, lookback_periods))
        
        # Sector momentum features
        features.update(self._sector_momentum_features(data_dict, target_symbol, lookback_periods))
        
        # Market-wide features
        features.update(self._market_wide_features(data_dict, target_symbol, lookback_periods))
        
        # Principal component features
        features.update(self._pca_features(data_dict, target_symbol, lookback_periods))
        
        # Cross-asset volatility features
        features.update(self._cross_volatility_features(data_dict, target_symbol, lookback_periods))
        
        return features
    
    def calculate_regime_features(self, data: pd.DataFrame, 
                                lookback_periods: List[int] = [20, 50, 100]) -> Dict[str, np.ndarray]:
        """Calculate market regime detection features."""
        features = {}
        
        # Market regime classification
        features.update(self._market_regime_classification(data, lookback_periods))
        
        # Volatility regime detection
        features.update(self._volatility_regime_detection(data, lookback_periods))
        
        # Trend regime detection
        features.update(self._trend_regime_detection(data, lookback_periods))
        
        # Structural break detection
        features.update(self._structural_break_detection(data, lookback_periods))
        
        # Regime transition probabilities
        features.update(self._regime_transition_features(data, lookback_periods))
        
        return features
    
    def _correlation_features(self, data_dict: Dict[str, pd.DataFrame], 
                            target_symbol: str, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate correlation-based features."""
        features = {}
        target_returns = data_dict[target_symbol]['Close'].pct_change().fillna(0)
        
        # Get other symbols for correlation analysis
        other_symbols = [s for s in data_dict.keys() if s != target_symbol]
        
        if not other_symbols:
            # Return zero features if no other symbols
            for period in periods:
                features[f'avg_correlation_{period}'] = np.zeros(len(target_returns))
                features[f'max_correlation_{period}'] = np.zeros(len(target_returns))
                features[f'correlation_dispersion_{period}'] = np.zeros(len(target_returns))
            return features
        
        # Calculate rolling correlations
        for period in periods:
            correlations_matrix = []
            
            for other_symbol in other_symbols[:10]:  # Limit to 10 symbols for performance
                if other_symbol in data_dict:
                    other_returns = data_dict[other_symbol]['Close'].pct_change().fillna(0)
                    
                    # Align data
                    min_len = min(len(target_returns), len(other_returns))
                    target_aligned = target_returns.iloc[-min_len:]
                    other_aligned = other_returns.iloc[-min_len:]
                    
                    # Rolling correlation
                    rolling_corr = target_aligned.rolling(period).corr(other_aligned).fillna(0)
                    correlations_matrix.append(rolling_corr.values)
            
            if correlations_matrix:
                correlations_matrix = np.array(correlations_matrix).T
                
                # Pad to match target length
                if correlations_matrix.shape[0] < len(target_returns):
                    padding = np.zeros((len(target_returns) - correlations_matrix.shape[0], 
                                      correlations_matrix.shape[1]))
                    correlations_matrix = np.vstack([padding, correlations_matrix])
                
                # Calculate correlation statistics
                features[f'avg_correlation_{period}'] = np.nanmean(correlations_matrix, axis=1)
                features[f'max_correlation_{period}'] = np.nanmax(correlations_matrix, axis=1)
                features[f'min_correlation_{period}'] = np.nanmin(correlations_matrix, axis=1)
                features[f'correlation_dispersion_{period}'] = np.nanstd(correlations_matrix, axis=1)
                
                # Correlation momentum (change in average correlation)
                avg_corr = features[f'avg_correlation_{period}']
                corr_momentum = np.diff(avg_corr, prepend=avg_corr[0])
                features[f'correlation_momentum_{period}'] = corr_momentum
            else:
                # Fallback if no correlations calculated
                features[f'avg_correlation_{period}'] = np.zeros(len(target_returns))
                features[f'max_correlation_{period}'] = np.zeros(len(target_returns))
                features[f'min_correlation_{period}'] = np.zeros(len(target_returns))
                features[f'correlation_dispersion_{period}'] = np.zeros(len(target_returns))
                features[f'correlation_momentum_{period}'] = np.zeros(len(target_returns))
        
        return features
    
    def _sector_momentum_features(self, data_dict: Dict[str, pd.DataFrame], 
                                target_symbol: str, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate sector momentum features."""
        features = {}
        target_data = data_dict[target_symbol]
        target_sector = self.sector_mappings.get(target_symbol, 'Unknown')
        
        # Group symbols by sector
        sector_groups = {}
        for symbol, data in data_dict.items():
            sector = self.sector_mappings.get(symbol, 'Unknown')
            if sector not in sector_groups:
                sector_groups[sector] = []
            sector_groups[sector].append(symbol)
        
        for period in periods:
            sector_momentum = {}
            
            # Calculate momentum for each sector
            for sector, symbols in sector_groups.items():
                if len(symbols) > 1:  # Need at least 2 symbols for sector analysis
                    sector_returns = []
                    
                    for symbol in symbols:
                        if symbol in data_dict:
                            returns = data_dict[symbol]['Close'].pct_change().fillna(0)
                            momentum = returns.rolling(period).mean().fillna(0)
                            sector_returns.append(momentum.values)
                    
                    if sector_returns:
                        # Average sector momentum
                        min_len = min(len(r) for r in sector_returns)
                        sector_returns_aligned = [r[-min_len:] for r in sector_returns]
                        avg_sector_momentum = np.mean(sector_returns_aligned, axis=0)
                        
                        # Pad to match target length
                        if len(avg_sector_momentum) < len(target_data):
                            padding = np.zeros(len(target_data) - len(avg_sector_momentum))
                            avg_sector_momentum = np.concatenate([padding, avg_sector_momentum])
                        
                        sector_momentum[sector] = avg_sector_momentum
            
            # Features relative to own sector and other sectors
            if target_sector in sector_momentum:
                features[f'own_sector_momentum_{period}'] = sector_momentum[target_sector]
            else:
                features[f'own_sector_momentum_{period}'] = np.zeros(len(target_data))
            
            # Relative performance vs other sectors
            other_sectors = [s for s in sector_momentum.keys() if s != target_sector]
            if other_sectors:
                other_sector_avg = np.mean([sector_momentum[s] for s in other_sectors], axis=0)
                own_momentum = features[f'own_sector_momentum_{period}']
                features[f'sector_relative_momentum_{period}'] = own_momentum - other_sector_avg
            else:
                features[f'sector_relative_momentum_{period}'] = np.zeros(len(target_data))
            
            # Sector dispersion (how much sectors are diverging)
            if len(sector_momentum) > 1:
                sector_values = list(sector_momentum.values())
                sector_dispersion = np.std(sector_values, axis=0)
                features[f'sector_dispersion_{period}'] = sector_dispersion
            else:
                features[f'sector_dispersion_{period}'] = np.zeros(len(target_data))
        
        return features
    
    def _market_wide_features(self, data_dict: Dict[str, pd.DataFrame], 
                            target_symbol: str, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate market-wide features."""
        features = {}
        target_data = data_dict[target_symbol]
        
        # Calculate market-wide statistics
        all_returns = []
        all_volumes = []
        
        for symbol, data in data_dict.items():
            returns = data['Close'].pct_change().fillna(0)
            volume = data['Volume']
            
            all_returns.append(returns.values)
            all_volumes.append(volume.values)
        
        if not all_returns:
            # Return zero features if no data
            for period in periods:
                features[f'market_momentum_{period}'] = np.zeros(len(target_data))
                features[f'market_volatility_{period}'] = np.zeros(len(target_data))
                features[f'market_volume_{period}'] = np.zeros(len(target_data))
            return features
        
        # Align all data to same length
        min_len = min(len(r) for r in all_returns)
        all_returns_aligned = [r[-min_len:] for r in all_returns]
        all_volumes_aligned = [v[-min_len:] for v in all_volumes]
        
        # Market-wide statistics
        market_returns = np.mean(all_returns_aligned, axis=0)
        market_volumes = np.mean(all_volumes_aligned, axis=0)
        
        # Pad to match target length
        if len(market_returns) < len(target_data):
            padding_returns = np.zeros(len(target_data) - len(market_returns))
            padding_volumes = np.zeros(len(target_data) - len(market_volumes))
            market_returns = np.concatenate([padding_returns, market_returns])
            market_volumes = np.concatenate([padding_volumes, market_volumes])
        
        for period in periods:
            # Market momentum
            market_momentum = pd.Series(market_returns).rolling(period).mean().fillna(0)
            features[f'market_momentum_{period}'] = market_momentum.values
            
            # Market volatility
            market_volatility = pd.Series(market_returns).rolling(period).std().fillna(0)
            features[f'market_volatility_{period}'] = market_volatility.values
            
            # Market volume trend
            market_volume_trend = pd.Series(market_volumes).rolling(period).mean().fillna(0)
            features[f'market_volume_{period}'] = market_volume_trend.values
            
            # Market breadth (percentage of stocks moving up)
            breadth = []
            for i in range(len(market_returns)):
                if i >= period:
                    period_returns = [r[i-period:i] for r in all_returns_aligned]
                    positive_count = sum(1 for r in period_returns if np.mean(r) > 0)
                    breadth.append(positive_count / len(period_returns))
                else:
                    breadth.append(0.5)  # Neutral
            
            # Pad breadth to match target length
            if len(breadth) < len(target_data):
                padding = [0.5] * (len(target_data) - len(breadth))
                breadth = padding + breadth
            
            features[f'market_breadth_{period}'] = np.array(breadth)
        
        return features
    
    def _pca_features(self, data_dict: Dict[str, pd.DataFrame], 
                     target_symbol: str, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate principal component features."""
        features = {}
        
        # Collect returns data
        returns_data = {}
        for symbol, data in data_dict.items():
            returns = data['Close'].pct_change().fillna(0)
            returns_data[symbol] = returns.values
        
        if len(returns_data) < 3:  # Need at least 3 assets for meaningful PCA
            for period in periods:
                features[f'pca_loading_1_{period}'] = np.zeros(len(data_dict[target_symbol]))
                features[f'pca_loading_2_{period}'] = np.zeros(len(data_dict[target_symbol]))
                features[f'pca_explained_var_{period}'] = np.zeros(len(data_dict[target_symbol]))
            return features
        
        # Align data
        min_len = min(len(r) for r in returns_data.values())
        aligned_data = {symbol: r[-min_len:] for symbol, r in returns_data.items()}
        
        target_len = len(data_dict[target_symbol])
        
        for period in periods:
            pca_loadings_1 = []
            pca_loadings_2 = []
            explained_var = []
            
            for i in range(period, min_len):
                # Get window data
                window_data = []
                symbols = list(aligned_data.keys())
                
                for symbol in symbols:
                    window_data.append(aligned_data[symbol][i-period:i])
                
                window_data = np.array(window_data).T
                
                # Remove any columns with zero variance
                valid_cols = np.var(window_data, axis=0) > 1e-10
                if np.sum(valid_cols) >= 2:
                    window_data_clean = window_data[:, valid_cols]
                    valid_symbols = [symbols[j] for j in range(len(symbols)) if valid_cols[j]]
                    
                    try:
                        # Fit PCA
                        pca = PCA(n_components=min(2, window_data_clean.shape[1]))
                        pca.fit(window_data_clean)
                        
                        # Get loadings for target symbol
                        if target_symbol in valid_symbols:
                            target_idx = valid_symbols.index(target_symbol)
                            loading_1 = pca.components_[0, target_idx] if pca.n_components_ > 0 else 0
                            loading_2 = pca.components_[1, target_idx] if pca.n_components_ > 1 else 0
                            explained = pca.explained_variance_ratio_[0] if pca.n_components_ > 0 else 0
                        else:
                            loading_1 = 0
                            loading_2 = 0
                            explained = 0
                        
                        pca_loadings_1.append(loading_1)
                        pca_loadings_2.append(loading_2)
                        explained_var.append(explained)
                        
                    except:
                        pca_loadings_1.append(0)
                        pca_loadings_2.append(0)
                        explained_var.append(0)
                else:
                    pca_loadings_1.append(0)
                    pca_loadings_2.append(0)
                    explained_var.append(0)
            
            # Pad to match target length
            padding_size = target_len - len(pca_loadings_1)
            if padding_size > 0:
                pca_loadings_1 = [0] * padding_size + pca_loadings_1
                pca_loadings_2 = [0] * padding_size + pca_loadings_2
                explained_var = [0] * padding_size + explained_var
            
            features[f'pca_loading_1_{period}'] = np.array(pca_loadings_1)
            features[f'pca_loading_2_{period}'] = np.array(pca_loadings_2)
            features[f'pca_explained_var_{period}'] = np.array(explained_var)
        
        return features
    
    def _cross_volatility_features(self, data_dict: Dict[str, pd.DataFrame], 
                                 target_symbol: str, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate cross-asset volatility features."""
        features = {}
        target_returns = data_dict[target_symbol]['Close'].pct_change().fillna(0)
        
        # Collect volatility data from other assets
        other_volatilities = []
        for symbol, data in data_dict.items():
            if symbol != target_symbol:
                returns = data['Close'].pct_change().fillna(0)
                volatility = returns.rolling(20).std().fillna(0)
                other_volatilities.append(volatility.values)
        
        if not other_volatilities:
            for period in periods:
                features[f'cross_volatility_avg_{period}'] = np.zeros(len(target_returns))
                features[f'cross_volatility_max_{period}'] = np.zeros(len(target_returns))
                features[f'volatility_spillover_{period}'] = np.zeros(len(target_returns))
            return features
        
        # Align data
        min_len = min(len(v) for v in other_volatilities)
        other_volatilities_aligned = [v[-min_len:] for v in other_volatilities]
        
        # Pad to match target length
        target_len = len(target_returns)
        if min_len < target_len:
            padding = np.zeros((target_len - min_len, len(other_volatilities_aligned)))
            other_volatilities_aligned = [
                np.concatenate([np.zeros(target_len - min_len), v]) 
                for v in other_volatilities_aligned
            ]
        
        other_vol_matrix = np.array(other_volatilities_aligned).T
        
        for period in periods:
            # Average cross-asset volatility
            avg_cross_vol = np.mean(other_vol_matrix, axis=1)
            features[f'cross_volatility_avg_{period}'] = avg_cross_vol
            
            # Maximum cross-asset volatility
            max_cross_vol = np.max(other_vol_matrix, axis=1)
            features[f'cross_volatility_max_{period}'] = max_cross_vol
            
            # Volatility spillover (correlation between target vol and cross vol)
            target_vol = target_returns.rolling(20).std().fillna(0)
            spillover = []
            
            for i in range(period, len(target_vol)):
                target_window = target_vol.iloc[i-period:i]
                cross_window = avg_cross_vol[i-period:i]
                
                if len(target_window) > 1 and len(cross_window) > 1:
                    corr = np.corrcoef(target_window, cross_window)[0, 1]
                    spillover.append(corr if not np.isnan(corr) else 0)
                else:
                    spillover.append(0)
            
            # Pad spillover
            spillover = [0] * (len(target_vol) - len(spillover)) + spillover
            features[f'volatility_spillover_{period}'] = np.array(spillover)
        
        return features
    
    def _market_regime_classification(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Classify market regimes (bull, bear, sideways)."""
        features = {}
        returns = data['Close'].pct_change().fillna(0)
        
        for period in periods:
            regime_labels = []
            
            for i in range(len(data)):
                if i < period:
                    regime_labels.append(0)  # Neutral/unknown
                else:
                    # Calculate period statistics
                    window_returns = returns.iloc[i-period:i]
                    cumulative_return = (1 + window_returns).prod() - 1
                    volatility = window_returns.std()
                    
                    # Regime classification thresholds
                    bull_threshold = 0.05  # 5% positive return
                    bear_threshold = -0.05  # 5% negative return
                    high_vol_threshold = 0.02  # 2% daily volatility
                    
                    if cumulative_return > bull_threshold and volatility < high_vol_threshold:
                        regime = 1  # Bull market
                    elif cumulative_return < bear_threshold:
                        regime = -1  # Bear market
                    elif volatility > high_vol_threshold:
                        regime = 2  # High volatility/crisis
                    else:
                        regime = 0  # Sideways/neutral
                    
                    regime_labels.append(regime)
            
            features[f'market_regime_{period}'] = np.array(regime_labels)
            
            # Regime persistence (how long in current regime)
            persistence = []
            current_regime = 0
            regime_count = 0
            
            for regime in regime_labels:
                if regime == current_regime:
                    regime_count += 1
                else:
                    current_regime = regime
                    regime_count = 1
                persistence.append(regime_count)
            
            features[f'regime_persistence_{period}'] = np.array(persistence)
        
        return features
    
    def _volatility_regime_detection(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Detect volatility regimes using statistical methods."""
        features = {}
        returns = data['Close'].pct_change().fillna(0)
        
        for period in periods:
            volatility = returns.rolling(period).std().fillna(0)
            
            # Use quantiles to define volatility regimes
            vol_regimes = []
            
            for i in range(len(volatility)):
                if i < period * 2:
                    vol_regimes.append(1)  # Normal regime
                else:
                    # Calculate historical quantiles
                    hist_vol = volatility.iloc[:i]
                    q25 = hist_vol.quantile(0.25)
                    q75 = hist_vol.quantile(0.75)
                    
                    current_vol = volatility.iloc[i]
                    
                    if current_vol > q75:
                        regime = 2  # High volatility
                    elif current_vol < q25:
                        regime = 0  # Low volatility
                    else:
                        regime = 1  # Normal volatility
                    
                    vol_regimes.append(regime)
            
            features[f'volatility_regime_{period}'] = np.array(vol_regimes)
            
            # Volatility regime transitions
            regime_changes = np.diff(vol_regimes, prepend=vol_regimes[0])
            features[f'vol_regime_change_{period}'] = regime_changes
        
        return features
    
    def _trend_regime_detection(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Detect trend regimes."""
        features = {}
        prices = data['Close']
        
        for period in periods:
            trend_regimes = []
            
            for i in range(len(prices)):
                if i < period:
                    trend_regimes.append(0)  # No trend
                else:
                    # Linear regression slope over period
                    window_prices = prices.iloc[i-period:i]
                    x = np.arange(len(window_prices))
                    
                    if len(window_prices) > 1:
                        slope, _, r_value, _, _ = stats.linregress(x, window_prices)
                        
                        # Normalize slope by price level
                        normalized_slope = slope / window_prices.mean()
                        
                        # Trend classification
                        if normalized_slope > 0.001 and r_value**2 > 0.5:
                            regime = 1  # Uptrend
                        elif normalized_slope < -0.001 and r_value**2 > 0.5:
                            regime = -1  # Downtrend
                        else:
                            regime = 0  # No clear trend
                    else:
                        regime = 0
                    
                    trend_regimes.append(regime)
            
            features[f'trend_regime_{period}'] = np.array(trend_regimes)
            
            # Trend strength
            trend_strength = []
            for i in range(len(prices)):
                if i < period:
                    trend_strength.append(0)
                else:
                    window_prices = prices.iloc[i-period:i]
                    x = np.arange(len(window_prices))
                    
                    if len(window_prices) > 1:
                        _, _, r_value, _, _ = stats.linregress(x, window_prices)
                        trend_strength.append(r_value**2)
                    else:
                        trend_strength.append(0)
            
            features[f'trend_strength_{period}'] = np.array(trend_strength)
        
        return features
    
    def _structural_break_detection(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Detect structural breaks in the time series."""
        features = {}
        returns = data['Close'].pct_change().fillna(0)
        
        for period in periods:
            break_indicators = []
            
            for i in range(len(returns)):
                if i < period * 2:
                    break_indicators.append(0)
                else:
                    # Compare recent period with historical period
                    recent_period = returns.iloc[i-period:i]
                    historical_period = returns.iloc[i-period*2:i-period]
                    
                    # Statistical tests for structural break
                    recent_mean = recent_period.mean()
                    historical_mean = historical_period.mean()
                    recent_std = recent_period.std()
                    historical_std = historical_period.std()
                    
                    # Mean shift detection
                    mean_shift = abs(recent_mean - historical_mean) / (historical_std + 1e-8)
                    
                    # Variance shift detection
                    var_ratio = recent_std / (historical_std + 1e-8)
                    
                    # Combined break indicator
                    if mean_shift > 2 or var_ratio > 2 or var_ratio < 0.5:
                        break_indicator = 1
                    else:
                        break_indicator = 0
                    
                    break_indicators.append(break_indicator)
            
            features[f'structural_break_{period}'] = np.array(break_indicators)
        
        return features
    
    def _regime_transition_features(self, data: pd.DataFrame, periods: List[int]) -> Dict[str, np.ndarray]:
        """Calculate regime transition probabilities."""
        features = {}
        returns = data['Close'].pct_change().fillna(0)
        
        # First calculate regimes
        regime_features = self._market_regime_classification(data, periods)
        
        for period in periods:
            regimes = regime_features[f'market_regime_{period}']
            
            # Transition probabilities
            transition_probs = []
            
            for i in range(len(regimes)):
                if i < 50:  # Need history to calculate probabilities
                    transition_probs.append(0.5)
                else:
                    # Calculate historical transition probabilities
                    hist_regimes = regimes[:i]
                    current_regime = regimes[i-1]
                    
                    # Count transitions from current regime
                    transitions = {}
                    total_from_current = 0
                    
                    for j in range(1, len(hist_regimes)):
                        if hist_regimes[j-1] == current_regime:
                            next_regime = hist_regimes[j]
                            transitions[next_regime] = transitions.get(next_regime, 0) + 1
                            total_from_current += 1
                    
                    # Probability of staying in current regime
                    if total_from_current > 0:
                        stay_prob = transitions.get(current_regime, 0) / total_from_current
                    else:
                        stay_prob = 0.5
                    
                    transition_probs.append(stay_prob)
            
            features[f'regime_stay_prob_{period}'] = np.array(transition_probs)
        
        return features
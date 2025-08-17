"""Extreme feature engineer for ultra-precision prediction."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

from ..core.interfaces import FeatureEngineer
from ..core.exceptions import FeatureEngineeringError
from .advanced_bollinger import AdvancedBollingerBands
from .multi_rsi import MultiRSISystem
from .microstructure import MarketMicrostructureAnalyzer
from .volatility_analysis import VolatilityAnalysisSystem


class ExtremeFeatureEngineer(FeatureEngineer):
    """Extreme feature engineer combining multiple advanced feature generation systems."""
    
    def __init__(self, 
                 enable_bollinger: bool = True,
                 enable_rsi: bool = True,
                 enable_microstructure: bool = True,
                 enable_volatility: bool = True):
        """Initialize extreme feature engineer.
        
        Args:
            enable_bollinger: Enable advanced Bollinger Bands features
            enable_rsi: Enable multi-RSI features
            enable_microstructure: Enable market microstructure features
            enable_volatility: Enable volatility analysis features
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize feature generators
        self.generators = {}
        
        if enable_bollinger:
            self.generators['bollinger'] = AdvancedBollingerBands()
            
        if enable_rsi:
            self.generators['rsi'] = MultiRSISystem()
            
        if enable_microstructure:
            self.generators['microstructure'] = MarketMicrostructureAnalyzer()
            
        if enable_volatility:
            self.generators['volatility'] = VolatilityAnalysisSystem()
        
        self.feature_names = []
        self.feature_importance = {}
        
        self.logger.info(f"Initialized ExtremeFeatureEngineer with {len(self.generators)} generators")
    
    def generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate extreme features using all enabled generators.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with all generated features
        """
        try:
            self.logger.info("Generating extreme features")
            
            # Validate input data
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise FeatureEngineeringError(f"Missing required columns: {missing_cols}")
            
            if len(data) == 0:
                raise FeatureEngineeringError("Input data is empty")
            
            # Start with original data
            result_df = data.copy()
            
            # Apply each feature generator
            for name, generator in self.generators.items():
                try:
                    self.logger.info(f"Applying {name} feature generator")
                    result_df = generator.generate_features(result_df)
                    self.logger.info(f"Applied {name} generator - now have {len(result_df.columns)} columns")
                    
                except Exception as e:
                    self.logger.error(f"Error in {name} generator: {str(e)}")
                    # Continue with other generators even if one fails
                    continue
            
            # Update feature names (exclude original columns)
            self.feature_names = [col for col in result_df.columns if col not in data.columns]
            
            # Initialize feature importance
            self.feature_importance = {name: 0.0 for name in self.feature_names}
            
            self.logger.info(f"Generated {len(self.feature_names)} extreme features")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error generating extreme features: {str(e)}")
            raise FeatureEngineeringError(f"Extreme feature generation failed: {str(e)}") from e
    
    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings."""
        return self.feature_importance.copy()
    
    def update_feature_importance(self, importance_dict: Dict[str, float]) -> None:
        """Update feature importance scores."""
        for feature_name, importance in importance_dict.items():
            if feature_name in self.feature_importance:
                self.feature_importance[feature_name] = importance
    
    def get_feature_statistics(self) -> Dict[str, any]:
        """Get statistics about generated features."""
        stats = {
            'total_features': len(self.feature_names),
            'enabled_generators': list(self.generators.keys()),
            'generator_stats': {}
        }
        
        # Get stats from each generator
        for name, generator in self.generators.items():
            if hasattr(generator, 'get_feature_statistics'):
                try:
                    stats['generator_stats'][name] = generator.get_feature_statistics()
                except:
                    stats['generator_stats'][name] = {'error': 'Could not get statistics'}
        
        return stats
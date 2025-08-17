"""Advanced MACD system for ultra-precision prediction"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import FeatureEngineeringError
from ..core.logging_config import setup_logging


class AdvancedMACDSystem:
    """Advanced MACD with multiple configurations and cycle analysis"""
    
    def __init__(self, macd_configs: Optional[List[Tuple[int, int, int]]] = None):
        self.macd_configs = macd_configs or [(8, 17, 9), (12, 26, 9), (19, 39, 9), (5, 13, 8), (21, 55, 13)]
        self.logger = setup_logging()
        self.feature_names = []
    
    def calculate_macd(self, data: pd.Series, fast: int, slow: int, signal: int) -> Dict[str, pd.Series]:
        """Calculate MACD components"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        }
    
    def extract_advanced_macd_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced MACD features"""
        try:
            df = data.copy()
            
            for fast, slow, signal in self.macd_configs:
                macd_data = self.calculate_macd(df['Close'], fast, slow, signal)
                
                # Normalized MACD values
                df[f'macd_{fast}_{slow}'] = macd_data['macd'] / df['Close']
                df[f'macd_signal_{fast}_{slow}'] = macd_data['signal'] / df['Close']
                df[f'macd_histogram_{fast}_{slow}'] = macd_data['histogram'] / df['Close']
                
                # MACD crossover signals
                df[f'macd_cross_{fast}_{slow}'] = (macd_data['macd'] > macd_data['signal']).astype(int)
                
                # MACD momentum and cycles
                df[f'macd_momentum_{fast}_{slow}'] = macd_data['histogram'].diff()
                df[f'macd_cycle_{fast}_{slow}'] = np.sin(2 * np.pi * macd_data['histogram'] / macd_data['histogram'].rolling(50).std())
                
                # MACD velocity and acceleration
                df[f'macd_velocity_{fast}_{slow}'] = macd_data['macd'].diff()
                df[f'macd_acceleration_{fast}_{slow}'] = df[f'macd_velocity_{fast}_{slow}'].diff()
                
                self.feature_names.extend([
                    f'macd_{fast}_{slow}', f'macd_signal_{fast}_{slow}', f'macd_histogram_{fast}_{slow}',
                    f'macd_cross_{fast}_{slow}', f'macd_momentum_{fast}_{slow}', f'macd_cycle_{fast}_{slow}',
                    f'macd_velocity_{fast}_{slow}', f'macd_acceleration_{fast}_{slow}'
                ])
            
            return df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to extract MACD features: {str(e)}")
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()
    
    def get_feature_count(self) -> int:
        return len(self.feature_names)
"""Multi-harmonic encoder for ultra-precision prediction"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..core.exceptions import FeatureEngineeringError
from ..core.logging_config import setup_logging


class MultiHarmonicEncoder:
    """Multi-harmonic encoder for cyclical time features"""
    
    def __init__(self):
        self.logger = setup_logging()
        self.feature_names = []
    
    def extract_multi_harmonic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract multi-harmonic cyclical features"""
        try:
            df = data.copy()
            
            # Time-based features with extreme precision
            if hasattr(df.index, 'hour'):
                df['hour'] = df.index.hour
                df['minute'] = df.index.minute
            else:
                df['hour'] = 12  # Default for daily data
                df['minute'] = 0
            
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding with multiple harmonics
            for harmonic in [1, 2, 3]:
                df[f'day_sin_{harmonic}'] = np.sin(2 * np.pi * harmonic * df['day_of_week'] / 7)
                df[f'day_cos_{harmonic}'] = np.cos(2 * np.pi * harmonic * df['day_of_week'] / 7)
                df[f'month_sin_{harmonic}'] = np.sin(2 * np.pi * harmonic * df['month'] / 12)
                df[f'month_cos_{harmonic}'] = np.cos(2 * np.pi * harmonic * df['month'] / 12)
                
                self.feature_names.extend([
                    f'day_sin_{harmonic}', f'day_cos_{harmonic}',
                    f'month_sin_{harmonic}', f'month_cos_{harmonic}'
                ])
            
            return df
            
        except Exception as e:
            raise FeatureEngineeringError(f"Failed to extract harmonic features: {str(e)}")
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()
    
    def get_feature_count(self) -> int:
        return len(self.feature_names)
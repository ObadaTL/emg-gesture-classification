from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any
from scipy import stats

class BaseFeatureExtractor(ABC):
    """Base class for feature extraction from EMG signals"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    @abstractmethod
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG data"""
        pass
    
    def _calculate_mav(self, data: np.ndarray) -> float:
        """Calculate Mean Absolute Value"""
        return np.mean(np.abs(data))
    
    def _calculate_rms(self, data: np.ndarray) -> float:
        """Calculate Root Mean Square"""
        return np.sqrt(np.mean(np.square(data)))
    
    def _calculate_variance(self, data: np.ndarray) -> float:
        """Calculate Variance"""
        return np.var(data)
    
    def _calculate_log_rms(self, data: np.ndarray) -> float:
        """Calculate Log of RMS"""
        return np.log(self._calculate_rms(data))
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate Kurtosis"""
        return stats.kurtosis(data, fisher=False)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate Skewness"""
        return stats.skew(data) 
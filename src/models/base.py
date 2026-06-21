from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

class BaseModel(ABC):
    """Base class for all EMG classification models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
    
    @abstractmethod
    def build(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load model from disk"""
        pass 
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BasePipeline(ABC):
    """Base class for EMG processing and classification pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = None
        self.feature_extractor = None
        self.model = None
        self.scaler = StandardScaler()
        
    @abstractmethod
    def setup(self):
        """Initialize data loader, feature extractor, and model"""
        pass
    
    @abstractmethod
    def _balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Dataset-specific balancing strategy"""
        pass
    
    def run(self):
        """Execute the complete pipeline"""
        print("Starting pipeline execution...")
        
        # Load and preprocess data
        print("Step 1: Loading data...")
        data = self.data_loader.load_raw_data()
        processed_data = self.data_loader.preprocess_data(data)
        
        # Extract features
        print("Step 2: Extracting features...")
        features = self.feature_extractor.extract_features(processed_data)
        
        # Split data
        print("Step 3: Splitting data...")
        X_train, X_test, y_train, y_test = self._prepare_data(features)
        
        # Train model
        print("Step 4: Training model...")
        metrics = self.model.train(X_train, y_train, X_test, y_test)
        
        print("Pipeline execution completed.")
        return metrics
    
    def _prepare_data(self, features: np.ndarray) -> tuple:
        """Prepare data for training"""
        print("Initial feature matrix shape:", features.shape)
        
        # Split features and labels
        X, y = features[:, :-1], features[:, -1]
        y = y.astype(int)
        
        print("Original class distribution:", np.bincount(y))
        
        # Apply dataset-specific balancing
        X_balanced, y_balanced = self._balance_data(X, y)
        
        print("Balanced class distribution:", np.bincount(y_balanced))
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, 
            test_size=0.3, 
            random_state=42,
            stratify=y_balanced
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Apply LDA for dimensionality reduction
        if self.config.get('processing', {}).get('use_lda', False):
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            n_components = min(self.config.get('processing', {}).get('lda_components', 4), 
                            len(np.unique(y_train)) - 1)  # LDA components cannot exceed classes-1
            print(f"\nApplying LDA dimensionality reduction with {n_components} components...")
            lda = LinearDiscriminantAnalysis(n_components=n_components)
            X_train = lda.fit_transform(X_train, y_train)
            X_test = lda.transform(X_test)
            print(f"Feature dimensions after LDA: {X_train.shape[1]}")
        
        # Print shapes after preprocessing
        print("\nAfter preprocessing:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_model(self, path: Path):
        """Save trained model"""
        self.model.save(path)
    
    def load_model(self, path: Path):
        """Load trained model"""
        self.model.load(path) 
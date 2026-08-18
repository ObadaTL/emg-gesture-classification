import json
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..models.tensorflow_classifier import TensorflowClassifier
from ..models.sklearn_classifier import SklearnClassifier
from .base_pipeline import BasePipeline

class DirectPipeline(BasePipeline):
    """Pipeline for training directly on a previously-extracted feature CSV.

    Note: the CSV has no subject/user_id column, so this path cannot do the
    subject-independent split that UCIPipeline uses (see base_pipeline.BasePipeline).
    It falls back to a random stratified split, which is only safe here because this
    pipeline also does no resampling (no BorderlineSMOTE/ENN), so it doesn't carry the
    resample-before-split leakage risk UCIPipeline had - but overlapping windows from
    the same subject can still land on both sides of the split.
    """
    
    def __init__(self, config_path: str, preprocessed_file: str):
        self.config_path = config_path
        self.preprocessed_file = preprocessed_file
        self.config = self._load_config(config_path)
        super().__init__(self.config)
    
    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup(self):
        """Initialize only the model component"""
        # Initialize model based on config
        if self.config['model']['type'].lower() == 'tensorflow':
            self.model = TensorflowClassifier(self.config)
        else:
            self.model = SklearnClassifier(self.config)
        
        # Build the model
        self.model.build()
    
    def run(self):
        """Execute the simplified pipeline"""
        print("Starting pipeline execution...")
        
        # Load preprocessed data
        print("Loading preprocessed data...")
        features = pd.read_csv(self.preprocessed_file)
        
        # Split data
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self._prepare_data(features.values)
        
        # Train model
        print("Training model...")
        metrics = self.model.train(X_train, y_train, X_test, y_test)
        
        print("Pipeline execution completed.")
        return metrics
    
    def run_experiment(self, experiment_name: str):
        """Run a complete experiment with logging"""
        print(f"Starting experiment: {experiment_name}")
        
        # Setup components
        self.setup()
        
        # Run pipeline
        metrics = self.run()
        
        # Save results if needed
        experiment_dir = Path(self.config['paths']['experiments']) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return metrics

    def _prepare_data(self, features: np.ndarray) -> tuple:
        """Prepare data for training without balancing"""
        print("Initial feature matrix shape:", features.shape)
        
        # Split features and labels
        X, y = features[:, :-1], features[:, -1]
        y = y.astype(int)
        
        print("Original class distribution:", np.bincount(y))
        
        # Split into train and test sets without balancing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, 
            random_state=42,
            stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Apply LDA with exactly 4 components
        # lda = LinearDiscriminantAnalysis(n_components=4)
        # X_train = lda.fit_transform(X_train, y_train)
        # X_test = lda.transform(X_test)
        
        # Print shapes after preprocessing
        print("\nAfter preprocessing:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def _save_experiment_results(self, experiment_name: str, metrics: dict):
        """Save experiment results to a file"""
        experiment_dir = Path(self.config['paths']['experiments']) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = experiment_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
        
        # Save model
        model_file = experiment_dir / 'model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f) 
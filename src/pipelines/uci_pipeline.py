from pathlib import Path
import json
from typing import Dict, Any, Tuple
import numpy as np
from imblearn.combine import SMOTETomek
from .base_pipeline import BasePipeline
from ..dataloader.uci_loader import UCIDataLoader
from ..features.uci_extractor import UCIFeatureExtractor
from ..models.tensorflow_classifier import TensorflowClassifier
from ..models.sklearn_classifier import SklearnClassifier

class UCIPipeline(BasePipeline):
    """Pipeline for processing and classifying UCI EMG dataset"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
    
    def setup(self):
        """Initialize pipeline components"""
        # Update config structure before initialization
        if 'model_type' in self.config:
            # Move model_type to correct location in config
            if 'model' not in self.config:
                self.config['model'] = {}
            self.config['model']['type'] = self.config.pop('model_type')
        
        # Ensure feature_dim is in the right place
        if 'feature_dim' not in self.config.get('data', {}):
            self.config.setdefault('data', {})['feature_dim'] = self.config.get('feature_dim')
        
        # Initialize components with updated config
        self.data_loader = UCIDataLoader(self.config)
        self.feature_extractor = UCIFeatureExtractor(self.config)
        
        # Initialize model
        model_type = (self.config.get('model', {}).get('type') or 
                     self.config.get('model', {}).get('test_type', 'tensorflow')).lower()
        
        print(f"Initializing model type: {model_type}")
        if model_type == 'sklearn':
            self.model = SklearnClassifier(self.config)
        elif model_type == 'ensemble':
            from ..models.ensemble_classifier import EnsembleClassifier
            self.model = EnsembleClassifier(self.config)
        else:
            self.model = TensorflowClassifier(self.config)
        
        self.model.build()
    
    def _balance_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """UCI-specific balancing strategy using advanced resampling techniques"""
        print("\nApplying advanced resampling strategy...")
        print("Before balancing - class distribution:", np.bincount(y))
        
        # Get the distribution of classes
        class_counts = np.bincount(y)
        
        # 1. First apply undersampling to the majority class (class 0)
        # Target a more reasonable count - approximately 3x the average of other classes
        other_classes_avg = np.mean([count for i, count in enumerate(class_counts) if i > 0])
        target_maj_count = int(min(class_counts[0], other_classes_avg * 3))
        
        from imblearn.under_sampling import RandomUnderSampler
        print(f"Undersampling majority class (0) from {class_counts[0]} to {target_maj_count} samples")
        
        # Create a specific sampling strategy for undersampling only class 0
        undersampling_strategy = {0: target_maj_count}
        for i in range(1, len(class_counts)):
            undersampling_strategy[i] = class_counts[i]  # Keep other classes as is
            
        # Apply undersampling
        rus = RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)
        X_undersampled, y_undersampled = rus.fit_resample(X, y)
        
        # 2. Apply Borderline-SMOTE with adaptive k-neighbors
        from imblearn.over_sampling import BorderlineSMOTE
        from imblearn.under_sampling import EditedNearestNeighbours
        from imblearn.pipeline import Pipeline
        
        # Calculate target counts for oversampling
        # Dynamic scaling - very small classes get more aggressive treatment
        oversampling_strategy = {}
        for i in range(1, len(class_counts)):
            # Very small classes need more samples but still maintain realism
            if class_counts[i] < 50:  # Extremely small (like class 7)
                target = min(target_maj_count, class_counts[i] * 6)  # More aggressive 
            elif class_counts[i] < 200:  # Very small classes
                target = min(target_maj_count, class_counts[i] * 3)  # Moderate increase
            else:
                # Larger minority classes - increase less aggressively
                target = min(target_maj_count, class_counts[i] * 2)  # Conservative increase
            oversampling_strategy[i] = int(target)
            
        print("SMOTE target distribution for minority classes:", oversampling_strategy)
        
        # Adaptive k-neighbors - smaller classes need fewer neighbors
        k_neighbors = max(min(5, min([count for i, count in enumerate(class_counts) if i > 0]) - 1), 2)
        
        print(f"Using Borderline-SMOTE with k_neighbors={k_neighbors}")
        
        # Create a pipeline: BorderlineSMOTE followed by cleaning with ENN
        # This creates better quality synthetic samples and removes ambiguous ones
        resampling_pipeline = Pipeline([
            ('borderline_smote', BorderlineSMOTE(
                sampling_strategy=oversampling_strategy,
                random_state=42,
                k_neighbors=k_neighbors,
                m_neighbors=10,  # More neighbors for determining borderline
                kind='borderline-1'  # More conservative approach
            )),
            ('enn_cleaning', EditedNearestNeighbours(
                sampling_strategy='all',  # Clean all classes
                n_neighbors=3
            ))
        ])
        
        # Apply the resampling pipeline
        X_balanced, y_balanced = resampling_pipeline.fit_resample(X_undersampled, y_undersampled)
        
        print("After balancing - class distribution:", np.bincount(y_balanced))
        return X_balanced, y_balanced
    
    def run_experiment(self, experiment_name: str):
        """Run a complete experiment with logging"""
        print(f"Starting experiment: {experiment_name}")
        
        # Setup components
        self.setup()
        
        # Run pipeline
        metrics = self.run()
        
        # Save results
        self._save_experiment_results(experiment_name, metrics)
        
        return metrics
    
    def _save_experiment_results(self, experiment_name: str, metrics: Dict[str, Any]):
        """Save experiment results and model"""
        experiment_dir = Path(self.config['paths']['experiments']) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = experiment_dir / 'model'
        self.save_model(model_path)
        
        # Save metrics
        metrics_path = experiment_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    def run_hyperparameter_tuning(self, experiment_name: str):
        """Run hyperparameter tuning and then train with the best hyperparameters"""
        print(f"Starting hyperparameter tuning: {experiment_name}")
        
        # Setup components
        self.setup()
        
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
        
        # Perform hyperparameter tuning (only for TensorFlow model)
        if isinstance(self.model, TensorflowClassifier):
            print("Step 4: Performing hyperparameter tuning...")
            tuning_results = self.model.hyperparameter_tuning(X_train, y_train, X_test, y_test)
            
            # Re-build the model with optimized hyperparameters
            print("Step 5: Rebuilding model with optimized hyperparameters...")
            self.model.build()
            
            # Train the model with the optimized hyperparameters
            print("Step 6: Training model with optimized hyperparameters...")
            metrics = self.model.train(X_train, y_train, X_test, y_test)
            
            # Add tuning results to metrics
            metrics['hyperparameter_tuning'] = tuning_results
        else:
            print("Hyperparameter tuning is only supported for TensorFlow models.")
            print("Step 4: Training model with default hyperparameters...")
            metrics = self.model.train(X_train, y_train, X_test, y_test)
        
        # Save results
        self._save_experiment_results(experiment_name, metrics)
        
        print("Hyperparameter tuning and training completed.")
        return metrics
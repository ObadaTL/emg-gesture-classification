import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import copy

from .base import BaseModel
from .tensorflow_classifier import TensorflowClassifier
from .sklearn_classifier import SklearnClassifier

class EnsembleClassifier(BaseModel):
    """Ensemble classifier that combines tensorflow and sklearn models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = []
        self.ensemble_model = None
        self.base_models = []
        self.history = None
        
    def build(self):
        """Initialize all models for the ensemble"""
        print("\nBuilding ensemble classifier...")
        
        # Create copies of the config for each base model
        tf_config = copy.deepcopy(self.config)
        tf_config['model']['type'] = 'tensorflow'
        
        sklearn_config = copy.deepcopy(self.config)
        sklearn_config['model']['type'] = 'sklearn'
        
        # Initialize base models
        self.tf_model = TensorflowClassifier(tf_config)
        self.sklearn_model = SklearnClassifier(sklearn_config)
        
        # Build the base models
        self.tf_model.build()
        self.sklearn_model.build()
        
        # Store base models in a list
        self.base_models = [self.tf_model, self.sklearn_model]
        
        # Define ensemble strategy from config
        ensemble_type = self.config.get('ensemble', {}).get('type', 'stacking')
        print(f"Using ensemble type: {ensemble_type}")
        
        if ensemble_type == 'stacking':
            # We'll define the stacking model during training to use the trained base models
            pass
        else:  # default to voting
            print("Using voting ensemble")
            self.ensemble_type = 'voting'
            
        print("Ensemble model initialized with base models:")
        print("- TensorflowClassifier")
        print("- SklearnClassifier")
    
    def _create_sklearn_wrapper(self, model, X=None, y=None):
        """Create a scikit-learn compatible wrapper for a model"""
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
                self.classes_ = np.array(range(len(model.config['data']['class_labels'])))
                
            def fit(self, X, y):
                # Already trained
                return self
                
            def predict(self, X):
                return self.model.predict(X)
                
            def predict_proba(self, X):
                if isinstance(self.model, TensorflowClassifier):
                    probs = self.model.model.predict(X)
                    return probs
                else:
                    # For sklearn model, use its predict_proba method if available
                    if hasattr(self.model.model, 'predict_proba'):
                        return self.model.model.predict_proba(X)
                    else:
                        # Fallback to one-hot encoding predictions
                        preds = self.model.predict(X)
                        probs = np.zeros((len(preds), len(self.classes_)))
                        for i, p in enumerate(preds):
                            probs[i, p] = 1
                        return probs
            
            # Required by scikit-learn's clone utility
            def get_params(self, deep=True):
                return {"model": self.model}
                
            # Required by scikit-learn's clone utility
            def set_params(self, **parameters):
                for parameter, value in parameters.items():
                    setattr(self, parameter, value)
                return self
        
        return ModelWrapper(model)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train the ensemble model and all base models"""
        print("\nStarting ensemble model training...")
        
        # Train base models on the full training set
        print("\nTraining TensorFlow model...")
        self.tf_model.train(X_train, y_train, X_test, y_test)
        
        print("\nTraining Scikit-learn model...")
        self.sklearn_model.train(X_train, y_train, X_test, y_test)
        
        # Evaluate base models on test set
        tf_preds = self.tf_model.predict(X_test)
        tf_acc = np.mean(tf_preds == y_test)
        
        sklearn_preds = self.sklearn_model.predict(X_test)
        sklearn_acc = np.mean(sklearn_preds == y_test)
        
        # Calculate weights based on accuracy
        total_acc = tf_acc + sklearn_acc
        tf_weight = tf_acc / total_acc
        sklearn_weight = sklearn_acc / total_acc
        
        print(f"\nModel weights based on validation accuracy:")
        print(f"TensorFlow weight: {tf_weight:.4f}")
        print(f"Scikit-learn weight: {sklearn_weight:.4f}")
        
        # Store weights for prediction
        self.model_weights = {
            'tensorflow': tf_weight,
            'sklearn': sklearn_weight
        }
        
        # Use a custom ensemble approach instead of scikit-learn's
        # Create a simple wrapper for our ensemble
        class CustomEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                
            def predict(self, X):
                # Get predictions from all models
                all_preds = [model.predict(X) for model in self.models]
                
                # For each sample, count weighted votes for each class
                n_samples = X.shape[0]
                n_classes = len(self.models[0].config['data']['class_labels'])
                
                # Initialize vote matrix
                votes = np.zeros((n_samples, n_classes))
                
                # Add weighted votes
                for i, (preds, weight) in enumerate(zip(all_preds, self.weights.values())):
                    for j, pred in enumerate(preds):
                        votes[j, pred] += weight
                        
                # Return class with most votes
                return np.argmax(votes, axis=1)
        
        # Create the ensemble
        self.ensemble_model = CustomEnsemble(
            models=[self.tf_model, self.sklearn_model],
            weights=self.model_weights
        )
        
        # Print detailed results
        self._print_training_results(X_train, y_train, X_test, y_test)
        
        # Get predictions for validation set
        y_pred = self.predict(X_test)
        val_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Return metrics
        metrics = {
            'accuracy': val_report['accuracy'],
            'f1_score': val_report['macro avg']['f1-score'],
            'precision': val_report['macro avg']['precision'],
            'recall': val_report['macro avg']['recall']
        }
        
        return metrics
        
    def _print_training_results(self, X_train, y_train, X_test, y_test):
        """Print detailed training results"""
        print("\nEnsemble Results:")
        print("-" * 50)
        
        # Training metrics
        train_preds = self.predict(X_train)
        train_acc = np.mean(train_preds == y_train)
        print(f"\nEnsemble Training Accuracy: {train_acc:.4f}")
        
        # Validation metrics
        val_preds = self.predict(X_test)
        val_acc = np.mean(val_preds == y_test)
        print(f"\nEnsemble Validation Accuracy: {val_acc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, val_preds)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, val_preds))
        
        # Compare with base models
        tf_preds = self.tf_model.predict(X_test)
        tf_acc = np.mean(tf_preds == y_test)
        
        sklearn_preds = self.sklearn_model.predict(X_test)
        sklearn_acc = np.mean(sklearn_preds == y_test)
        
        print(f"\nBase Model Performance:")
        print(f"TensorFlow Accuracy: {tf_acc:.4f}")
        print(f"Scikit-learn Accuracy: {sklearn_acc:.4f}")
        print(f"Ensemble Accuracy: {val_acc:.4f}")
        
        # Improvement
        best_base = max(tf_acc, sklearn_acc)
        improvement = val_acc - best_base
        
        print(f"\nImprovement over best base model: {improvement:.4f} ({improvement*100:.2f}%)")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble model"""
        if self.ensemble_model is None:
            raise ValueError("Model needs to be trained before making predictions")
            
        return self.ensemble_model.predict(X)
    
    def save(self, path: Path):
        """Save model to disk"""
        if not hasattr(self, 'model_weights'):
            raise ValueError("No model to save")
            
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save base models first
        tf_path = path.parent / 'tensorflow_base'
        sklearn_path = path.parent / 'sklearn_base'
        
        self.tf_model.save(tf_path)
        self.sklearn_model.save(sklearn_path)
        
        # Save ensemble weights
        weights_path = path.parent / 'ensemble_weights.json'
        with open(weights_path, 'w') as f:
            import json
            json.dump(self.model_weights, f)
        
        # Save paths for loading
        paths_info = {
            'tf_model': str(tf_path),
            'sklearn_model': str(sklearn_path),
            'weights_path': str(weights_path)
        }
        
        with open(path.parent / 'model_paths.json', 'w') as f:
            import json
            json.dump(paths_info, f)
            
        print(f"Ensemble model saved to {path.parent}")
    
    def load(self, path: Path):
        """Load model from disk"""
        if not path.parent.exists():
            raise ValueError(f"No model directory found at {path.parent}")
            
        # Load paths info
        import json
        with open(path.parent / 'model_paths.json', 'r') as f:
            paths_info = json.load(f)
            
        # Load base models
        self.tf_model = TensorflowClassifier(self.config)
        self.sklearn_model = SklearnClassifier(self.config)
        
        self.tf_model.load(Path(paths_info['tf_model']))
        self.sklearn_model.load(Path(paths_info['sklearn_model']))
        
        # Load ensemble weights
        with open(paths_info['weights_path'], 'r') as f:
            self.model_weights = json.load(f)
        
        # Recreate the custom ensemble
        class CustomEnsemble:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
                
            def predict(self, X):
                # Get predictions from all models
                all_preds = [model.predict(X) for model in self.models]
                
                # For each sample, count weighted votes for each class
                n_samples = X.shape[0]
                n_classes = len(self.models[0].config['data']['class_labels'])
                
                # Initialize vote matrix
                votes = np.zeros((n_samples, n_classes))
                
                # Add weighted votes
                for i, (preds, weight) in enumerate(zip(all_preds, self.weights.values())):
                    for j, pred in enumerate(preds):
                        votes[j, pred] += weight
                        
                # Return class with most votes
                return np.argmax(votes, axis=1)
                
        # Create the ensemble
        self.ensemble_model = CustomEnsemble(
            models=[self.tf_model, self.sklearn_model],
            weights=self.model_weights
        )
        
        self.base_models = [self.tf_model, self.sklearn_model]
        
        print(f"Ensemble model loaded from {path.parent}") 
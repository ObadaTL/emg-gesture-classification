from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from .base import BaseModel

class SklearnClassifier(BaseModel):
    """Sklearn-based EMG classifier using MLPClassifier"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.training_loss = []  # Store loss curve
    
    def build(self):
        """Initialize the MLPClassifier using similar architecture to TensorFlow"""
        hidden_layer_sizes = self._get_hidden_layer_sizes()
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=self.config['model']['learning_rate'],
            max_iter=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            activation='relu',
            solver='adam',
            random_state=42,
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.2,  # Use 20% of training data for early stopping
            n_iter_no_change=20,  # Similar to patience in TensorFlow
            verbose=True  # Show training progress
        )
        
        print("\nModel Architecture:")
        print(f"Input Layer: {self.config['data']['feature_dim']} features")
        print(f"Hidden Layers: {hidden_layer_sizes}")
        print(f"Output Layer: {len(self.config['data']['class_labels'])} classes")
    
    def _get_hidden_layer_sizes(self) -> tuple:
        """Get hidden layer sizes from config"""
        layers = self.config['model']['architecture']['layers']
        units = self.config['model']['architecture']['units']
        return tuple(units[:layers])
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model and print detailed results"""
        if self.model is None:
            self.build()
        
        print("\nStarting sklearn model training...")
        self.model.fit(X_train, y_train)
        
        # Store loss curve
        self.training_loss = self.model.loss_curve_
        
        # Print detailed results for visualization
        self._print_training_results(X_train, y_train, X_val, y_val)
        
        # Plot training history
        self._plot_training_history()
        
        # Return metrics based on validation data if provided
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            val_report = classification_report(y_val, y_pred_val, output_dict=True)
            
            # Return metrics in a consistent format using validation data
            metrics = {
                'accuracy': val_report['accuracy'],
                'f1_score': val_report['macro avg']['f1-score'],
                'precision': val_report['macro avg']['precision'],
                'recall': val_report['macro avg']['recall'],
                'best_loss': min(self.training_loss) if self.training_loss else None,
                'n_iterations': len(self.training_loss) if self.training_loss else None
            }
        else:
            # Fallback to training metrics if no validation data
            y_pred_train = self.model.predict(X_train)
            train_report = classification_report(y_train, y_pred_train, output_dict=True)
            
            metrics = {
                'accuracy': train_report['accuracy'],
                'f1_score': train_report['macro avg']['f1-score'],
                'precision': train_report['macro avg']['precision'],
                'recall': train_report['macro avg']['recall'],
                'best_loss': min(self.training_loss) if self.training_loss else None,
                'n_iterations': len(self.training_loss) if self.training_loss else None
            }
        
        return metrics
    
    def _print_training_results(self, X_train, y_train, X_val, y_val):
        """Print detailed training results"""
        print("\nTraining Results:")
        print("-" * 50)
        
        metrics = {}
        
        # Training metrics
        y_pred_train = self.model.predict(X_train)
        train_report = classification_report(y_train, y_pred_train, output_dict=True)
        metrics['train_report'] = train_report
        
        print(f"\nTraining Accuracy: {train_report['accuracy']:.4f}")
        print(f"Final Loss: {self.model.loss_:.4f}")
        
        # Training confusion matrix
        train_cm = confusion_matrix(y_train, y_pred_train)
        print("\nTraining Confusion Matrix:")
        print(train_cm)
        
        print("\nTraining Classification Report:")
        print(classification_report(y_train, y_pred_train))
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            print("\nValidation Results:")
            print("-" * 50)
            
            y_pred_val = self.model.predict(X_val)
            val_report = classification_report(y_val, y_pred_val, output_dict=True)
            metrics['val_report'] = val_report
            
            print(f"\nValidation Accuracy: {val_report['accuracy']:.4f}")
            
            # Validation confusion matrix
            val_cm = confusion_matrix(y_val, y_pred_val)
            print("\nValidation Confusion Matrix:")
            print(val_cm)
            
            print("\nValidation Classification Report:")
            print(classification_report(y_val, y_pred_val))
        
        return metrics
    
    def _plot_training_history(self):
        """Plot training history"""
        if not self.training_loss:
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label='Training Loss')
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('sklearn_training_history.png')
        plt.close()
        
        print(f"\nTraining history plot saved as 'sklearn_training_history.png'")
        print(f"Number of iterations: {len(self.training_loss)}")
        print(f"Best loss: {min(self.training_loss):.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model needs to be trained before making predictions")
        return self.model.predict(X)
    
    def save(self, path: Path):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        # Ensure parent directories exist
        path.parent.mkdir(parents=True, exist_ok=True)
        # Add .joblib extension if not present
        path = Path(str(path) + '.joblib' if not str(path).endswith('.joblib') else str(path))
        
        save_dict = {
            'model': self.model,
            'training_loss': self.training_loss
        }
        joblib.dump(save_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk"""
        # Add .joblib extension if not present
        path = Path(str(path) + '.joblib' if not str(path).endswith('.joblib') else str(path))
        if not path.exists():
            raise ValueError(f"No model file found at {path}")
            
        save_dict = joblib.load(path)
        self.model = save_dict['model']
        self.training_loss = save_dict.get('training_loss', [])
        print(f"Model loaded from {path}") 
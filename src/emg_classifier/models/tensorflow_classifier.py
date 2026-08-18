import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .base import BaseModel

class TensorflowClassifier(BaseModel):
    """Tensorflow-based EMG classifier with optimized architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.history = None
    
    def build(self):
        """Build model with configured architecture"""
        tf.random.set_seed(42)
        
        # Get the actual input dimension from the data
        input_dim = self.config.get('data', {}).get('feature_dim')
        if self.config.get('processing', {}).get('use_lda', False):
            # If using LDA, update input dimension to match LDA components
            input_dim = self.config.get('processing', {}).get('lda_components', 4)
            print(f"\nBuilding model with LDA-reduced input dimension: {input_dim}")
        
        self.model = self._build_model(input_dim)
        self.model.summary()
    
    def _build_model(self, input_dim: int):
        """Build model with configured parameters"""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        # Get architecture parameters from config
        n_layers = self.config['model']['architecture']['layers']
        units = self.config['model']['architecture']['units']
        dropout_rates = self.config['model']['architecture']['dropout_rates']
        learning_rate = self.config['model']['learning_rate']
        
        # Adjust units for smaller input dimension if using LDA
        if self.config.get('processing', {}).get('use_lda', False):
            # Scale down the units proportionally
            scale_factor = input_dim / self.config.get('data', {}).get('feature_dim', 104)
            units = [int(u * scale_factor) for u in units]
            print(f"Adjusted layer units for LDA: {units}")
        
        # Build layers
        for i in range(n_layers):
            model.add(tf.keras.layers.Dense(units[i], activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(dropout_rates[i]))
        
        # Output layer
        model.add(tf.keras.layers.Dense(
            len(self.config['data']['class_labels']), 
            activation='softmax'
        ))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Train the model and print detailed results"""
        print("\nStarting tensorflow model training...")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['model']['epochs'],
            batch_size=self.config['model']['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=self._get_callbacks(),
            verbose=1
        )
        
        # Print detailed results
        self._print_training_results(X_train, y_train, X_test, y_test)
        
        # Plot training history
        self._plot_training_history()
        
        # Get predictions for validation set
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        val_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Return metrics in consistent format
        metrics = {
            'accuracy': val_report['accuracy'],
            'f1_score': val_report['macro avg']['f1-score'],
            'precision': val_report['macro avg']['precision'],
            'recall': val_report['macro avg']['recall']
        }
        
        return metrics
    
    def _print_training_results(self, X_train, y_train, X_test, y_test):
        """Print detailed training results"""
        print("\nTraining Results:")
        print("-" * 50)
        
        # Training metrics
        train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
        print(f"\nTraining Accuracy: {train_acc:.4f}")
        print(f"Training Loss: {train_loss:.4f}")
        
        # Validation metrics
        val_loss, val_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nValidation Accuracy: {val_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Confusion Matrix
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def _plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def _get_callbacks(self):
        """Get callbacks for training"""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=50,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=20,
                min_lr=1e-6
            )
        ]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model needs to be trained before making predictions")
        return np.argmax(self.model.predict(X), axis=1)
    
    def save(self, path: Path):
        """Save model to disk
        
        Args:
            path: Path where to save the model
                 Will append .keras extension if not present
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # Ensure the path has .keras extension
        path = Path(str(path) + '.keras' if not str(path).endswith('.keras') else str(path))
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model from disk
        
        Args:
            path: Path to the saved model
                 Will append .keras extension if not present
        """
        # Ensure the path has .keras extension
        path = Path(str(path) + '.keras' if not str(path).endswith('.keras') else str(path))
        
        if not path.exists():
            raise ValueError(f"No model file found at {path}")
            
        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Perform hyperparameter tuning using Keras Tuner"""
        import keras_tuner as kt
        from tensorflow import keras
        import time
        
        print("\nStarting hyperparameter optimization...")
        
        # Define the model-building function for Keras Tuner
        def build_model(hp):
            model = keras.Sequential()
            
            # Input layer
            model.add(keras.layers.Input(shape=(X_train.shape[1],)))
            
            # Tune number of layers
            n_layers = hp.Int('n_layers', min_value=2, max_value=5, step=1)
            
            # First hidden layer (always present)
            units = hp.Int('units_0', min_value=16, max_value=128, step=16)
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            dropout_rate = hp.Float('dropout_0', min_value=0.0, max_value=0.5, step=0.1)
            model.add(keras.layers.Dropout(dropout_rate))
            
            # Additional hidden layers
            for i in range(1, n_layers):
                units = hp.Int(f'units_{i}', min_value=8, max_value=64, step=8)
                model.add(keras.layers.Dense(units, activation='relu'))
                model.add(keras.layers.BatchNormalization())
                dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.4, step=0.1)
                model.add(keras.layers.Dropout(dropout_rate))
            
            # Output layer
            model.add(keras.layers.Dense(
                len(self.config['data']['class_labels']), 
                activation='softmax'
            ))
            
            # Tune learning rate
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
            
            # Compile model
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        # Set up the tuner
        tuner = kt.Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=100,
            factor=3,
            directory='hyperparameter_tuning',
            project_name='emg_classifier',
            overwrite=True
        )
        
        # Print search space summary
        tuner.search_space_summary()
        
        # Start timing
        start_time = time.time()
        
        # Perform the search
        tuner.search(
            X_train, y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # End timing
        end_time = time.time()
        tuning_time = end_time - start_time
        
        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Build the model with the best hyperparameters
        model = tuner.hypermodel.build(best_hps)
        
        # Print results
        print("\nBest Hyperparameters:")
        print(f"Number of layers: {best_hps.get('n_layers')}")
        print("Units per layer:")
        for i in range(best_hps.get('n_layers')):
            print(f"  Layer {i+1}: {best_hps.get(f'units_{i}')} units")
        print("Dropout rates:")
        for i in range(best_hps.get('n_layers')):
            print(f"  Layer {i+1}: {best_hps.get(f'dropout_{i}')} dropout")
        print(f"Learning rate: {best_hps.get('learning_rate')}")
        
        print(f"\nHyperparameter tuning time: {tuning_time:.2f} seconds")
        
        # Update config with best hyperparameters
        self.config['model']['architecture']['layers'] = best_hps.get('n_layers')
        self.config['model']['architecture']['units'] = [best_hps.get(f'units_{i}') for i in range(best_hps.get('n_layers'))]
        self.config['model']['architecture']['dropout_rates'] = [best_hps.get(f'dropout_{i}') for i in range(best_hps.get('n_layers'))]
        self.config['model']['learning_rate'] = best_hps.get('learning_rate')
        
        # Print updated config
        print("\nUpdated model configuration:")
        print(f"Layers: {self.config['model']['architecture']['layers']}")
        print(f"Units: {self.config['model']['architecture']['units']}")
        print(f"Dropout rates: {self.config['model']['architecture']['dropout_rates']}")
        print(f"Learning rate: {self.config['model']['learning_rate']}")
        
        # Return the best hyperparameters
        return {
            'best_hyperparameters': {
                'n_layers': best_hps.get('n_layers'),
                'units': [best_hps.get(f'units_{i}') for i in range(best_hps.get('n_layers'))],
                'dropout_rates': [best_hps.get(f'dropout_{i}') for i in range(best_hps.get('n_layers'))],
                'learning_rate': best_hps.get('learning_rate')
            },
            'tuning_time': tuning_time
        } 
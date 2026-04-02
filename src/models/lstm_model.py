"""
LSTM Model for Electricity Demand Prediction
Implements a deep learning model using LSTM layers for time-series forecasting.
"""
import numpy as np
import os
import joblib

# Try to import TensorFlow, provide fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Using lightweight fallback mode.")



class ElectricityLSTMModel:
    """
    LSTM-based model for electricity demand forecasting.
    """
    
    def __init__(self, 
                 input_shape=(24, 30),  # (sequence_length, n_features)
                 lstm_units=[128, 64],
                 dense_units=[64, 32],
                 dropout_rate=0.2,
                 learning_rate=0.001):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, n_features)
        lstm_units : list
            Number of units in each LSTM layer
        dense_units : list
            Number of units in each Dense layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            self.lstm_units[0],
            return_sequences=True,
            input_shape=self.input_shape,
            activation='tanh'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                activation='tanh'
            ))
            if return_sequences:
                model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate / 2))
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        print("Model built successfully!")
        self.model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=1,
              model_save_path='models/best_lstm_model.h5'):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity mode (0, 1, or 2)
        model_save_path : str
            Path to save the best model
            
        Returns:
        --------
        history : History object
            Training history
        """
        if self.model is None:
            # Update input shape based on training data
            self.input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model()
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        # Save best model
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    filepath=model_save_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=verbose
                )
            )
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.history = history
        print("\nTraining completed!")
        
        return history
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, verbose=1):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        verbose : int
            Verbosity mode
            
        Returns:
        --------
        metrics : dict
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        metrics = self.model.evaluate(X_test, y_test, verbose=verbose)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, metrics))
    
    def save(self, filepath='models/lstm_model.h5'):
        """Save the entire model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/best_lstm_model.h5'):
        """Load a trained model from disk."""
        model = load_model(filepath)
        lstm_model = ElectricityLSTMModel()
        lstm_model.model = model
        print(f"Model loaded from {filepath}")
        return lstm_model
    
    def get_feature_importance(self, X, y):
        """
        Estimate feature importance using permutation importance.
        Note: This is a simplified version for demonstration.
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        baseline_pred = self.model.predict(X)
        baseline_mse = np.mean((baseline_pred.flatten() - y) ** 2)
        
        importance_scores = []
        n_features = X.shape[-1]
        
        for feat_idx in range(n_features):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, :, feat_idx])
            
            permuted_pred = self.model.predict(X_permuted, verbose=0)
            permuted_mse = np.mean((permuted_pred.flatten() - y) ** 2)
            
            importance = permuted_mse - baseline_mse
            importance_scores.append(importance)
        
        return np.array(importance_scores)


if __name__ == "__main__":
    # Example usage
    print("Creating sample data for model testing...")
    
    # Generate dummy sequence data
    n_samples = 1000
    seq_length = 24
    n_features = 30
    
    X_dummy = np.random.randn(n_samples, seq_length, n_features)
    y_dummy = np.random.randn(n_samples, 1)
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_dummy[:split_idx], X_dummy[split_idx:]
    y_train, y_val = y_dummy[:split_idx], y_dummy[split_idx:]
    
    # Create and train model
    model = ElectricityLSTMModel(
        input_shape=(seq_length, n_features),
        lstm_units=[64, 32],
        dense_units=[32, 16],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    model.build_model()
    
    print("\nTraining model with dummy data...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    print("\nModel training complete!")

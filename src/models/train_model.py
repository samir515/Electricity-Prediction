"""
Model Training Script
End-to-end training pipeline for the electricity demand prediction model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from pipeline.data_generator import generate_synthetic_data
from pipeline.preprocessor import DataPreprocessor, prepare_sequences
from models.lstm_model import ElectricityLSTMModel


def train_pipeline(data_path='data/historical_data.csv',
                   n_days=365,
                   sequence_length=24,
                   test_split=0.15,
                   val_split=0.15,
                   epochs=50,
                   batch_size=32):
    """
    Complete training pipeline for electricity demand prediction.
    
    Parameters:
    -----------
    data_path : str
        Path to historical data CSV
    n_days : int
        Number of days of data to generate/use
    sequence_length : int
        Length of input sequences for LSTM
    test_split : float
        Fraction of data for testing
    val_split : float
        Fraction of training data for validation
    epochs : int
        Number of training epochs
    batch_size : int
        Training batch size
    """
    print("=" * 60)
    print("ELECTRICITY DEMAND PREDICTION - TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Generate or load data
    print("\n[Step 1] Loading/Generating Data...")
    if not os.path.exists(data_path):
        print(f"Data file not found. Generating {n_days} days of synthetic data...")
        df = generate_synthetic_data(n_days=n_days, output_path=data_path)
    else:
        print(f"Loading existing data from {data_path}...")
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        print(f"Loaded {len(df)} records")
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing Data...")
    preprocessor = DataPreprocessor(lookback_hours=sequence_length)
    X_scaled, y_scaled, df_processed = preprocessor.fit_transform(df)
    print(f"Features shape: {X_scaled.shape}")
    
    # Save preprocessor
    preprocessor.save('models/preprocessor.joblib')
    
    # Step 3: Prepare sequences
    print("\n[Step 3] Preparing Sequences for LSTM...")
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, sequence_length=sequence_length)
    print(f"Sequence shape: X={X_seq.shape}, y={y_seq.shape}")
    
    # Step 4: Split data
    print("\n[Step 4] Splitting Data (Train/Val/Test)...")
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_seq, y_seq, test_size=test_split, random_state=42, shuffle=False
    )
    
    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split/(1-test_split), random_state=42, shuffle=False
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 5: Build and train model
    print("\n[Step 5] Building and Training LSTM Model...")
    n_features = X_train.shape[2]
    
    model = ElectricityLSTMModel(
        input_shape=(sequence_length, n_features),
        lstm_units=[128, 64],
        dense_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        model_save_path='models/best_lstm_model.h5'
    )
    
    # Step 6: Evaluate model
    print("\n[Step 6] Evaluating Model on Test Set...")
    test_metrics = model.evaluate(X_test, y_test, verbose=0)
    print("\nTest Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Step 7: Make sample predictions and inverse transform
    print("\n[Step 7] Sample Predictions...")
    sample_predictions_scaled = model.predict(X_test[:10])
    sample_predictions = preprocessor.inverse_transform_target(sample_predictions_scaled)
    sample_actual = preprocessor.inverse_transform_target(y_test[:10])
    
    print("\nSample Predictions vs Actual (MW):")
    for i in range(min(5, len(sample_predictions))):
        print(f"  Predicted: {sample_predictions[i]:.2f} MW, Actual: {sample_actual[i]:.2f} MW")
    
    # Step 8: Plot training history
    print("\n[Step 8] Saving Training Plots...")
    os.makedirs('models/plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mae'], label='Train MAE')
    axes[1].plot(history.history['val_mae'], label='Val MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # MAPE plot
    axes[2].plot(history.history['mape'], label='Train MAPE')
    axes[2].plot(history.history['val_mape'], label='Val MAPE')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAPE (%)')
    axes[2].set_title('Mean Absolute Percentage Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/plots/training_history.png', dpi=150)
    plt.close()
    print("Training history plot saved to models/plots/training_history.png")
    
    # Step 9: Prediction vs Actual plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get more predictions for visualization
    n_samples_plot = min(200, len(X_test))
    predictions_scaled = model.predict(X_test[:n_samples_plot])
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    actual = preprocessor.inverse_transform_target(y_test[:n_samples_plot])
    
    ax.plot(actual[:100], label='Actual Demand', linewidth=2)
    ax.plot(predictions[:100], label='Predicted Demand', linewidth=2, linestyle='--')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Demand (MW)')
    ax.set_title('Electricity Demand: Actual vs Predicted')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/plots/predictions_vs_actual.png', dpi=150)
    plt.close()
    print("Predictions plot saved to models/plots/predictions_vs_actual.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: models/best_lstm_model.h5")
    print(f"Preprocessor saved to: models/preprocessor.joblib")
    print(f"\nFinal Test Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  - {metric_name}: {value:.4f}")
    
    return model, preprocessor, test_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Electricity Demand Prediction Model')
    parser.add_argument('--data', type=str, default='data/historical_data.csv',
                        help='Path to historical data CSV')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days of data to generate')
    parser.add_argument('--seq-length', type=int, default=24,
                        help='Sequence length for LSTM input')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    
    args = parser.parse_args()
    
    model, preprocessor, metrics = train_pipeline(
        data_path=args.data,
        n_days=args.days,
        sequence_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

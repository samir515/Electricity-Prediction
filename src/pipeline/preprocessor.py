"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and normalization for the electricity demand prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from datetime import datetime


class DataPreprocessor:
    """
    Preprocessor for electricity demand data.
    Handles feature engineering, scaling, and data transformation.
    """
    
    def __init__(self, lookback_hours=168):  # Default: 1 week of historical data
        self.lookback_hours = lookback_hours
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None
        self.is_fitted = False
    
    def load_data(self, filepath='data/historical_data.csv'):
        """Load historical data from CSV file."""
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    
    def add_features(self, df):
        """
        Add engineered features to the dataset.
        
        Features include:
        - Cyclical encoding of time variables
        - Rolling statistics
        - Lag features
        - Interaction terms
        """
        df = df.copy()
        
        # Cyclical encoding for hour (sin/cos transformation)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Time since start (for trend detection)
        df['time_index'] = range(len(df))
        
        # Rolling statistics for demand (if available)
        if 'demand_mw' in df.columns:
            # Rolling mean and std for past 24 hours
            df['demand_rolling_mean_24h'] = df['demand_mw'].rolling(window=24, min_periods=1).mean()
            df['demand_rolling_std_24h'] = df['demand_mw'].rolling(window=24, min_periods=1).std()
            
            # Rolling mean for past week (168 hours)
            df['demand_rolling_mean_168h'] = df['demand_mw'].rolling(window=168, min_periods=1).mean()
            
            # Lag features
            for lag in [1, 2, 3, 24, 48, 168]:
                df[f'demand_lag_{lag}h'] = df['demand_mw'].shift(lag)
        
        # Temperature squared (for non-linear relationship)
        df['temp_squared'] = df['temperature_c'] ** 2
        
        # Heat index approximation (simplified)
        df['heat_index'] = df['temperature_c'] + 0.5 * df['humidity_pct'] - 10
        
        # Binary flags for peak hours (e.g., 9-11 AM and 6-9 PM)
        df['is_morning_peak'] = ((df['hour'] >= 9) & (df['hour'] <= 11)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
        df['is_peak_hour'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
        
        # Fill NaN values created by rolling/lag operations
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns for modeling."""
        exclude_cols = ['timestamp', 'demand_mw']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def fit(self, df):
        """
        Fit the preprocessor on training data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Training data with all features
        """
        # Add features
        df_processed = self.add_features(df)
        
        # Get feature columns
        self.feature_columns = self.get_feature_columns(df_processed)
        
        # Fit scalers
        X = df_processed[self.feature_columns].values
        y = df_processed['demand_mw'].values.reshape(-1, 1)
        
        self.feature_scaler.fit(X)
        self.target_scaler.fit(y)
        
        self.is_fitted = True
        print(f"Preprocessor fitted with {len(self.feature_columns)} features")
        
        return self
    
    def transform(self, df):
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
            
        Returns:
        --------
        tuple : (X_scaled, y_scaled, df_processed)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        # Add features
        df_processed = self.add_features(df)
        
        # Extract features and target
        X = df_processed[self.feature_columns].values
        y = df_processed['demand_mw'].values.reshape(-1, 1)
        
        # Scale
        X_scaled = self.feature_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y)
        
        return X_scaled, y_scaled, df_processed
    
    def fit_transform(self, df):
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform_target(self, y_scaled):
        """Inverse transform scaled target values to original scale."""
        if len(y_scaled.shape) == 1:
            y_scaled = y_scaled.reshape(-1, 1)
        return self.target_scaler.inverse_transform(y_scaled).flatten()
    
    def save(self, filepath='models/preprocessor.joblib'):
        """Save the fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/preprocessor.joblib'):
        """Load a preprocessor from disk."""
        preprocessor = joblib.load(filepath)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def prepare_sequences(X, y, sequence_length=24):
    """
    Prepare sequences for LSTM model.
    
    Parameters:
    -----------
    X : np.ndarray
        Scaled feature matrix
    y : np.ndarray
        Scaled target values
    sequence_length : int
        Number of time steps in each sequence
        
    Returns:
    --------
    tuple : (X_seq, y_seq)
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    # Test the preprocessor
    from data_generator import generate_synthetic_data
    
    # Generate sample data
    df = generate_synthetic_data(n_days=100)
    
    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor(lookback_hours=168)
    X_scaled, y_scaled, df_processed = preprocessor.fit_transform(df)
    
    print(f"\nFeature columns ({len(preprocessor.feature_columns)}):")
    print(preprocessor.feature_columns)
    
    print(f"\nOriginal shape: X={X_scaled.shape}, y={y_scaled.shape}")
    
    # Prepare sequences
    X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, sequence_length=24)
    print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    
    # Test inverse transform
    y_original = preprocessor.inverse_transform_target(y_scaled[:5])
    print(f"\nSample inverse transform: {y_original}")
    
    # Save preprocessor
    preprocessor.save()

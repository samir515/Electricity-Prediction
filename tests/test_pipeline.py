"""
Unit Tests for Data Pipeline Module
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from src.pipeline.data_generator import generate_synthetic_data
from src.pipeline.preprocessor import DataPreprocessor, prepare_sequences


class TestDataGenerator:
    """Tests for data generation module."""
    
    def test_generate_synthetic_data(self, tmp_path):
        """Test synthetic data generation."""
        output_file = tmp_path / "test_data.csv"
        
        df = generate_synthetic_data(n_days=7, output_path=str(output_file))
        
        # Check DataFrame structure
        assert len(df) == 7 * 24  # 7 days * 24 hours
        assert 'timestamp' in df.columns
        assert 'demand_mw' in df.columns
        assert 'temperature_c' in df.columns
        assert 'humidity_pct' in df.columns
        
        # Check data validity
        assert df['demand_mw'].min() >= 100  # Minimum demand threshold
        assert df['temperature_c'].notna().all()
        assert df['humidity_pct'].between(0, 100).all()
        
        # Check file was created
        assert output_file.exists()


class TestPreprocessor:
    """Tests for data preprocessing module."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=168, freq='H')
        return pd.DataFrame({
            'timestamp': dates,
            'demand_mw': np.random.normal(500, 50, len(dates)),
            'temperature_c': np.random.normal(20, 5, len(dates)),
            'humidity_pct': np.random.uniform(30, 90, len(dates)),
            'rainfall_mm': np.zeros(len(dates)),
            'wind_speed_kmh': np.random.uniform(5, 30, len(dates)),
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'day_of_year': dates.dayofyear,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'month': dates.month
        })
    
    def test_preprocessor_fit_transform(self, sample_data):
        """Test preprocessor fit and transform."""
        preprocessor = DataPreprocessor(lookback_hours=24)
        X_scaled, y_scaled, df_processed = preprocessor.fit_transform(sample_data)
        
        # Check output shapes
        assert X_scaled.shape[0] == len(sample_data)
        assert y_scaled.shape[0] == len(sample_data)
        
        # Check features were added
        assert 'hour_sin' in df_processed.columns
        assert 'hour_cos' in df_processed.columns
        assert 'dow_sin' in df_processed.columns
        
        # Check scaling
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6) or X_scaled.shape[1] > 0
    
    def test_inverse_transform(self, sample_data):
        """Test inverse transformation of target variable."""
        preprocessor = DataPreprocessor()
        X_scaled, y_scaled, _ = preprocessor.fit_transform(sample_data)
        
        # Inverse transform
        y_original = preprocessor.inverse_transform_target(y_scaled)
        
        # Check values are restored
        assert len(y_original) == len(y_scaled)
        assert np.allclose(y_original, sample_data['demand_mw'].values, rtol=1e-5)
    
    def test_prepare_sequences(self, sample_data):
        """Test sequence preparation for LSTM."""
        preprocessor = DataPreprocessor()
        X_scaled, y_scaled, _ = preprocessor.fit_transform(sample_data)
        
        seq_length = 24
        X_seq, y_seq = prepare_sequences(X_scaled, y_scaled, seq_length)
        
        # Check sequence shapes
        expected_samples = len(X_scaled) - seq_length
        assert X_seq.shape[0] == expected_samples
        assert X_seq.shape[1] == seq_length
        assert y_seq.shape[0] == expected_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

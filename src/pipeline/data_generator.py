"""
Data Generator Module
Generates synthetic historical electricity consumption data with weather conditions and seasonal patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(n_days=365, output_path='data/historical_data.csv'):
    """
    Generate synthetic electricity demand data with realistic patterns.
    
    Parameters:
    -----------
    n_days : int
        Number of days of historical data to generate
    output_path : str
        Path to save the generated CSV file
    
    Returns:
    --------
    pd.DataFrame
        Generated dataset
    """
    np.random.seed(42)
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='H')[:-1]  # Exclude last hour
    
    n_hours = len(dates)
    
    # Base demand (MW) with random variation
    base_demand = 500
    demand = np.random.normal(base_demand, 50, n_hours)
    
    # Time of day pattern (higher during day, lower at night)
    hour_of_day = dates.hour
    daily_pattern = 100 * np.sin((hour_of_day - 6) * np.pi / 12)
    daily_pattern = np.where(daily_pattern > 0, daily_pattern, 0)
    demand += daily_pattern
    
    # Day of week pattern (lower on weekends)
    day_of_week = dates.dayofweek
    weekend_effect = np.where(day_of_week >= 5, -80, 0)
    demand += weekend_effect
    
    # Seasonal pattern (higher in summer/winter for AC/heating)
    day_of_year = dates.dayofyear
    seasonal_pattern = 60 * np.cos((day_of_year - 15) * 2 * np.pi / 365)
    demand += seasonal_pattern
    
    # Temperature simulation (correlated with season)
    avg_temp = 20 + 15 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
    temperature = avg_temp + np.random.normal(0, 5, n_hours)
    
    # Add temperature effect on demand (extreme temps increase demand)
    temp_effect = 5 * np.abs(temperature - 20)
    demand += temp_effect
    
    # Humidity (random with seasonal variation)
    humidity = 60 + 20 * np.sin((day_of_year - 200) * 2 * np.pi / 365) + np.random.normal(0, 10, n_hours)
    humidity = np.clip(humidity, 20, 100)
    
    # Rainfall (mm) - sporadic events
    rainfall = np.zeros(n_hours)
    rain_events = np.random.choice(n_hours, size=int(n_hours * 0.05), replace=False)
    rainfall[rain_events] = np.random.exponential(5, len(rain_events))
    
    # Wind speed (km/h)
    wind_speed = 15 + 10 * np.sin(hour_of_day * np.pi / 12) + np.random.normal(0, 5, n_hours)
    wind_speed = np.clip(wind_speed, 0, 50)
    
    # Festival effects (random special days with increased demand)
    festival_dates = [
        datetime(dates.year.min(), 1, 1),   # New Year
        datetime(dates.year.min(), 7, 4),   # Independence Day
        datetime(dates.year.min(), 12, 25), # Christmas
    ]
    festival_effect = np.zeros(n_hours)
    for festival in festival_dates:
        festival_mask = (dates.date == festival.date())
        festival_effect[festival_mask] = 50
        # Also add effect for hours around the festival
        festival_effect[np.roll(festival_mask, 1)] = 30
        festival_effect[np.roll(festival_mask, -1)] = 30
    demand += festival_effect
    
    # Add some noise
    demand += np.random.normal(0, 20, n_hours)
    
    # Ensure non-negative demand
    demand = np.maximum(demand, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'demand_mw': demand.round(2),
        'temperature_c': temperature.round(2),
        'humidity_pct': humidity.round(2),
        'rainfall_mm': rainfall.round(2),
        'wind_speed_kmh': wind_speed.round(2),
        'hour': hour_of_day,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year,
        'is_weekend': (day_of_week >= 5).astype(int),
        'month': dates.month
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} hours of synthetic data saved to {output_path}")
    
    return df


if __name__ == "__main__":
    df = generate_synthetic_data()
    print("\nData Summary:")
    print(df.describe())
    print("\nFirst few rows:")
    print(df.head())

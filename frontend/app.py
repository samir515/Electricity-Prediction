"""
Streamlit Frontend for Electricity Demand Prediction
Interactive dashboard for operators to visualize predictions and monitor trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests

# Page configuration
st.set_page_config(
    page_title="Electricity Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"


def get_prediction(weather_data, forecast_hours=24):
    """Get prediction from API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={
                "weather": weather_data,
                "forecast_hours": forecast_hours,
                "include_confidence": True
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None


def generate_synthetic_forecast(forecast_hours=24):
    """Generate synthetic forecast data for demo when API is unavailable."""
    now = datetime.now()
    timestamps = [now + timedelta(hours=i+1) for i in range(forecast_hours)]
    
    predictions = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday()
        
        base = 500
        daily_pattern = 100 * np.sin((hour - 6) * np.pi / 12)
        daily_pattern = max(0, daily_pattern)
        weekend_effect = -80 if day_of_week >= 5 else 0
        
        pred_value = base + daily_pattern + weekend_effect + np.random.normal(0, 15)
        pred_value = max(100, pred_value)
        
        predictions.append({
            "timestamp": ts,
            "predicted_demand_mw": round(pred_value, 2),
            "lower_bound_mw": round(pred_value * 0.9, 2),
            "upper_bound_mw": round(pred_value * 1.1, 2)
        })
    
    return {
        "status": "success (demo mode)",
        "generated_at": now.isoformat(),
        "forecast_hours": forecast_hours,
        "predictions": predictions,
        "summary": {
            "min_demand_mw": round(min(p["predicted_demand_mw"] for p in predictions), 2),
            "max_demand_mw": round(max(p["predicted_demand_mw"] for p in predictions), 2),
            "avg_demand_mw": round(np.mean([p["predicted_demand_mw"] for p in predictions]), 2),
            "total_energy_mwh": round(sum(p["predicted_demand_mw"] for p in predictions), 2)
        }
    }


def create_forecast_chart(predictions_df):
    """Create interactive forecast chart using Plotly."""
    fig = go.Figure()
    
    # Main prediction line
    fig.add_trace(go.Scatter(
        x=predictions_df['timestamp'],
        y=predictions_df['predicted_demand_mw'],
        mode='lines+markers',
        name='Predicted Demand',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=8)
    ))
    
    # Confidence interval
    if 'lower_bound_mw' in predictions_df.columns and 'upper_bound_mw' in predictions_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([predictions_df['timestamp'], predictions_df['timestamp'][::-1]]),
            y=pd.concat([predictions_df['upper_bound_mw'], predictions_df['lower_bound_mw'][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 107, 53, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name='Confidence Interval',
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title="Electricity Demand Forecast",
        xaxis_title="Time",
        yaxis_title="Demand (MW)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_hourly_distribution_chart(predictions_df):
    """Create hourly demand distribution chart."""
    predictions_df['hour'] = predictions_df['timestamp'].dt.hour
    
    hourly_stats = predictions_df.groupby('hour')['predicted_demand_mw'].agg(['mean', 'min', 'max']).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hourly_stats['hour'],
        y=hourly_stats['mean'],
        name='Average Demand',
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        title="Hourly Demand Distribution",
        xaxis_title="Hour of Day",
        yaxis_title="Demand (MW)",
        template='plotly_white',
        height=400
    )
    
    return fig


def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1a1a2e;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #FF6B35;
        }
        .metric-label {
            color: #666;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">⚡ Electricity Demand Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for input parameters
    with st.sidebar:
        st.header("📋 Input Parameters")
        
        st.subheader("Weather Conditions")
        temperature = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=45.0,
            value=25.0,
            step=0.5
        )
        
        humidity = st.slider(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=5.0
        )
        
        rainfall = st.slider(
            "Rainfall (mm)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )
        
        wind_speed = st.slider(
            "Wind Speed (km/h)",
            min_value=0.0,
            max_value=100.0,
            value=15.0,
            step=5.0
        )
        
        st.subheader("Forecast Settings")
        forecast_hours = st.select_slider(
            "Forecast Horizon (hours)",
            options=[6, 12, 24, 48, 72, 168],
            value=24
        )
        
        predict_button = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses machine learning to predict short-term electricity demand based on:
        - Historical consumption patterns
        - Weather conditions
        - Seasonal variations
        - Time-based features
        """)
    
    # Main content area
    if predict_button or 'predictions' not in st.session_state:
        with st.spinner("Generating forecast..."):
            weather_data = {
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "rainfall_mm": rainfall,
                "wind_speed_kmh": wind_speed
            }
            
            # Try API first, fall back to synthetic data
            result = get_prediction(weather_data, forecast_hours)
            
            if result is None:
                st.info("ℹ️ Running in demo mode (API not available). Predictions are synthetic.")
                result = generate_synthetic_forecast(forecast_hours)
            
            st.session_state.predictions = result
    
    result = st.session_state.get('predictions')
    
    if result:
        # Convert predictions to DataFrame
        predictions_list = result['predictions']
        predictions_df = pd.DataFrame(predictions_list)
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Summary metrics row
        st.subheader("📊 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Avg Demand",
                value=f"{result['summary']['avg_demand_mw']:,.0f} MW",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Peak Demand",
                value=f"{result['summary']['max_demand_mw']:,.0f} MW",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Min Demand",
                value=f"{result['summary']['min_demand_mw']:,.0f} MW",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Total Energy",
                value=f"{result['summary']['total_energy_mwh']:,.0f} MWh",
                delta=None
            )
        
        st.markdown("---")
        
        # Main forecast chart
        st.subheader("📈 Demand Forecast")
        forecast_chart = create_forecast_chart(predictions_df)
        st.plotly_chart(forecast_chart, use_container_width=True)
        
        # Additional charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🕐 Hourly Distribution")
            hourly_chart = create_hourly_distribution_chart(predictions_df)
            st.plotly_chart(hourly_chart, use_container_width=True)
        
        with col2:
            st.subheader("📋 Prediction Details")
            display_df = predictions_df.copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.rename(columns={
                'predicted_demand_mw': 'Demand (MW)',
                'lower_bound_mw': 'Lower Bound',
                'upper_bound_mw': 'Upper Bound'
            })
            st.dataframe(
                display_df[['timestamp', 'Demand (MW)', 'Lower Bound', 'Upper Bound']],
                use_container_width=True,
                hide_index=True
            )
        
        # Alert section
        st.markdown("---")
        st.subheader("⚠️ Grid Alerts")
        
        peak_threshold = result['summary']['max_demand_mw'] * 0.95
        high_demand_hours = predictions_df[predictions_df['predicted_demand_mw'] > peak_threshold]
        
        if len(high_demand_hours) > 0:
            st.warning(f"""
            **High Demand Alert**: {len(high_demand_hours)} hours are expected to have demand above {peak_threshold:,.0f} MW.
            
            **Peak Hour**: {high_demand_hours.iloc[high_demand_hours['predicted_demand_mw'].idxmax()]['timestamp'].strftime('%Y-%m-%d %H:%M')}
            with {high_demand_hours['predicted_demand_mw'].max():,.0f} MW
            
            Consider activating additional generation capacity during these periods.
            """)
        else:
            st.success("✅ No high-demand alerts. Expected demand is within normal operating range.")
        
        # Last updated info
        st.markdown("---")
        st.caption(f"Last updated: {result['generated_at']} | Status: {result['status']}")
    
    else:
        # Initial state
        st.info("👈 Configure parameters in the sidebar and click 'Generate Prediction' to see the forecast.")


if __name__ == "__main__":
    main()

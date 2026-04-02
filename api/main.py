"""
FastAPI Backend for Electricity Demand Prediction
Provides RESTful endpoints for demand forecasting and model management.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import uvicorn

# Import project modules
from src.pipeline.preprocessor import DataPreprocessor
from src.models.lstm_model import ElectricityLSTMModel


# Initialize FastAPI app
app = FastAPI(
    title="Electricity Demand Prediction API",
    description="Real-time electricity demand forecasting using LSTM models",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models for Request/Response
class WeatherInput(BaseModel):
    temperature_c: float = Field(..., description="Temperature in Celsius")
    humidity_pct: float = Field(..., description="Humidity percentage")
    rainfall_mm: float = Field(default=0.0, description="Rainfall in mm")
    wind_speed_kmh: float = Field(default=0.0, description="Wind speed in km/h")


class PredictionRequest(BaseModel):
    weather: WeatherInput
    forecast_hours: int = Field(default=24, ge=1, le=168, description="Hours to forecast (1-168)")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")


class HourlyPrediction(BaseModel):
    timestamp: str
    predicted_demand_mw: float
    lower_bound_mw: Optional[float] = None
    upper_bound_mw: Optional[float] = None


class PredictionResponse(BaseModel):
    status: str
    generated_at: str
    forecast_hours: int
    predictions: List[HourlyPrediction]
    summary: Dict[str, Any]


class ModelStatus(BaseModel):
    model_loaded: bool
    preprocessor_loaded: bool
    last_prediction: Optional[str] = None
    total_predictions: int = 0


# Global variables for model and state
model = None
preprocessor = None
last_prediction_time = None
prediction_count = 0


def load_models():
    """Load trained model and preprocessor."""
    global model, preprocessor
    
    try:
        # Load preprocessor
        preprocessor_path = 'models/preprocessor.joblib'
        if os.path.exists(preprocessor_path):
            preprocessor = DataPreprocessor.load(preprocessor_path)
        else:
            print(f"Warning: Preprocessor not found at {preprocessor_path}")
        
        # Load model
        model_path = 'models/best_lstm_model.h5'
        if os.path.exists(model_path):
            model = ElectricityLSTMModel.load(model_path)
        else:
            print(f"Warning: Model not found at {model_path}")
        
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    print("Starting Electricity Demand Prediction API...")
    load_models()
    print("API ready!")


@app.get("/", tags=["Root"])
async def root():
    """API welcome endpoint."""
    return {
        "message": "Welcome to the Electricity Demand Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/status", response_model=ModelStatus, tags=["Model"])
async def get_status():
    """Get current model and API status."""
    return ModelStatus(
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
        last_prediction=last_prediction_time.isoformat() if last_prediction_time else None,
        total_predictions=prediction_count
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_demand(request: PredictionRequest):
    """
    Predict electricity demand for the specified forecast period.
    
    This endpoint takes current weather conditions and generates hourly
    demand predictions for the specified number of hours.
    """
    global last_prediction_time, prediction_count
    
    if model is None or preprocessor is not None:
        # For demo purposes, generate synthetic predictions if model not loaded
        return generate_synthetic_predictions(request)
    
    try:
        # Generate future timestamps
        now = datetime.now()
        forecast_hours = request.forecast_hours
        timestamps = [now + timedelta(hours=i+1) for i in range(forecast_hours)]
        
        # Create feature matrix for prediction
        # In a real scenario, you would use recent historical data + forecasted weather
        predictions = []
        
        for i, ts in enumerate(timestamps):
            # Simulate weather forecast (in production, use actual weather API)
            hour = ts.hour
            day_of_week = ts.weekday()
            
            # Simple weather projection with some variation
            temp = request.weather.temperature_c + np.sin(hour * np.pi / 12) * 3
            humidity = request.weather.humidity_pct + np.random.normal(0, 5)
            
            # Create input features (simplified for demo)
            # In production, build proper sequence from historical data
            pred_value = estimate_demand(
                hour=hour,
                day_of_week=day_of_week,
                temperature=temp,
                humidity=humidity,
                month=ts.month
            )
            
            # Add confidence interval (simplified)
            confidence_width = pred_value * 0.1  # 10% confidence interval
            
            predictions.append(HourlyPrediction(
                timestamp=ts.isoformat(),
                predicted_demand_mw=round(pred_value, 2),
                lower_bound_mw=round(pred_value - confidence_width, 2) if request.include_confidence else None,
                upper_bound_mw=round(pred_value + confidence_width, 2) if request.include_confidence else None
            ))
        
        # Update stats
        last_prediction_time = datetime.now()
        prediction_count += 1
        
        # Calculate summary statistics
        pred_values = [p.predicted_demand_mw for p in predictions]
        summary = {
            "min_demand_mw": round(min(pred_values), 2),
            "max_demand_mw": round(max(pred_values), 2),
            "avg_demand_mw": round(np.mean(pred_values), 2),
            "peak_hour": timestamps[pred_values.index(max(pred_values))].isoformat(),
            "total_energy_mwh": round(sum(pred_values), 2)
        }
        
        return PredictionResponse(
            status="success",
            generated_at=datetime.now().isoformat(),
            forecast_hours=forecast_hours,
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_synthetic_predictions(request: PredictionRequest) -> PredictionResponse:
    """Generate synthetic predictions when model is not loaded (for demo)."""
    now = datetime.now()
    forecast_hours = request.forecast_hours
    timestamps = [now + timedelta(hours=i+1) for i in range(forecast_hours)]
    
    predictions = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday()
        
        # Base demand with patterns
        base = 500
        daily_pattern = 100 * np.sin((hour - 6) * np.pi / 12)
        daily_pattern = max(0, daily_pattern)
        weekend_effect = -80 if day_of_week >= 5 else 0
        
        # Temperature effect
        temp_effect = 5 * abs(request.weather.temperature_c - 20)
        
        pred_value = base + daily_pattern + weekend_effect + temp_effect + np.random.normal(0, 20)
        pred_value = max(100, pred_value)
        
        confidence_width = pred_value * 0.1
        
        predictions.append(HourlyPrediction(
            timestamp=ts.isoformat(),
            predicted_demand_mw=round(pred_value, 2),
            lower_bound_mw=round(pred_value - confidence_width, 2) if request.include_confidence else None,
            upper_bound_mw=round(pred_value + confidence_width, 2) if request.include_confidence else None
        ))
    
    pred_values = [p.predicted_demand_mw for p in predictions]
    summary = {
        "min_demand_mw": round(min(pred_values), 2),
        "max_demand_mw": round(max(pred_values), 2),
        "avg_demand_mw": round(np.mean(pred_values), 2),
        "peak_hour": timestamps[pred_values.index(max(pred_values))].isoformat(),
        "total_energy_mwh": round(sum(pred_values), 2)
    }
    
    return PredictionResponse(
        status="success (synthetic - model not loaded)",
        generated_at=datetime.now().isoformat(),
        forecast_hours=forecast_hours,
        predictions=predictions,
        summary=summary
    )


def estimate_demand(hour: int, day_of_week: int, temperature: float, 
                   humidity: float, month: int) -> float:
    """Simple demand estimation function (fallback when model not loaded)."""
    base = 500
    
    # Daily pattern
    daily = 100 * np.sin((hour - 6) * np.pi / 12)
    daily = max(0, daily)
    
    # Weekend effect
    weekend = -80 if day_of_week >= 5 else 0
    
    # Temperature effect
    temp_effect = 5 * abs(temperature - 20)
    
    # Seasonal effect
    seasonal = 60 * np.cos((month - 1) * 2 * np.pi / 12)
    
    return base + daily + weekend + temp_effect + seasonal + np.random.normal(0, 15)


@app.get("/historical-data", tags=["Data"])
async def get_historical_data(days: int = 7):
    """Retrieve recent historical demand data."""
    data_path = 'data/historical_data.csv'
    
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Historical data not found")
    
    try:
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df = df.sort_values('timestamp', ascending=False)
        
        # Get last N days
        cutoff = datetime.now() - timedelta(days=days)
        recent_data = df[df['timestamp'] >= cutoff]
        
        return {
            "status": "success",
            "records": len(recent_data),
            "days": days,
            "data": recent_data[['timestamp', 'demand_mw', 'temperature_c']].to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# Electricity Demand Prediction System

A machine learning-based system that predicts short-term electricity demand for power grids using historical consumption data, weather conditions, and seasonal patterns.

## Features

- **Data Pipeline**: Collects and preprocesses real-time and historical data
- **Time-Series Forecasting**: LSTM-based model for hourly demand predictions
- **Scalable API**: Low-latency backend for serving predictions
- **Frontend Dashboard**: Interactive interface for visualization and monitoring

## Project Structure

```
/workspace
├── data/                 # Data storage (historical, processed)
├── models/               # Trained ML models
├── api/                  # FastAPI backend
├── frontend/             # Streamlit frontend
├── src/
│   ├── pipeline/         # Data collection and preprocessing
│   ├── models/           # ML model definitions and training
│   └── utils/            # Utility functions
├── tests/                # Unit and integration tests
└── requirements.txt      # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
python src/pipeline/data_generator.py
```

### 3. Train the Model

```bash
python src/models/train_model.py
```

### 4. Run the API Server

```bash
python api/main.py
```

### 5. Launch the Frontend

```bash
streamlit run frontend/app.py
```

## Architecture

1. **Data Layer**: Historical consumption, weather data, seasonal features
2. **ML Layer**: LSTM model for time-series forecasting
3. **API Layer**: RESTful endpoints for predictions
4. **UI Layer**: Interactive dashboard for operators

## Usage

Access the API at `http://localhost:8000` and the frontend at `http://localhost:8501`.

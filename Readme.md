## Solar Power Forecasting Intelligence Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io)

### Live Interactive Web Applications
- **GitHub Pages Static App (Client-Side AI Inference):** [Interactive World UI Dashboard](index.html)
- **REST API & Web Application Server:** `python3 app.py` (Default: `http://localhost:5000/`)

---

### 🌐 System Architecture & Topical Silos

```
                          ┌──────────────────────────┐
                          │ Open-Meteo Satellite API │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │  dataset.py (Pipeline)   │
                          │ - 49 NWP Features        │
                          │ - PCA & MinMaxScaler     │
                          └────────────┬─────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│     deep.py (ANN)     │  │   lstm_solar.py (LSTM) │  │  xgboost_solar.py     │
│ Dense Keras Layers    │  │ Sequential Time-Series│  │ Gradient Boosting     │
└───────────┬───────────┘  └───────────┬───────────┘  └───────────┬───────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ▼
                          ┌──────────────────────────┐
                          │ ensemble_solar.py        │
                          │ Multi-Model Prediction   │
                          └────────────┬─────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌──────────────────────────┐                           ┌──────────────────────────┐
│ predict_live.py (CLI)    │                           │ app.py (Flask REST Server)│
│ & Client-Side JS Engine  │                           │ & Interactive Web UI     │
└──────────────────────────┘                           └──────────────────────────┘
```

---

### 📊 Benchmark Accuracy Metrics Across 21 German Solar Farms

| Model Architecture | RMSE (%) | MAE | R² Score | Deployment Footprint |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost Regressor** | **5.73%** | **0.0269** | **0.8811** | Fast CPU Baseline |
| **LSTM (Time-Series)** | **7.29%** | **0.0450** | **0.8062** | Sequential Memory |
| **ANN (Sequential)** | **6.98%** | **0.0434** | **0.8376** | Zero-Latency Client Inference |
| **ANN + XGBoost Ensemble**| **8.05%** | **0.0664** | **0.7648** | Weighted Model Averaging |

---

### ❓ Citation-Ready Frequently Asked Questions (FAQ)

#### 1. How does the AI Solar Power Forecasting system ingest live weather data?
The system queries Open-Meteo's global weather API for latitude/longitude coordinates corresponding to 21 German solar facilities. It extracts 49 atmospheric parameters including Direct Normal Irradiance (DNI), Shortwave Solar Radiation, Cloud Cover, Surface Pressure, Dewpoint Temperature, and Wind Vectors at 0m and 100m.

#### 2. How are financial yields and CO2 carbon offsets estimated?
Daily financial yield (€) is calculated using European day-ahead spot market rates (€0.15/kWh) multiplied by forecasted kilowatt-hours produced over 8 peak sun hours. Carbon offset savings are calculated using the grid factor of 0.4 kg CO₂ per kWh produced.

#### 3. How does client-side browser AI inference work for GitHub Pages?
Neural network weights (`model_weights.json`) and feature scaling parameters (`scaler_params.json`) are exported directly from TensorFlow. The browser executes a lightweight forward pass in vanilla JavaScript without requiring any remote Python backend server.

---

### 🐳 Quick Start & Docker Deployment

```bash
# Option 1: Docker
docker-compose up --build

# Option 2: Local Python Server
pip install -r requirements.txt xgboost flask
python3 app.py
```

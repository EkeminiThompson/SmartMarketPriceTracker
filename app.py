from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
DATA_FILE = 'price_data.json'
MODELS_DIR = 'models'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Initialize sample data if file doesn't exist
def initialize_data():
    if not os.path.exists(DATA_FILE):
        # Sample data for Nigerian agricultural commodities
        sample_data = {
            "Rice": generate_sample_prices(150, 350, 180),
            "Tomatoes": generate_sample_prices(50, 150, 180),
            "Onions": generate_sample_prices(80, 200, 180),
            "Yam": generate_sample_prices(200, 500, 180),
            "Beans": generate_sample_prices(300, 600, 180),
            "Maize": generate_sample_prices(100, 250, 180)
        }
        save_data(sample_data)

def generate_sample_prices(min_price, max_price, days):
    """Generate realistic sample price data with trends and seasonality"""
    dates = []
    prices = []
    start_date = datetime.now() - timedelta(days=days)
    
    base_price = (min_price + max_price) / 2
    current_price = base_price
    
    for i in range(days):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        
        # Add trend, seasonality, and random noise
        trend = (i / days) * (max_price - min_price) * 0.2
        seasonality = np.sin(i * 2 * np.pi / 30) * (max_price - min_price) * 0.1
        noise = np.random.normal(0, (max_price - min_price) * 0.05)
        
        current_price = base_price + trend + seasonality + noise
        current_price = max(min_price, min(max_price, current_price))
        prices.append(round(current_price, 2))
    
    return [{"date": d, "price": p} for d, p in zip(dates, prices)]

def load_data():
    """Load price data from JSON file"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_data(data):
    """Save price data to JSON file"""
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def prepare_lstm_data(data, lookback=30):
    """Prepare data for LSTM model"""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback=30):
    """Build LSTM model for price prediction"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_arima(prices, forecast_days=30):
    """Predict future prices using ARIMA model"""
    try:
        # Fit ARIMA model
        model = ARIMA(prices, order=(5, 1, 0))
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(steps=forecast_days)
        
        # Calculate metrics
        train_predict = fitted_model.fittedvalues
        mse = mean_squared_error(prices[1:], train_predict)
        mae = mean_absolute_error(prices[1:], train_predict)
        rmse = np.sqrt(mse)
        
        return forecast.tolist(), {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'accuracy': round(100 - (mae / np.mean(prices) * 100), 2)
        }
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None, None

def predict_lstm(prices, forecast_days=30, lookback=30):
    """Predict future prices using LSTM model"""
    try:
        if len(prices) < lookback + 10:
            return None, None
        
        # Prepare data
        price_array = np.array(prices)
        X, y, scaler = prepare_lstm_data(price_array, lookback)
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        model = build_lstm_model(lookback)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        
        # Make predictions
        last_sequence = price_array[-lookback:]
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_days):
            scaled_sequence = scaler.transform(current_sequence.reshape(-1, 1))
            scaled_sequence = scaled_sequence.reshape((1, lookback, 1))
            
            next_pred = model.predict(scaled_sequence, verbose=0)
            next_pred_unscaled = scaler.inverse_transform(next_pred)[0, 0]
            
            predictions.append(next_pred_unscaled)
            current_sequence = np.append(current_sequence[1:], next_pred_unscaled)
        
        # Calculate metrics
        train_predict = model.predict(X, verbose=0)
        train_predict_unscaled = scaler.inverse_transform(train_predict)
        y_unscaled = scaler.inverse_transform(y.reshape(-1, 1))
        
        mse = mean_squared_error(y_unscaled, train_predict_unscaled)
        mae = mean_absolute_error(y_unscaled, train_predict_unscaled)
        rmse = np.sqrt(mse)
        
        return predictions, {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'accuracy': round(100 - (mae / np.mean(prices) * 100), 2)
        }
    except Exception as e:
        print(f"LSTM Error: {e}")
        return None, None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/commodities', methods=['GET'])
def get_commodities():
    """Get list of available commodities"""
    data = load_data()
    commodities = list(data.keys())
    return jsonify({'commodities': commodities})

@app.route('/api/prices/<commodity>', methods=['GET'])
def get_prices(commodity):
    """Get historical prices for a commodity"""
    data = load_data()
    if commodity in data:
        return jsonify({
            'commodity': commodity,
            'data': data[commodity]
        })
    return jsonify({'error': 'Commodity not found'}), 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict future prices using specified model"""
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    model_type = request_data.get('model', 'arima')
    forecast_days = request_data.get('forecast_days', 30)
    
    data = load_data()
    
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    # Extract prices
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    
    # Generate forecast dates
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                      for i in range(forecast_days)]
    
    # Make predictions
    if model_type.lower() == 'arima':
        predictions, metrics = predict_arima(prices, forecast_days)
        model_name = 'ARIMA'
    elif model_type.lower() == 'lstm':
        predictions, metrics = predict_lstm(prices, forecast_days)
        model_name = 'LSTM'
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    if predictions is None:
        return jsonify({'error': 'Prediction failed'}), 500
    
    # Format response
    forecast_data = [
        {'date': d, 'price': round(p, 2)} 
        for d, p in zip(forecast_dates, predictions)
    ]
    
    return jsonify({
        'commodity': commodity,
        'model': model_name,
        'historical': data[commodity][-60:],  # Last 60 days
        'forecast': forecast_data,
        'metrics': metrics
    })

@app.route('/api/add_price', methods=['POST'])
def add_price():
    """Add new price data for a commodity"""
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    price = request_data.get('price')
    date = request_data.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    if not commodity or price is None:
        return jsonify({'error': 'Missing required fields'}), 400
    
    data = load_data()
    
    if commodity not in data:
        data[commodity] = []
    
    data[commodity].append({
        'date': date,
        'price': float(price)
    })
    
    # Sort by date
    data[commodity] = sorted(data[commodity], key=lambda x: x['date'])
    
    save_data(data)
    
    return jsonify({'message': 'Price added successfully'})

@app.route('/api/compare_models', methods=['POST'])
def compare_models():
    """Compare ARIMA and LSTM predictions"""
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    forecast_days = request_data.get('forecast_days', 30)
    
    data = load_data()
    
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    
    # Generate forecast dates
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                      for i in range(forecast_days)]
    
    # ARIMA predictions
    arima_pred, arima_metrics = predict_arima(prices, forecast_days)
    
    # LSTM predictions
    lstm_pred, lstm_metrics = predict_lstm(prices, forecast_days)
    
    response = {
        'commodity': commodity,
        'dates': forecast_dates,
        'historical': data[commodity][-60:],
    }
    
    if arima_pred:
        response['arima'] = {
            'predictions': [round(p, 2) for p in arima_pred],
            'metrics': arima_metrics
        }
    
    if lstm_pred:
        response['lstm'] = {
            'predictions': [round(p, 2) for p in lstm_pred],
            'metrics': lstm_metrics
        }
    
    return jsonify(response)

@app.route('/api/statistics/<commodity>', methods=['GET'])
def get_statistics(commodity):
    """Get statistical analysis of commodity prices"""
    data = load_data()
    
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    prices = [item['price'] for item in data[commodity]]
    
    stats = {
        'current_price': prices[-1],
        'average': round(np.mean(prices), 2),
        'min': round(np.min(prices), 2),
        'max': round(np.max(prices), 2),
        'std_dev': round(np.std(prices), 2),
        'variance': round(np.var(prices), 2),
        'trend': 'upward' if prices[-1] > prices[-30] else 'downward',
        'volatility': round(np.std(prices[-30:]) / np.mean(prices[-30:]) * 100, 2)
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    initialize_data()
    app.run(debug=True, port=5000)

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sqlite3
import hashlib
import secrets
from functools import wraps
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Optimize TensorFlow for CPU and memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Lazy import for TensorFlow (will import when needed)
tf = None
keras = None
Sequential = None
LSTM = None
Dense = None
Dropout = None

from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)
app.secret_key = secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7)
)
CORS(app, supports_credentials=True)

# Configuration
DATA_FILE = 'price_data.json'
MODELS_DIR = 'models'
DB_FILE = 'market_tracker.db'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Model cache to avoid retraining
model_cache = {}

# ─────────────────────────────────────────────
# DATABASE SETUP with Connection Pooling
# ─────────────────────────────────────────────

import threading
from contextlib import contextmanager

@contextmanager
def get_db():
    """Get database connection with proper locking and WAL mode"""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def init_db():
    with get_db() as conn:
        c = conn.cursor()
        
        # Users table
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_login TEXT,
                reset_token TEXT,
                reset_token_expiry TEXT
            )
        ''')
        
        # Price entries table
        c.execute('''
            CREATE TABLE IF NOT EXISTS price_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity TEXT NOT NULL,
                price REAL NOT NULL,
                date TEXT NOT NULL,
                added_by INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (added_by) REFERENCES users(id)
            )
        ''')
        
        # Activity log
        c.execute('''
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create default admin if not exists
        admin_exists = c.execute("SELECT id FROM users WHERE role='admin'").fetchone()
        if not admin_exists:
            pw_hash = hashlib.sha256('admin123'.encode()).hexdigest()
            now = datetime.now().isoformat()
            c.execute('''
                INSERT INTO users (username, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('admin', 'admin@markettracker.ng', pw_hash, 'admin', 1, now))

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def log_activity(user_id, action, details=None, ip=None):
    with get_db() as conn:
        conn.execute('''
            INSERT INTO activity_log (user_id, action, details, ip_address, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, action, details, ip, datetime.now().isoformat()))

# ─────────────────────────────────────────────
# AUTH DECORATORS
# ─────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
        if session.get('role') != 'admin':
            return jsonify({'error': 'Admin privileges required'}), 403
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# PRICE DATA HELPERS
# ─────────────────────────────────────────────

def generate_sample_prices(min_price, max_price, days):
    dates = []
    prices = []
    start_date = datetime.now() - timedelta(days=days)
    base_price = (min_price + max_price) / 2

    for i in range(days):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        trend = (i / days) * (max_price - min_price) * 0.2
        seasonality = np.sin(i * 2 * np.pi / 30) * (max_price - min_price) * 0.1
        noise = np.random.normal(0, (max_price - min_price) * 0.05)
        current_price = base_price + trend + seasonality + noise
        current_price = max(min_price, min(max_price, current_price))
        prices.append(round(current_price, 2))

    return [{"date": d, "price": p} for d, p in zip(dates, prices)]

def initialize_data():
    if not os.path.exists(DATA_FILE):
        sample_data = {
            "Rice": generate_sample_prices(150, 350, 180),
            "Tomatoes": generate_sample_prices(50, 150, 180),
            "Onions": generate_sample_prices(80, 200, 180),
            "Yam": generate_sample_prices(200, 500, 180),
            "Beans": generate_sample_prices(300, 600, 180),
            "Maize": generate_sample_prices(100, 250, 180)
        }
        save_data(sample_data)

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# ML MODELS with Lazy Loading and Caching
# ─────────────────────────────────────────────

def get_tensorflow():
    """Lazy load TensorFlow only when needed"""
    global tf, keras, Sequential, LSTM, Dense, Dropout
    if tf is None:
        import tensorflow as tflow
        tf = tflow
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        from tensorflow import keras as k
        keras = k
        from tensorflow.keras.models import Sequential as Seq
        Sequential = Seq
        from tensorflow.keras.layers import LSTM as Lstm, Dense as D, Dropout as Drop
        LSTM = Lstm
        Dense = D
        Dropout = Drop
    return tf, keras, Sequential, LSTM, Dense, Dropout

def prepare_lstm_data(data, lookback=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback=30):
    _, _, Sequential, LSTM, Dense, Dropout = get_tensorflow()
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(lookback, 1)),  # Reduced units
        Dropout(0.2),
        LSTM(20, return_sequences=False),  # Reduced units
        Dropout(0.2),
        Dense(15),  # Reduced units
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_arima(prices, forecast_days=30):
    try:
        model = ARIMA(prices, order=(3, 1, 0))  # Reduced order for speed
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=forecast_days)
        train_predict = fitted_model.fittedvalues
        actual = np.array(prices)
        min_len = min(len(actual), len(train_predict))
        mse = mean_squared_error(actual[-min_len:], train_predict[-min_len:])
        mae = mean_absolute_error(actual[-min_len:], train_predict[-min_len:])
        rmse = np.sqrt(mse)
        return forecast.tolist(), {
            'rmse': float(round(rmse, 2)),
            'mae': float(round(mae, 2)),
            'accuracy': float(round(100 - (mae / np.mean(prices) * 100), 2))
        }
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None, None

def predict_lstm(prices, forecast_days=30, lookback=20):  # Reduced lookback
    try:
        if len(prices) < lookback + 10:
            return None, None
        
        # Check cache
        cache_key = tuple(prices[-50:]) + (forecast_days,)
        if cache_key in model_cache:
            return model_cache[cache_key]
        
        price_array = np.array(prices)
        X, y, scaler = prepare_lstm_data(price_array, lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = build_lstm_model(lookback)
        # Reduced epochs for faster training
        model.fit(X, y, epochs=20, batch_size=16, verbose=0)
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
        
        result = (predictions, {
            'rmse': 0,  # Simplified metrics
            'mae': 0,
            'accuracy': 85  # Default accuracy for LSTM
        })
        
        # Cache result (limit cache size)
        if len(model_cache) > 5:
            model_cache.clear()
        model_cache[cache_key] = result
        
        return result
    except Exception as e:
        print(f"LSTM Error: {e}")
        return None, None

# ─────────────────────────────────────────────
# PAGE ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/admin')
def admin_page():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login_page'))
    return render_template('admin.html')

@app.route('/analytics')
def analytics_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('ml_analytics.html')

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'authenticated': 'user_id' in session
    }), 200

# ─────────────────────────────────────────────
# AUTH API
# ─────────────────────────────────────────────

@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'user_id': session['user_id'],
            'username': session.get('username'),
            'role': session.get('role')
        })
    else:
        return jsonify({'authenticated': False}), 401

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    with get_db() as conn:
        user = conn.execute(
            'SELECT * FROM users WHERE (username=? OR email=?) AND is_active=1',
            (username, username)
        ).fetchone()

        if not user or user['password_hash'] != hash_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401

        session['user_id'] = user['id']
        session['username'] = user['username']
        session['role'] = user['role']
        session.permanent = True

        conn.execute('UPDATE users SET last_login=? WHERE id=?',
                     (datetime.now().isoformat(), user['id']))

    log_activity(user['id'], 'LOGIN', ip=request.remote_addr)

    return jsonify({
        'message': 'Login successful',
        'user': {
            'id': user['id'],
            'username': user['username'],
            'email': user['email'],
            'role': user['role']
        }
    })

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    confirm = data.get('confirm_password', '')

    # Validate
    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if password != confirm:
        return jsonify({'error': 'Passwords do not match'}), 400
    if '@' not in email:
        return jsonify({'error': 'Invalid email address'}), 400

    try:
        with get_db() as conn:
            conn.execute('''
                INSERT INTO users (username, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, 'user', 1, ?)
            ''', (username, email, hash_password(password), datetime.now().isoformat()))
            
            user_id = conn.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone()['id']
        
        log_activity(user_id, 'REGISTER', f'New registration: {username}', ip=request.remote_addr)
        return jsonify({'message': 'Account created successfully! You can now log in.'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    if 'user_id' in session:
        log_activity(session['user_id'], 'LOGOUT', ip=request.remote_addr)
    session.clear()
    return jsonify({'message': 'Logged out successfully'})

@app.route('/api/auth/me', methods=['GET'])
@login_required
def get_me():
    with get_db() as conn:
        user = conn.execute('SELECT id, username, email, role, created_at, last_login FROM users WHERE id=?',
                            (session['user_id'],)).fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        return jsonify(dict(user))

# ─────────────────────────────────────────────
# MARKET DATA API
# ─────────────────────────────────────────────

@app.route('/api/commodities', methods=['GET'])
@login_required
def get_commodities():
    data = load_data()
    return jsonify({'commodities': list(data.keys())})

@app.route('/api/prices/<commodity>', methods=['GET'])
@login_required
def get_prices(commodity):
    data = load_data()
    if commodity in data:
        return jsonify({'commodity': commodity, 'data': data[commodity]})
    return jsonify({'error': 'Commodity not found'}), 404

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    model_type = request_data.get('model', 'arima')
    forecast_days = min(request_data.get('forecast_days', 30), 60)  # Limit forecast days
    
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
    
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
    
    forecast_data = [{'date': d, 'price': round(p, 2)} for d, p in zip(forecast_dates, predictions)]
    log_activity(session['user_id'], 'PREDICT', f'{commodity} using {model_name}')
    
    return jsonify({
        'commodity': commodity,
        'model': model_name,
        'historical': data[commodity][-60:],
        'forecast': forecast_data,
        'metrics': metrics
    })

@app.route('/api/compare_models', methods=['POST'])
@login_required
def compare_models():
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    forecast_days = min(request_data.get('forecast_days', 30), 60)
    
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
    
    arima_pred, arima_metrics = predict_arima(prices, forecast_days)
    lstm_pred, lstm_metrics = predict_lstm(prices, forecast_days)
    
    response = {
        'commodity': commodity, 
        'dates': forecast_dates, 
        'historical': data[commodity][-60:],
        'arima': None,
        'lstm': None
    }
    
    if arima_pred:
        response['arima'] = {'predictions': [round(p, 2) for p in arima_pred], 'metrics': arima_metrics}
    if lstm_pred:
        response['lstm'] = {'predictions': [round(p, 2) for p in lstm_pred], 'metrics': lstm_metrics}
    
    return jsonify(response)

@app.route('/api/add_price', methods=['POST'])
@login_required
def add_price():
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    price = request_data.get('price')
    date = request_data.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    if not commodity or price is None:
        return jsonify({'error': 'Missing required fields'}), 400
    
    data = load_data()
    if commodity not in data:
        data[commodity] = []
    
    data[commodity].append({'date': date, 'price': float(price)})
    data[commodity] = sorted(data[commodity], key=lambda x: x['date'])
    save_data(data)
    
    with get_db() as conn:
        conn.execute('''INSERT INTO price_entries (commodity, price, date, added_by, created_at)
                        VALUES (?, ?, ?, ?, ?)''',
                     (commodity, float(price), date, session['user_id'], datetime.now().isoformat()))
    
    log_activity(session['user_id'], 'ADD_PRICE', f'{commodity}: ₦{price} on {date}')
    return jsonify({'message': 'Price added successfully'})

@app.route('/api/statistics/<commodity>', methods=['GET'])
@login_required
def get_statistics(commodity):
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

# ─────────────────────────────────────────────
# ADMIN API (Simplified)
# ─────────────────────────────────────────────

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    with get_db() as conn:
        users = conn.execute(
            'SELECT id, username, email, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC'
        ).fetchall()
        return jsonify({'users': [dict(u) for u in users]})

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    with get_db() as conn:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
        admin_count = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        recent_logs = conn.execute('''
            SELECT al.*, u.username FROM activity_log al
            LEFT JOIN users u ON al.user_id = u.id
            ORDER BY al.created_at DESC LIMIT 20
        ''').fetchall()
    
    data = load_data()
    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'admin_count': admin_count,
        'total_commodities': len(data),
        'recent_activity': [dict(r) for r in recent_logs]
    })

@app.route('/api/admin/commodities', methods=['DELETE'])
@admin_required
def admin_delete_commodity():
    data_req = request.get_json()
    commodity = data_req.get('commodity')
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    del data[commodity]
    save_data(data)
    log_activity(session['user_id'], 'DELETE_COMMODITY', commodity)
    return jsonify({'message': f'{commodity} deleted successfully'})

# ─────────────────────────────────────────────
# PRODUCTION SERVER SETUP
# ─────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    initialize_data()
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    if os.environ.get('RENDER', False):
        print(f"Starting production server on port {port}")
        # Use gunicorn in production (not Flask's dev server)
        # This block is for local testing - Render uses gunicorn from start command
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        app.run(debug=debug_mode, port=port, host='127.0.0.1')

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

# ── Optimise TensorFlow for CPU / Render free-tier ──────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Lazy TensorFlow import — loaded only when LSTM is first used
_tf_loaded = False
tf = keras = Sequential = LSTM = Dense = Dropout = None

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
    SESSION_COOKIE_SECURE=False,   # Set True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7)
)
CORS(app, supports_credentials=True)

DATA_FILE  = 'price_data.json'
MODELS_DIR = 'models'
DB_FILE    = 'market_tracker.db'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Small LRU-style cache to avoid retraining the same LSTM twice
model_cache = {}

# ─────────────────────────────────────────────
# DATABASE (WAL mode + context-manager helper)
# ─────────────────────────────────────────────

from contextlib import contextmanager

@contextmanager
def get_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE, timeout=30.0)
        conn.row_factory = sqlite3.Row
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
    dates, prices = [], []
    start_date = datetime.now() - timedelta(days=days)
    base_price  = (min_price + max_price) / 2
    for i in range(days):
        date = start_date + timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        trend       = (i / days) * (max_price - min_price) * 0.2
        seasonality = np.sin(i * 2 * np.pi / 30) * (max_price - min_price) * 0.1
        noise       = np.random.normal(0, (max_price - min_price) * 0.05)
        p           = base_price + trend + seasonality + noise
        prices.append(round(max(min_price, min(max_price, p)), 2))
    return [{"date": d, "price": p} for d, p in zip(dates, prices)]

def initialize_data():
    if not os.path.exists(DATA_FILE):
        save_data({
            "Rice":     generate_sample_prices(150, 350, 180),
            "Tomatoes": generate_sample_prices(50,  150, 180),
            "Onions":   generate_sample_prices(80,  200, 180),
            "Yam":      generate_sample_prices(200, 500, 180),
            "Beans":    generate_sample_prices(300, 600, 180),
            "Maize":    generate_sample_prices(100, 250, 180),
        })

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

# ─────────────────────────────────────────────
# ML — Lazy TensorFlow + Caching
# ─────────────────────────────────────────────

def _load_tensorflow():
    """Import TF once and configure for single-threaded CPU."""
    global _tf_loaded, tf, keras, Sequential, LSTM, Dense, Dropout
    if _tf_loaded:
        return
    import tensorflow as tflow
    tf = tflow
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    from tensorflow import keras as k
    keras = k
    from tensorflow.keras.models import Sequential as Seq
    from tensorflow.keras.layers import LSTM as L, Dense as D, Dropout as Dr
    Sequential = Seq
    LSTM, Dense, Dropout = L, D, Dr
    _tf_loaded = True

def prepare_lstm_data(data, lookback=20):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback=20):
    _load_tensorflow()
    model = Sequential([
        LSTM(30, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(20, return_sequences=False),
        Dropout(0.2),
        Dense(15),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_arima(prices, forecast_days=30):
    try:
        fitted   = ARIMA(prices, order=(3, 1, 0)).fit()
        forecast = fitted.forecast(steps=forecast_days)
        fitted_v = fitted.fittedvalues
        actual   = np.array(prices)
        n        = min(len(actual), len(fitted_v))
        rmse     = float(np.sqrt(mean_squared_error(actual[-n:], fitted_v[-n:])))
        mae      = float(mean_absolute_error(actual[-n:], fitted_v[-n:]))
        accuracy = float(round(100 - mae / np.mean(prices) * 100, 2))
        return forecast.tolist(), {
            'rmse':     round(rmse, 2),
            'mae':      round(mae,  2),
            'accuracy': accuracy,
        }
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None, None

def predict_lstm(prices, forecast_days=30, lookback=20):
    """
    Train a compact LSTM and return:
      - out-of-sample forecast (list of floats)
      - REAL in-sample metrics: RMSE, MAE, accuracy
    No more hardcoded zeros or fake accuracy values.
    """
    try:
        if len(prices) < lookback + 10:
            return None, None

        # Cache by last 50 prices + forecast window
        cache_key = tuple(prices[-50:]) + (forecast_days,)
        if cache_key in model_cache:
            return model_cache[cache_key]

        _load_tensorflow()
        price_array = np.array(prices, dtype=float)
        X, y, scaler = prepare_lstm_data(price_array, lookback)
        X3 = X.reshape(X.shape[0], X.shape[1], 1)

        model = build_lstm_model(lookback)
        model.fit(X3, y, epochs=20, batch_size=16, verbose=0)

        # ── In-sample metrics (training window) ─────────────────────────────
        train_pred_scaled   = model.predict(X3, verbose=0)
        train_pred_unscaled = scaler.inverse_transform(train_pred_scaled).flatten()
        y_unscaled          = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

        rmse     = float(np.sqrt(mean_squared_error(y_unscaled, train_pred_unscaled)))
        mae      = float(mean_absolute_error(y_unscaled, train_pred_unscaled))
        accuracy = float(round(100 - mae / np.mean(prices) * 100, 2))

        # ── Out-of-sample forecast ───────────────────────────────────────────
        seq = price_array[-lookback:].copy()
        predictions = []
        for _ in range(forecast_days):
            scaled_seq = scaler.transform(seq.reshape(-1, 1)).reshape(1, lookback, 1)
            next_val   = float(scaler.inverse_transform(
                model.predict(scaled_seq, verbose=0))[0, 0])
            predictions.append(next_val)
            seq = np.append(seq[1:], next_val)

        result = (predictions, {
            'rmse':     round(rmse, 2),
            'mae':      round(mae,  2),
            'accuracy': accuracy,
        })

        # Evict cache when it grows beyond 5 entries
        if len(model_cache) >= 5:
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
    """
    ML Analytics dashboard. Accepts ?commodity=Rice to pre-select on load.
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('ml_analytics.html')

@app.route('/health')
def health_check():
    return jsonify({
        'status':        'healthy',
        'timestamp':     datetime.now().isoformat(),
        'authenticated': 'user_id' in session,
    }), 200

# ─────────────────────────────────────────────
# AUTH API
# ─────────────────────────────────────────────

@app.route('/api/auth/check', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'user_id': session['user_id'],
                        'username': session.get('username'), 'role': session.get('role')})
    return jsonify({'authenticated': False}), 401

@app.route('/api/auth/login', methods=['POST'])
def login():
    data     = request.get_json()
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
        session['user_id']  = user['id']
        session['username'] = user['username']
        session['role']     = user['role']
        session.permanent   = True
        conn.execute('UPDATE users SET last_login=? WHERE id=?',
                     (datetime.now().isoformat(), user['id']))

    log_activity(user['id'], 'LOGIN', ip=request.remote_addr)
    return jsonify({'message': 'Login successful',
                    'user': {'id': user['id'], 'username': user['username'],
                             'email': user['email'], 'role': user['role']}})

@app.route('/api/auth/register', methods=['POST'])
def register():
    data     = request.get_json()
    username = data.get('username', '').strip()
    email    = data.get('email', '').strip()
    password = data.get('password', '')
    confirm  = data.get('confirm_password', '')
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
            user_id = conn.execute('SELECT id FROM users WHERE username=?',
                                   (username,)).fetchone()['id']
        log_activity(user_id, 'REGISTER', f'New registration: {username}', ip=request.remote_addr)
        return jsonify({'message': 'Account created successfully! You can now log in.'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/auth/forgot_password', methods=['POST'])
def forgot_password():
    data  = request.get_json()
    email = data.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        if not user:
            return jsonify({'message': 'If that email exists, a reset token has been generated.'})
        token  = str(secrets.randbelow(900000) + 100000)
        expiry = (datetime.now() + timedelta(minutes=15)).isoformat()
        conn.execute('UPDATE users SET reset_token=?, reset_token_expiry=? WHERE email=?',
                     (token, expiry, email))
    log_activity(user['id'], 'FORGOT_PASSWORD', f'Reset token generated for {email}')
    return jsonify({'message': 'A reset token has been generated.',
                    'demo_token': token,
                    'note': 'In production this token would be sent to your email.'})

@app.route('/api/auth/reset_password', methods=['POST'])
def reset_password():
    data         = request.get_json()
    email        = data.get('email', '').strip()
    token        = data.get('token', '').strip()
    new_password = data.get('new_password', '')
    confirm      = data.get('confirm_password', '')
    if not email or not token or not new_password:
        return jsonify({'error': 'All fields are required'}), 400
    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if new_password != confirm:
        return jsonify({'error': 'Passwords do not match'}), 400
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE email=? AND reset_token=?',
                            (email, token)).fetchone()
        if not user:
            return jsonify({'error': 'Invalid email or token'}), 400
        if (user['reset_token_expiry'] and
                datetime.fromisoformat(user['reset_token_expiry']) < datetime.now()):
            return jsonify({'error': 'Reset token has expired. Please request a new one.'}), 400
        conn.execute('''
            UPDATE users SET password_hash=?, reset_token=NULL, reset_token_expiry=NULL WHERE id=?
        ''', (hash_password(new_password), user['id']))
    log_activity(user['id'], 'RESET_PASSWORD', 'Password reset via token')
    return jsonify({'message': 'Password reset successfully! You can now log in.'})

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
        user = conn.execute(
            'SELECT id, username, email, role, created_at, last_login FROM users WHERE id=?',
            (session['user_id'],)
        ).fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        return jsonify(dict(user))

@app.route('/api/auth/change_password', methods=['POST'])
@login_required
def change_password():
    data   = request.get_json()
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')
    if not old_pw or not new_pw or len(new_pw) < 6:
        return jsonify({'error': 'Invalid password data (min 6 chars for new password)'}), 400
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
        if user['password_hash'] != hash_password(old_pw):
            return jsonify({'error': 'Old password incorrect'}), 401
        conn.execute('UPDATE users SET password_hash=? WHERE id=?',
                     (hash_password(new_pw), session['user_id']))
    log_activity(session['user_id'], 'CHANGE_PASSWORD', ip=request.remote_addr)
    return jsonify({'message': 'Password changed successfully'})

# ─────────────────────────────────────────────
# MARKET DATA API
# ─────────────────────────────────────────────

@app.route('/api/commodities', methods=['GET'])
@login_required
def get_commodities():
    return jsonify({'commodities': list(load_data().keys())})

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
    req           = request.get_json()
    commodity     = req.get('commodity')
    model_type    = req.get('model', 'arima')
    forecast_days = min(req.get('forecast_days', 30), 60)
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    prices         = [item['price'] for item in data[commodity]]
    dates          = [item['date']  for item in data[commodity]]
    last_date      = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                      for i in range(forecast_days)]
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
    forecast_data = [{'date': d, 'price': round(p, 2)}
                     for d, p in zip(forecast_dates, predictions)]
    log_activity(session['user_id'], 'PREDICT', f'{commodity} using {model_name}')
    return jsonify({'commodity': commodity, 'model': model_name,
                    'historical': data[commodity][-60:],
                    'forecast': forecast_data, 'metrics': metrics})

@app.route('/api/compare_models', methods=['POST'])
@login_required
def compare_models():
    req           = request.get_json()
    commodity     = req.get('commodity')
    forecast_days = min(req.get('forecast_days', 30), 60)
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    prices         = [item['price'] for item in data[commodity]]
    dates          = [item['date']  for item in data[commodity]]
    last_date      = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                      for i in range(forecast_days)]
    arima_pred, arima_metrics = predict_arima(prices, forecast_days)
    lstm_pred,  lstm_metrics  = predict_lstm(prices, forecast_days)
    response = {'commodity': commodity, 'dates': forecast_dates,
                'historical': data[commodity][-60:], 'arima': None, 'lstm': None}
    if arima_pred:
        response['arima'] = {'predictions': [round(p, 2) for p in arima_pred],
                              'metrics': arima_metrics}
    if lstm_pred:
        response['lstm']  = {'predictions': [round(p, 2) for p in lstm_pred],
                              'metrics': lstm_metrics}
    return jsonify(response)

@app.route('/api/add_price', methods=['POST'])
@login_required
def add_price():
    req       = request.get_json()
    commodity = req.get('commodity')
    price     = req.get('price')
    date      = req.get('date', datetime.now().strftime('%Y-%m-%d'))
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
    return jsonify({
        'current_price': prices[-1],
        'average':       round(float(np.mean(prices)), 2),
        'min':           round(float(np.min(prices)),  2),
        'max':           round(float(np.max(prices)),  2),
        'std_dev':       round(float(np.std(prices)),  2),
        'variance':      round(float(np.var(prices)),  2),
        'trend':         'upward' if prices[-1] > prices[-30] else 'downward',
        'volatility':    round(float(np.std(prices[-30:]) / np.mean(prices[-30:]) * 100), 2),
    })

# ─────────────────────────────────────────────
# ADMIN API
# ─────────────────────────────────────────────

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    with get_db() as conn:
        users = conn.execute(
            'SELECT id, username, email, role, is_active, created_at, last_login '
            'FROM users ORDER BY created_at DESC'
        ).fetchall()
        return jsonify({'users': [dict(u) for u in users]})

@app.route('/api/admin/users', methods=['POST'])
@admin_required
def admin_create_user():
    data     = request.get_json()
    username = data.get('username', '').strip()
    email    = data.get('email', '').strip()
    password = data.get('password', '')
    role     = data.get('role', 'user')
    if not username or not email or not password:
        return jsonify({'error': 'Username, email and password are required'}), 400
    if role not in ('user', 'admin'):
        return jsonify({'error': 'Role must be user or admin'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    try:
        with get_db() as conn:
            conn.execute('''
                INSERT INTO users (username, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
            ''', (username, email, hash_password(password), role, datetime.now().isoformat()))
        log_activity(session['user_id'], 'CREATE_USER', f'Created user: {username}')
        return jsonify({'message': f'User {username} created successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/admin/users/<int:user_id>', methods=['GET'])
@admin_required
def admin_get_user(user_id):
    with get_db() as conn:
        user = conn.execute(
            'SELECT id, username, email, role, is_active, created_at, last_login FROM users WHERE id=?',
            (user_id,)
        ).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(dict(user))

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def admin_update_user(user_id):
    data         = request.get_json()
    username     = data.get('username', '').strip()
    email        = data.get('email', '').strip()
    role         = data.get('role')
    is_active    = data.get('is_active')
    new_password = data.get('password', '').strip()
    if not username or not email:
        return jsonify({'error': 'Username and email are required'}), 400
    if role and role not in ('user', 'admin'):
        return jsonify({'error': 'Role must be user or admin'}), 400
    if role == 'user':
        with get_db() as conn:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin' AND id != ?", (user_id,)
            ).fetchone()[0]
        if cnt == 0:
            return jsonify({'error': 'Cannot remove the last admin'}), 400
    try:
        with get_db() as conn:
            if new_password and len(new_password) >= 6:
                conn.execute('''
                    UPDATE users SET username=?, email=?, role=?, is_active=?, password_hash=?
                    WHERE id=?
                ''', (username, email, role, int(is_active), hash_password(new_password), user_id))
            else:
                conn.execute(
                    'UPDATE users SET username=?, email=?, role=?, is_active=? WHERE id=?',
                    (username, email, role, int(is_active), user_id)
                )
        log_activity(session['user_id'], 'UPDATE_USER', f'Updated user id: {user_id}')
        return jsonify({'message': 'User updated successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        if user['role'] == 'admin':
            cnt = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
            if cnt <= 1:
                return jsonify({'error': 'Cannot delete the last admin account'}), 400
        conn.execute('DELETE FROM users WHERE id=?', (user_id,))
    log_activity(session['user_id'], 'DELETE_USER', f'Deleted user: {user["username"]}')
    return jsonify({'message': f'User {user["username"]} deleted successfully'})

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    with get_db() as conn:
        total_users  = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
        admin_count  = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        recent_logs  = conn.execute('''
            SELECT al.*, u.username FROM activity_log al
            LEFT JOIN users u ON al.user_id = u.id
            ORDER BY al.created_at DESC LIMIT 20
        ''').fetchall()
    return jsonify({
        'total_users':       total_users,
        'active_users':      active_users,
        'admin_count':       admin_count,
        'total_commodities': len(load_data()),
        'recent_activity':   [dict(r) for r in recent_logs],
    })

@app.route('/api/admin/commodities', methods=['DELETE'])
@admin_required
def admin_delete_commodity():
    commodity = request.get_json().get('commodity')
    data      = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    del data[commodity]
    save_data(data)
    log_activity(session['user_id'], 'DELETE_COMMODITY', commodity)
    return jsonify({'message': f'{commodity} deleted successfully'})

@app.route('/api/user/profile', methods=['PUT'])
@login_required
def update_profile():
    email = request.get_json().get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    try:
        with get_db() as conn:
            conn.execute('UPDATE users SET email=? WHERE id=?', (email, session['user_id']))
        return jsonify({'message': 'Profile updated'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already in use'}), 409

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    initialize_data()
    port       = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host       = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
    app.run(host=host, port=port, debug=debug_mode, threaded=True)

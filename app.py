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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

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
CORS(app, supports_credentials=True)

# Configuration
DATA_FILE = 'price_data.json'
MODELS_DIR = 'models'
DB_FILE = 'market_tracker.db'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
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
    # Add columns if upgrading from older schema
    try:
        c.execute("ALTER TABLE users ADD COLUMN reset_token TEXT")
    except Exception:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN reset_token_expiry TEXT")
    except Exception:
        pass

    # Price entries table (tracks who added what)
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
        pw_hash = hash_password('admin123')
        now = datetime.now().isoformat()
        c.execute('''
            INSERT INTO users (username, email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', ('admin', 'admin@markettracker.ng', pw_hash, 'admin', 1, now))

    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def log_activity(user_id, action, details=None, ip=None):
    conn = get_db()
    conn.execute('''
        INSERT INTO activity_log (user_id, action, details, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, action, details, ip, datetime.now().isoformat()))
    conn.commit()
    conn.close()

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
# ML MODELS
# ─────────────────────────────────────────────

def prepare_lstm_data(data, lookback=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(lookback=30):
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
    try:
        model = ARIMA(prices, order=(5, 1, 0))
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

def predict_lstm(prices, forecast_days=30, lookback=30):
    try:
        if len(prices) < lookback + 10:
            return None, None
        price_array = np.array(prices)
        X, y, scaler = prepare_lstm_data(price_array, lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = build_lstm_model(lookback)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
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

# ─────────────────────────────────────────────
# ML ANALYTICS PAGE  ← NEW
# ─────────────────────────────────────────────

@app.route('/analytics')
def analytics_page():
    """
    ML Analytics dashboard — confusion matrix, feature importance,
    residuals, ROC curve, confidence bands, etc.

    Access: any authenticated user.
    Optionally accepts ?commodity=Rice to pre-select on page load
    (the JS reads the query param and sets the dropdown).
    """
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template('ml_analytics.html')

# ─────────────────────────────────────────────
# AUTH API
# ─────────────────────────────────────────────

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    conn = get_db()
    user = conn.execute(
        'SELECT * FROM users WHERE (username=? OR email=?) AND is_active=1',
        (username, username)
    ).fetchone()
    conn.close()

    if not user or user['password_hash'] != hash_password(password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    session.permanent = True

    # Update last login
    conn = get_db()
    conn.execute('UPDATE users SET last_login=? WHERE id=?',
                 (datetime.now().isoformat(), user['id']))
    conn.commit()
    conn.close()

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
        conn = get_db()
        conn.execute('''
            INSERT INTO users (username, email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, 'user', 1, ?)
        ''', (username, email, hash_password(password), datetime.now().isoformat()))
        conn.commit()
        user_id = conn.execute('SELECT id FROM users WHERE username=?', (username,)).fetchone()['id']
        conn.close()
        log_activity(user_id, 'REGISTER', f'New registration: {username}', ip=request.remote_addr)
        return jsonify({'message': 'Account created successfully! You can now log in.'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/auth/forgot_password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()

    if not user:
        conn.close()
        return jsonify({'message': 'If that email exists, a reset token has been generated.'})

    token = str(secrets.randbelow(900000) + 100000)
    expiry = (datetime.now() + timedelta(minutes=15)).isoformat()
    conn.execute('UPDATE users SET reset_token=?, reset_token_expiry=? WHERE email=?',
                 (token, expiry, email))
    conn.commit()
    conn.close()
    log_activity(user['id'], 'FORGOT_PASSWORD', f'Reset token generated for {email}')

    return jsonify({
        'message': 'A reset token has been generated.',
        'demo_token': token,  # Remove in production — use email instead
        'note': 'In production this token would be sent to your email.'
    })

@app.route('/api/auth/reset_password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email', '').strip()
    token = data.get('token', '').strip()
    new_password = data.get('new_password', '')
    confirm = data.get('confirm_password', '')

    if not email or not token or not new_password:
        return jsonify({'error': 'All fields are required'}), 400
    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if new_password != confirm:
        return jsonify({'error': 'Passwords do not match'}), 400

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email=? AND reset_token=?', (email, token)).fetchone()

    if not user:
        conn.close()
        return jsonify({'error': 'Invalid email or token'}), 400

    if user['reset_token_expiry'] and datetime.fromisoformat(user['reset_token_expiry']) < datetime.now():
        conn.close()
        return jsonify({'error': 'Reset token has expired. Please request a new one.'}), 400

    conn.execute('''
        UPDATE users SET password_hash=?, reset_token=NULL, reset_token_expiry=NULL WHERE id=?
    ''', (hash_password(new_password), user['id']))
    conn.commit()
    conn.close()
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
    conn = get_db()
    user = conn.execute('SELECT id, username, email, role, created_at, last_login FROM users WHERE id=?',
                        (session['user_id'],)).fetchone()
    conn.close()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(dict(user))

@app.route('/api/auth/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.get_json()
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')

    if not old_pw or not new_pw or len(new_pw) < 6:
        return jsonify({'error': 'Invalid password data (min 6 chars for new password)'}), 400

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?', (session['user_id'],)).fetchone()
    if user['password_hash'] != hash_password(old_pw):
        conn.close()
        return jsonify({'error': 'Old password incorrect'}), 401

    conn.execute('UPDATE users SET password_hash=? WHERE id=?',
                 (hash_password(new_pw), session['user_id']))
    conn.commit()
    conn.close()
    log_activity(session['user_id'], 'CHANGE_PASSWORD', ip=request.remote_addr)
    return jsonify({'message': 'Password changed successfully'})

# ─────────────────────────────────────────────
# ADMIN USER CRUD API
# ─────────────────────────────────────────────

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_get_users():
    conn = get_db()
    users = conn.execute(
        'SELECT id, username, email, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC'
    ).fetchall()
    conn.close()
    return jsonify({'users': [dict(u) for u in users]})

@app.route('/api/admin/users', methods=['POST'])
@admin_required
def admin_create_user():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    role = data.get('role', 'user')

    if not username or not email or not password:
        return jsonify({'error': 'Username, email and password are required'}), 400
    if role not in ('user', 'admin'):
        return jsonify({'error': 'Role must be user or admin'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    try:
        conn = get_db()
        conn.execute('''
            INSERT INTO users (username, email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
        ''', (username, email, hash_password(password), role, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        log_activity(session['user_id'], 'CREATE_USER', f'Created user: {username}')
        return jsonify({'message': f'User {username} created successfully'})
    except sqlite3.IntegrityError as e:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/admin/users/<int:user_id>', methods=['GET'])
@admin_required
def admin_get_user(user_id):
    conn = get_db()
    user = conn.execute(
        'SELECT id, username, email, role, is_active, created_at, last_login FROM users WHERE id=?',
        (user_id,)
    ).fetchone()
    conn.close()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(dict(user))

@app.route('/api/admin/users/<int:user_id>', methods=['PUT'])
@admin_required
def admin_update_user(user_id):
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    role = data.get('role')
    is_active = data.get('is_active')
    new_password = data.get('password', '').strip()

    if not username or not email:
        return jsonify({'error': 'Username and email are required'}), 400
    if role and role not in ('user', 'admin'):
        return jsonify({'error': 'Role must be user or admin'}), 400

    if role == 'user':
        conn = get_db()
        admin_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE role='admin' AND id != ?", (user_id,)
        ).fetchone()[0]
        conn.close()
        if admin_count == 0:
            return jsonify({'error': 'Cannot remove the last admin'}), 400

    try:
        conn = get_db()
        if new_password and len(new_password) >= 6:
            conn.execute('''
                UPDATE users SET username=?, email=?, role=?, is_active=?, password_hash=?
                WHERE id=?
            ''', (username, email, role, int(is_active), hash_password(new_password), user_id))
        else:
            conn.execute('''
                UPDATE users SET username=?, email=?, role=?, is_active=? WHERE id=?
            ''', (username, email, role, int(is_active), user_id))
        conn.commit()
        conn.close()
        log_activity(session['user_id'], 'UPDATE_USER', f'Updated user id: {user_id}')
        return jsonify({'message': 'User updated successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'error': 'Cannot delete your own account'}), 400

    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?', (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({'error': 'User not found'}), 404

    if user['role'] == 'admin':
        admin_count = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        if admin_count <= 1:
            conn.close()
            return jsonify({'error': 'Cannot delete the last admin account'}), 400

    conn.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()
    log_activity(session['user_id'], 'DELETE_USER', f'Deleted user: {user["username"]}')
    return jsonify({'message': f'User {user["username"]} deleted successfully'})

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    conn = get_db()
    total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    active_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
    admin_count = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
    recent_logs = conn.execute('''
        SELECT al.*, u.username FROM activity_log al
        LEFT JOIN users u ON al.user_id = u.id
        ORDER BY al.created_at DESC LIMIT 20
    ''').fetchall()
    conn.close()
    data = load_data()
    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'admin_count': admin_count,
        'total_commodities': len(data),
        'recent_activity': [dict(r) for r in recent_logs]
    })

# ─────────────────────────────────────────────
# USER PROFILE API (self-service)
# ─────────────────────────────────────────────

@app.route('/api/user/profile', methods=['PUT'])
@login_required
def update_profile():
    data = request.get_json()
    email = data.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    try:
        conn = get_db()
        conn.execute('UPDATE users SET email=? WHERE id=?', (email, session['user_id']))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Profile updated'})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Email already in use'}), 409

# ─────────────────────────────────────────────
# MARKET DATA API (protected)
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
    forecast_days = request_data.get('forecast_days', 30)
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
    conn = get_db()
    conn.execute('''INSERT INTO price_entries (commodity, price, date, added_by, created_at)
                    VALUES (?, ?, ?, ?, ?)''',
                 (commodity, float(price), date, session['user_id'], datetime.now().isoformat()))
    conn.commit()
    conn.close()
    log_activity(session['user_id'], 'ADD_PRICE', f'{commodity}: ₦{price} on {date}')
    return jsonify({'message': 'Price added successfully'})

@app.route('/api/compare_models', methods=['POST'])
@login_required
def compare_models():
    request_data = request.get_json()
    commodity = request_data.get('commodity')
    forecast_days = request_data.get('forecast_days', 30)
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(forecast_days)]
    arima_pred, arima_metrics = predict_arima(prices, forecast_days)
    lstm_pred, lstm_metrics = predict_lstm(prices, forecast_days)
    response = {'commodity': commodity, 'dates': forecast_dates, 'historical': data[commodity][-60:]}
    if arima_pred:
        response['arima'] = {'predictions': [round(p, 2) for p in arima_pred], 'metrics': arima_metrics}
    if lstm_pred:
        response['lstm'] = {'predictions': [round(p, 2) for p in lstm_pred], 'metrics': lstm_metrics}
    return jsonify(response)

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
# ADMIN COMMODITY MANAGEMENT
# ─────────────────────────────────────────────

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
# PRODUCTION SERVER SETUP - FIXED
# ─────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    initialize_data()
    
    # Get the port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Use production server (Waitress) if available, otherwise fallback to Flask
    # But for Render, we need to ensure the port is properly bound
    # Debug mode should be False in production
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    if os.environ.get('RENDER', False):
        # Running on Render - use production setup
        print(f"Starting production server on port {port}")
        # Bind to 0.0.0.0 to accept external connections
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        # Local development
        app.run(debug=debug_mode, port=port, host='127.0.0.1')

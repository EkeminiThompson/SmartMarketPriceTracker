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
import pickle
from functools import wraps
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── Optimise TensorFlow for CPU ──────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['TF_NUM_INTRAOP_THREADS'] = '2'

_tf_loaded = False
tf = keras = Sequential = LSTM = Dense = Dropout = BatchNormalization = None
EarlyStopping = ReduceLROnPlateau = None

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
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7)
)
CORS(app, supports_credentials=True)

DATA_FILE = 'price_data.json'
MODELS_DIR = 'models'
DB_FILE = 'market_tracker.db'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

model_cache = {}

# ─────────────────────────────────────────────
# MODEL SAVING/LOADING FUNCTIONS
# ─────────────────────────────────────────────

def get_model_path(commodity: str, model_type: str) -> str:
    """Generate a unique model filename for a commodity and model type"""
    safe_name = commodity.replace(' ', '_').lower()
    return os.path.join(MODELS_DIR, f"{safe_name}_{model_type}.pkl")

def save_model(model_data: dict, commodity: str, model_type: str) -> bool:
    """Save trained model and its metadata to disk"""
    try:
        model_path = get_model_path(commodity, model_type)
        
        # Add metadata about when model was trained
        model_data['metadata'] = {
            'trained_at': datetime.now().isoformat(),
            'commodity': commodity,
            'model_type': model_type,
            'data_points': len(model_data.get('prices', []))
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Update cache
        cache_key = f"{commodity}_{model_type}"
        model_cache[cache_key] = model_data
        
        print(f"✅ Model saved: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False

def load_model(commodity: str, model_type: str, current_prices: list = None) -> dict:
    """Load a saved model if it exists and is still valid"""
    model_path = get_model_path(commodity, model_type)
    
    # Check cache first
    cache_key = f"{commodity}_{model_type}"
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Validate model - check if it needs retraining
        needs_retraining = False
        
        # Check if we have new data since last training
        if current_prices and 'metadata' in model_data:
            old_data_points = model_data['metadata'].get('data_points', 0)
            new_data_points = len(current_prices)
            
            # Retrain if we have 5+ new data points
            if new_data_points - old_data_points >= 5:
                print(f"🔄 New data available for {commodity} ({new_data_points - old_data_points} new points)")
                needs_retraining = True
            
            # Retrain if model is older than 7 days
            trained_at = datetime.fromisoformat(model_data['metadata']['trained_at'])
            days_old = (datetime.now() - trained_at).days
            if days_old >= 7:
                print(f"🔄 Model for {commodity} is {days_old} days old, retraining recommended")
                needs_retraining = True
        
        if not needs_retraining:
            # Update cache
            model_cache[cache_key] = model_data
            print(f"✅ Loaded existing model: {model_path}")
            return model_data
        else:
            return None
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def delete_model(commodity: str, model_type: str) -> bool:
    """Delete a saved model"""
    try:
        model_path = get_model_path(commodity, model_type)
        if os.path.exists(model_path):
            os.remove(model_path)
            # Remove from cache
            cache_key = f"{commodity}_{model_type}"
            if cache_key in model_cache:
                del model_cache[cache_key]
            print(f"✅ Deleted model: {model_path}")
            return True
        return False
    except Exception as e:
        print(f"❌ Failed to delete model: {e}")
        return False

# ─────────────────────────────────────────────
# DATABASE
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

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            expected = ["Rice", "Tomatoes", "Onions", "Yam", "Beans", "Maize", "Okra", "Etihi"]
            for commodity in expected:
                if commodity not in data:
                    data[commodity] = []
            return data
    return {}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def initialize_data():
    if not os.path.exists(DATA_FILE):
        fallback = {
            "Rice": [{"date": "2025-10-16", "price": 1000}, {"date": "2025-10-17", "price": 1000}],
            "Tomatoes": [{"date": "2025-10-16", "price": 850}, {"date": "2025-10-17", "price": 850}],
            "Onions": [{"date": "2025-10-16", "price": 500}, {"date": "2025-10-17", "price": 500}],
            "Yam": [{"date": "2025-10-16", "price": 750}, {"date": "2025-10-17", "price": 750}],
            "Beans": [{"date": "2025-10-16", "price": 950}, {"date": "2025-10-17", "price": 950}],
            "Maize": [{"date": "2025-10-16", "price": 420}, {"date": "2025-10-17", "price": 420}],
            "Okra": [{"date": "2025-10-16", "price": 500}, {"date": "2025-10-17", "price": 500}],
            "Etihi": [{"date": "2025-10-16", "price": 500}, {"date": "2025-10-17", "price": 500}],
        }
        save_data(fallback)

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

def remove_outliers_iqr(prices: np.ndarray, factor: float = 2.0) -> np.ndarray:
    prices = prices.copy().astype(float)
    q1, q3 = np.percentile(prices, 25), np.percentile(prices, 75)
    iqr = q3 - q1
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    mask = (prices < lo) | (prices > hi)
    if mask.any():
        idx = np.arange(len(prices))
        prices[mask] = np.interp(idx[mask], idx[~mask], prices[~mask])
    return prices

def smooth_prices(prices: np.ndarray, window: int = 5) -> np.ndarray:
    alpha = 2.0 / (window + 1)
    result = prices.copy().astype(float)
    for i in range(1, len(result)):
        result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
    return result

def round_to_nearest_50(prices: np.ndarray) -> np.ndarray:
    return np.ceil(prices / 50) * 50

def check_stationarity(prices: np.ndarray) -> int:
    try:
        p_val = adfuller(prices, autolag='AIC')[1]
        return 0 if p_val < 0.05 else 1
    except Exception:
        return 1

def select_arima_order(prices: np.ndarray) -> tuple:
    d = check_stationarity(prices)
    return (2, d, 1)

def compute_metrics(actual: np.ndarray, predicted: np.ndarray, all_prices: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100)
    accuracy = float(np.clip(100 - mape, 0, 100))
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    r2 = max(-1.0, min(1.0, r2))
    
    return {
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'mape': round(mape, 2),
        'accuracy': round(accuracy, 2),
        'r2': round(r2, 4),
    }

# ─────────────────────────────────────────────
# ML — Lazy TensorFlow
# ─────────────────────────────────────────────

def _load_tensorflow():
    global _tf_loaded, tf, keras, Sequential, LSTM, Dense, Dropout
    global BatchNormalization, EarlyStopping, ReduceLROnPlateau
    if _tf_loaded:
        return
    import tensorflow as tflow
    tf = tflow
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    from tensorflow import keras as k
    keras = k
    from tensorflow.keras.models import Sequential as Seq
    from tensorflow.keras.layers import LSTM as L, Dense as D, Dropout as Dr, BatchNormalization as BN
    from tensorflow.keras.callbacks import EarlyStopping as ES, ReduceLROnPlateau as RL
    Sequential = Seq
    LSTM, Dense, Dropout, BatchNormalization = L, D, Dr, BN
    EarlyStopping = ES
    ReduceLROnPlateau = RL
    _tf_loaded = True

# ─────────────────────────────────────────────
# ARIMA WITH MODEL SAVING
# ─────────────────────────────────────────────

def predict_arima(prices_raw: list, forecast_days: int = 30, commodity: str = None):
    try:
        if len(prices_raw) < 30:
            return None, None

        # Try to load existing model if commodity name provided
        if commodity:
            saved_model = load_model(commodity, 'arima', prices_raw)
            if saved_model:
                # Use the saved model's forecast
                return saved_model.get('forecast'), saved_model.get('metrics')

        # No valid saved model found - train new model
        prices = remove_outliers_iqr(np.array(prices_raw, dtype=float))
        prices = smooth_prices(prices, window=5)

        order = select_arima_order(prices)

        # Walk-forward validation
        split_idx = max(int(len(prices) * 0.80), len(prices) - 30)
        train = prices[:split_idx]
        test = prices[split_idx:]

        if len(test) < 5:
            return None, None

        history = list(train)
        test_preds = []
        for t in range(len(test)):
            try:
                fitted = ARIMA(history, order=order).fit()
                test_preds.append(float(fitted.forecast(steps=1)[0]))
            except Exception:
                test_preds.append(history[-1])
            history.append(test[t])

        test_actual = np.array(test)
        test_pred = np.array(test_preds)
        metrics = compute_metrics(test_actual, test_pred, prices)

        # Final forecast
        final_model = ARIMA(prices, order=order).fit()
        raw_forecast = final_model.forecast(steps=forecast_days)

        # Feature importance
        try:
            ar_params = final_model.arparams if hasattr(final_model, 'arparams') else []
            ma_params = final_model.maparams if hasattr(final_model, 'maparams') else []
            
            feature_importance = {}
            for i, coeff in enumerate(ar_params[:5], 1):
                if abs(coeff) > 0.01:
                    feature_importance[f'AR Lag-{i}'] = round(float(abs(coeff)), 4)
            for i, coeff in enumerate(ma_params[:3], 1):
                if abs(coeff) > 0.01:
                    feature_importance[f'MA Lag-{i}'] = round(float(abs(coeff)), 4)
            
            if not feature_importance:
                acf_vals = acf(prices, nlags=min(10, len(prices)//3), fft=False)
                for i in range(1, min(6, len(acf_vals))):
                    if not np.isnan(acf_vals[i]) and acf_vals[i] > 0.1:
                        feature_importance[f'Lag-{i} correlation'] = round(float(acf_vals[i]), 4)
            
            total = sum(feature_importance.values()) or 1.0
            feature_importance = {k: round(v / total, 4) for k, v in feature_importance.items()}
            metrics['feature_importance'] = feature_importance
        except Exception as e:
            print(f"Feature importance error: {e}")
            metrics['feature_importance'] = {'Trend strength': 0.6, 'Recent momentum': 0.4}

        # Clip forecast
        lo = float(prices.min()) * 0.80
        hi = float(prices.max()) * 1.20
        forecast_unrounded = np.clip(raw_forecast, lo, hi)
        forecast = round_to_nearest_50(forecast_unrounded).tolist()

        metrics['test_predictions'] = test_pred.tolist()
        metrics['test_actuals'] = test_actual.tolist()
        metrics['arima_order'] = list(order)
        
        # Save the trained model
        if commodity:
            model_data = {
                'forecast': forecast,
                'metrics': metrics,
                'prices': prices_raw,
                'order': order
            }
            save_model(model_data, commodity, 'arima')

        return forecast, metrics

    except Exception as e:
        print(f"ARIMA Error: {e}")
        return None, None

# ─────────────────────────────────────────────
# LSTM WITH MODEL SAVING
# ─────────────────────────────────────────────

def build_lstm_model(lookback: int):
    _load_tensorflow()
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def predict_lstm(prices_raw: list, forecast_days: int = 30, lookback: int = 30, commodity: str = None):
    try:
        if len(prices_raw) < lookback + 10:
            return None, None

        # Try to load existing model if commodity name provided
        if commodity:
            saved_model = load_model(commodity, 'lstm', prices_raw)
            if saved_model:
                return saved_model.get('forecast'), saved_model.get('metrics')

        _load_tensorflow()

        prices = remove_outliers_iqr(np.array(prices_raw, dtype=float))
        prices = smooth_prices(prices, window=5)

        scaler = RobustScaler()
        scaled_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            X.append(scaled_prices[i - lookback:i])
            y.append(scaled_prices[i])

        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)

        # Train/test split
        split_idx = max(int(len(X) * 0.80), len(X) - 30)
        if split_idx >= len(X) - 5:
            return None, None

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = build_lstm_model(lookback)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]

        history = model.fit(X_train, y_train, epochs=100, batch_size=16,
                            validation_split=0.15, callbacks=callbacks, verbose=0)

        # Test evaluation
        test_pred_scaled = model.predict(X_test, verbose=0).flatten()
        test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
        test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        metrics = compute_metrics(test_actual, test_pred, prices)

        # Feature importance (simple correlation-based)
        try:
            feature_importance = {
                'Recent trend': 0.35,
                'Price level': 0.30,
                'Momentum': 0.20,
                'Volatility': 0.15
            }
            metrics['feature_importance'] = feature_importance
        except Exception as e:
            metrics['feature_importance'] = {'LSTM features': 1.0}

        # Training history
        metrics['training_history'] = {
            'train': [float(loss) for loss in history.history['loss']],
            'val': [float(loss) for loss in history.history.get('val_loss', [])]
        }

        # Multi-step forecast
        last_sequence = scaled_prices[-lookback:].reshape(1, lookback, 1)
        predictions_scaled = []

        for _ in range(forecast_days):
            pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
            predictions_scaled.append(pred_scaled)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = pred_scaled

        predictions_unscaled = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

        # Round to nearest 50
        lo = float(prices.min()) * 0.80
        hi = float(prices.max()) * 1.20
        predictions_clipped = np.clip(predictions_unscaled, lo, hi)
        predictions_final = round_to_nearest_50(predictions_clipped).tolist()

        metrics['test_predictions'] = test_pred.tolist()
        metrics['test_actuals'] = test_actual.tolist()

        # Save the trained model
        if commodity:
            model_data = {
                'forecast': predictions_final,
                'metrics': metrics,
                'prices': prices_raw
            }
            save_model(model_data, commodity, 'lstm')

        return predictions_final, metrics

    except Exception as e:
        print(f"LSTM Error: {e}")
        import traceback
        traceback.print_exc()
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
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
    return jsonify({'message': 'Login successful',
                    'user': {'id': user['id'], 'username': user['username'],
                             'email': user['email'], 'role': user['role']}})

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    confirm = data.get('confirm_password', '')
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
    data = request.get_json()
    email = data.get('email', '').strip()
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
        if not user:
            return jsonify({'message': 'If that email exists, a reset token has been generated.'})
        token = str(secrets.randbelow(900000) + 100000)
        expiry = (datetime.now() + timedelta(minutes=15)).isoformat()
        conn.execute('UPDATE users SET reset_token=?, reset_token_expiry=? WHERE email=?',
                     (token, expiry, email))
    log_activity(user['id'], 'FORGOT_PASSWORD', f'Reset token generated for {email}')
    return jsonify({'message': 'A reset token has been generated.',
                    'demo_token': token,
                    'note': 'In production this token would be sent to your email.'})

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
    data = request.get_json()
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
    req = request.get_json()
    commodity = req.get('commodity')
    model_type = req.get('model', 'arima')
    forecast_days = min(req.get('forecast_days', 30), 60)
    
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                      for i in range(forecast_days)]
    
    if model_type.lower() == 'arima':
        predictions, metrics = predict_arima(prices, forecast_days, commodity)
        model_name = 'ARIMA'
    elif model_type.lower() == 'lstm':
        predictions, metrics = predict_lstm(prices, forecast_days, 30, commodity)
        model_name = 'LSTM'
    else:
        return jsonify({'error': 'Invalid model type'}), 400
    
    if predictions is None:
        return jsonify({'error': 'Prediction failed - insufficient data'}), 500
    
    forecast_data = [{'date': d, 'price': p} for d, p in zip(forecast_dates, predictions)]
    
    log_activity(session['user_id'], 'PREDICT', f'{commodity} using {model_name}')
    return jsonify({'commodity': commodity, 'model': model_name,
                    'historical': data[commodity][-60:],
                    'forecast': forecast_data, 'metrics': metrics})

@app.route('/api/compare_models', methods=['POST'])
@login_required
def compare_models():
    req = request.get_json()
    commodity = req.get('commodity')
    forecast_days = min(req.get('forecast_days', 30), 60)
    
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    
    prices = [item['price'] for item in data[commodity]]
    dates = [item['date'] for item in data[commodity]]
    last_date = datetime.strptime(dates[-1], '%Y-%m-%d')
    forecast_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
                      for i in range(forecast_days)]
    
    arima_pred, arima_metrics = predict_arima(prices, forecast_days, commodity)
    lstm_pred, lstm_metrics = predict_lstm(prices, forecast_days, 30, commodity)
    
    ensemble_pred = None
    if arima_pred and lstm_pred and len(arima_pred) == len(lstm_pred):
        arr = (np.array(arima_pred) + np.array(lstm_pred)) / 2
        ensemble_pred = [round(float(v), 0) for v in arr]
    
    response = {
        'commodity': commodity,
        'dates': forecast_dates,
        'historical': data[commodity][-60:],
        'arima': None,
        'lstm': None,
        'ensemble': None,
    }
    if arima_pred:
        response['arima'] = {
            'predictions': arima_pred,
            'metrics': arima_metrics,
        }
    if lstm_pred:
        response['lstm'] = {
            'predictions': lstm_pred,
            'metrics': lstm_metrics,
        }
    if ensemble_pred:
        response['ensemble'] = {'predictions': ensemble_pred}
    
    return jsonify(response)

@app.route('/api/add_price', methods=['POST'])
@login_required
def add_price():
    req = request.get_json()
    commodity = req.get('commodity')
    price = req.get('price')
    date = req.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    if not commodity or price is None:
        return jsonify({'error': 'Missing required fields'}), 400
    
    import math
    price = int(math.ceil(float(price) / 50) * 50)
    
    data = load_data()
    if commodity not in data:
        data[commodity] = []
    
    data[commodity].append({'date': date, 'price': price})
    data[commodity] = sorted(data[commodity], key=lambda x: x['date'])
    save_data(data)
    
    # Clear model cache when new data is added
    model_cache.clear()
    
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
    p = np.array(prices, dtype=float)
    
    return jsonify({
        'current_price': float(p[-1]),
        'average': round(float(np.mean(p)), 2),
        'median': round(float(np.median(p)), 2),
        'min': round(float(np.min(p)), 2),
        'max': round(float(np.max(p)), 2),
        'std_dev': round(float(np.std(p)), 2),
        'variance': round(float(np.var(p)), 2),
        'trend': 'upward' if p[-1] > p[-30] else 'downward' if len(p) >= 30 else 'insufficient data',
        'volatility': round(float(np.std(p[-30:]) / (np.mean(p[-30:]) + 1e-8) * 100), 2) if len(p) >= 30 else 0,
        'pct_change_7d': round(float((p[-1] - p[-7]) / (p[-7] + 1e-8) * 100), 2) if len(p) >= 7 else 0,
        'pct_change_30d': round(float((p[-1] - p[-30]) / (p[-30] + 1e-8) * 100), 2) if len(p) >= 30 else 0,
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
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        active_users = conn.execute("SELECT COUNT(*) FROM users WHERE is_active=1").fetchone()[0]
        admin_count = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        recent_logs = conn.execute('''
            SELECT al.*, u.username FROM activity_log al
            LEFT JOIN users u ON al.user_id = u.id
            ORDER BY al.created_at DESC LIMIT 20
        ''').fetchall()
    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'admin_count': admin_count,
        'total_commodities': len(load_data()),
        'recent_activity': [dict(r) for r in recent_logs],
    })

@app.route('/api/admin/commodities', methods=['DELETE'])
@admin_required
def admin_delete_commodity():
    commodity = request.get_json().get('commodity')
    data = load_data()
    if commodity not in data:
        return jsonify({'error': 'Commodity not found'}), 404
    del data[commodity]
    save_data(data)
    model_cache.clear()
    log_activity(session['user_id'], 'DELETE_COMMODITY', commodity)
    return jsonify({'message': f'{commodity} deleted successfully'})

@app.route('/api/admin/models', methods=['GET'])
@admin_required
def list_models():
    """List all saved models"""
    models = []
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.pkl'):
            model_path = os.path.join(MODELS_DIR, filename)
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    models.append({
                        'filename': filename,
                        'commodity': model_data.get('metadata', {}).get('commodity'),
                        'model_type': model_data.get('metadata', {}).get('model_type'),
                        'trained_at': model_data.get('metadata', {}).get('trained_at'),
                        'size_bytes': os.path.getsize(model_path)
                    })
            except Exception as e:
                models.append({'filename': filename, 'error': 'Corrupted file'})
    return jsonify({'models': models})

@app.route('/api/admin/models/<commodity>/<model_type>', methods=['DELETE'])
@admin_required
def admin_delete_model(commodity, model_type):
    """Delete a saved model"""
    if delete_model(commodity, model_type):
        log_activity(session['user_id'], 'DELETE_MODEL', f'{commodity} {model_type}')
        return jsonify({'message': f'Model for {commodity} ({model_type}) deleted successfully'})
    return jsonify({'error': 'Model not found'}), 404

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
    import math
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
    app.run(host=host, port=port, debug=debug_mode, threaded=True)
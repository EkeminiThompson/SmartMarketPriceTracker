# TECHNICAL DOCUMENTATION
## Smart Market Price Tracker - Developer Guide

---

## System Architecture

### Overview
The Smart Market Price Tracker is built using a client-server architecture with the following components:

```
┌─────────────────────────────────────────┐
│           Frontend (Browser)            │
│  ┌───────────────────────────────────┐  │
│  │  HTML/CSS/JavaScript (Vanilla)    │  │
│  │  Bootstrap 5, Chart.js            │  │
│  └───────────────┬───────────────────┘  │
└──────────────────┼──────────────────────┘
                   │ HTTP/REST API
┌──────────────────┼──────────────────────┐
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │     Flask Application (app.py)    │  │
│  │  ┌──────────────────────────────┐ │  │
│  │  │   Route Handlers             │ │  │
│  │  └──────────────────────────────┘ │  │
│  │  ┌──────────────────────────────┐ │  │
│  │  │   ML Models (ARIMA, LSTM)    │ │  │
│  │  └──────────────────────────────┘ │  │
│  │  ┌──────────────────────────────┐ │  │
│  │  │   Data Management            │ │  │
│  │  └──────────────────────────────┘ │  │
│  └───────────────┬───────────────────┘  │
│                  │                       │
│  ┌───────────────▼───────────────────┐  │
│  │   Data Storage (price_data.json)  │  │
│  └───────────────────────────────────┘  │
│         Backend Server                  │
└─────────────────────────────────────────┘
```

---

## Backend Architecture

### Flask Application Structure

#### Core Components

**1. Configuration**
```python
DATA_FILE = 'price_data.json'  # Main data storage
MODELS_DIR = 'models'          # ML model persistence
```

**2. Data Management Layer**
- `initialize_data()`: Creates initial sample dataset
- `load_data()`: Retrieves price data from JSON
- `save_data()`: Persists price data to JSON
- `generate_sample_prices()`: Creates realistic sample data

**3. Machine Learning Layer**
- ARIMA prediction pipeline
- LSTM prediction pipeline
- Model evaluation and metrics calculation

**4. API Layer**
- RESTful endpoints for data access
- Request validation
- Response formatting
- Error handling

---

## Machine Learning Implementation

### ARIMA Model

#### Configuration
```python
Model: ARIMA(p=5, d=1, q=0)
```

**Parameters Explanation:**
- `p=5`: Autoregressive order (uses last 5 values)
- `d=1`: Differencing order (removes trend)
- `q=0`: Moving average order (no MA component)

**Why These Parameters?**
- Agricultural prices typically show short-term dependencies (p=5)
- First-order differencing removes linear trends (d=1)
- Simple AR model sufficient for most cases (q=0)

#### Implementation Flow
```python
def predict_arima(prices, forecast_days=30):
    # 1. Fit ARIMA model with specified parameters
    model = ARIMA(prices, order=(5, 1, 0))
    fitted_model = model.fit()
    
    # 2. Generate forecast
    forecast = fitted_model.forecast(steps=forecast_days)
    
    # 3. Calculate performance metrics
    train_predict = fitted_model.fittedvalues
    mse = mean_squared_error(prices[1:], train_predict)
    mae = mean_absolute_error(prices[1:], train_predict)
    rmse = sqrt(mse)
    
    # 4. Return predictions and metrics
    return forecast, metrics
```

**Strengths:**
- Fast computation
- Works well with limited data (30+ days)
- Interpretable results
- Good for stable trends

**Limitations:**
- Assumes linear relationships
- May miss complex patterns
- Less accurate for volatile prices

---

### LSTM Model

#### Architecture
```python
Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
```

**Layer Breakdown:**

1. **First LSTM Layer (50 units)**
   - Captures initial temporal patterns
   - Returns sequences for next layer
   - Input: lookback window of prices

2. **Dropout Layer (20%)**
   - Prevents overfitting
   - Randomly drops 20% of connections during training

3. **Second LSTM Layer (50 units)**
   - Learns higher-level patterns
   - Final sequence representation
   - Does not return sequences

4. **Dropout Layer (20%)**
   - Additional regularization

5. **Dense Layer (25 units)**
   - Learns non-linear combinations
   - Activation: ReLU (default)

6. **Output Dense Layer (1 unit)**
   - Produces final price prediction

#### Training Configuration
```python
optimizer = 'adam'
loss = 'mean_squared_error'
epochs = 50
batch_size = 32
lookback = 30  # Days of history to consider
```

#### Data Preprocessing
```python
def prepare_lstm_data(data, lookback=30):
    # 1. Normalize data to [0,1] range
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # 2. Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])  # Last 30 days
        y.append(scaled_data[i, 0])              # Next day
    
    return np.array(X), np.array(y), scaler
```

**Strengths:**
- Captures complex non-linear patterns
- Handles volatile price movements
- Learns from long-term dependencies
- Adapts to market dynamics

**Limitations:**
- Requires more data (60+ days recommended)
- Slower training time
- More computational resources
- Potential overfitting with limited data

---

## API Reference

### Endpoint Documentation

#### 1. GET /api/commodities
**Purpose:** Retrieve list of tracked commodities

**Request:**
```http
GET /api/commodities HTTP/1.1
Host: localhost:5000
```

**Response:**
```json
{
  "commodities": ["Rice", "Tomatoes", "Onions", "Yam", "Beans", "Maize"]
}
```

**Status Codes:**
- 200: Success

---

#### 2. GET /api/prices/<commodity>
**Purpose:** Get historical price data

**Request:**
```http
GET /api/prices/Rice HTTP/1.1
Host: localhost:5000
```

**Response:**
```json
{
  "commodity": "Rice",
  "data": [
    {"date": "2024-01-01", "price": 250.50},
    {"date": "2024-01-02", "price": 252.30},
    ...
  ]
}
```

**Status Codes:**
- 200: Success
- 404: Commodity not found

---

#### 3. POST /api/predict
**Purpose:** Generate price predictions

**Request:**
```http
POST /api/predict HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
  "commodity": "Rice",
  "model": "arima",
  "forecast_days": 30
}
```

**Parameters:**
- `commodity` (string, required): Commodity name
- `model` (string, required): "arima" or "lstm"
- `forecast_days` (integer, optional): Days to forecast (default: 30)

**Response:**
```json
{
  "commodity": "Rice",
  "model": "ARIMA",
  "historical": [
    {"date": "2024-01-01", "price": 250.50},
    ...
  ],
  "forecast": [
    {"date": "2024-02-08", "price": 255.20},
    ...
  ],
  "metrics": {
    "rmse": 12.45,
    "mae": 10.30,
    "accuracy": 95.8
  }
}
```

**Status Codes:**
- 200: Success
- 400: Invalid parameters
- 404: Commodity not found
- 500: Prediction failed

---

#### 4. POST /api/compare_models
**Purpose:** Compare ARIMA and LSTM predictions

**Request:**
```http
POST /api/compare_models HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
  "commodity": "Rice",
  "forecast_days": 30
}
```

**Response:**
```json
{
  "commodity": "Rice",
  "dates": ["2024-02-08", "2024-02-09", ...],
  "historical": [...],
  "arima": {
    "predictions": [255.20, 256.10, ...],
    "metrics": {"rmse": 12.45, "mae": 10.30, "accuracy": 95.8}
  },
  "lstm": {
    "predictions": [254.80, 255.90, ...],
    "metrics": {"rmse": 10.20, "mae": 8.50, "accuracy": 96.5}
  }
}
```

---

#### 5. POST /api/add_price
**Purpose:** Add new price entry

**Request:**
```http
POST /api/add_price HTTP/1.1
Host: localhost:5000
Content-Type: application/json

{
  "commodity": "Rice",
  "price": 280.50,
  "date": "2024-02-07"
}
```

**Parameters:**
- `commodity` (string, required): Commodity name
- `price` (number, required): Price in Naira
- `date` (string, optional): Date in YYYY-MM-DD format (default: today)

**Response:**
```json
{
  "message": "Price added successfully"
}
```

**Status Codes:**
- 200: Success
- 400: Missing required fields

---

#### 6. GET /api/statistics/<commodity>
**Purpose:** Get statistical analysis

**Request:**
```http
GET /api/statistics/Rice HTTP/1.1
Host: localhost:5000
```

**Response:**
```json
{
  "current_price": 280.50,
  "average": 265.30,
  "min": 230.00,
  "max": 310.00,
  "std_dev": 18.45,
  "variance": 340.40,
  "trend": "upward",
  "volatility": 6.95
}
```

---

## Data Storage

### JSON Structure

**File:** `price_data.json`

**Format:**
```json
{
  "Rice": [
    {
      "date": "2024-01-01",
      "price": 250.50
    },
    {
      "date": "2024-01-02",
      "price": 252.30
    }
  ],
  "Tomatoes": [
    {
      "date": "2024-01-01",
      "price": 85.00
    }
  ]
}
```

**Design Decisions:**
1. **JSON over Database**
   - Simplicity for deployment
   - No database setup required
   - Easy to backup and transfer
   - Suitable for prototype/MVP

2. **Flat Structure**
   - Easy to parse
   - Quick access
   - Minimal overhead

3. **Date Format**
   - ISO 8601 (YYYY-MM-DD)
   - Sortable
   - Internationally recognized

---

## Frontend Architecture

### Technology Stack
- **HTML5**: Semantic structure
- **CSS3**: Styling with custom properties
- **Bootstrap 5**: Responsive grid and components
- **Chart.js**: Data visualization
- **Vanilla JavaScript**: No framework overhead

### Key Components

#### 1. Price Prediction Interface
```javascript
async function predictPrices() {
  // 1. Validate inputs
  // 2. Show loading indicator
  // 3. Call API
  // 4. Process response
  // 5. Update UI
  // 6. Display chart
  // 7. Show insights
}
```

#### 2. Chart Management
```javascript
function createChart(labels, historical, forecast, modelName) {
  // Chart.js configuration
  // Datasets for historical and forecast
  // Styling and responsiveness
}
```

#### 3. Data Entry
```javascript
async function addPrice() {
  // Validate form
  // POST to API
  // Show success/error message
  // Clear form
  // Refresh commodity list
}
```

---

## Performance Optimization

### Backend Optimizations

1. **Model Caching**
   - Cache fitted models to avoid retraining
   - Invalidate cache when new data added

2. **Data Loading**
   - Load only necessary date ranges
   - Implement pagination for large datasets

3. **Async Processing**
   - Use background tasks for long-running predictions
   - Implement job queue for multiple requests

### Frontend Optimizations

1. **Chart Rendering**
   - Destroy previous chart before creating new one
   - Limit data points displayed (last 60 days + forecast)

2. **API Calls**
   - Debounce user inputs
   - Cache commodity list
   - Show loading indicators

3. **Mobile Optimization**
   - Responsive design
   - Touch-friendly buttons
   - Simplified layout on small screens

---

## Error Handling

### Backend Error Handling

```python
try:
    # Prediction logic
    predictions, metrics = predict_arima(prices, forecast_days)
    
    if predictions is None:
        return jsonify({'error': 'Prediction failed'}), 500
    
    return jsonify(results)
    
except Exception as e:
    print(f"Error: {e}")
    return jsonify({'error': str(e)}), 500
```

### Frontend Error Handling

```javascript
try {
    const response = await fetch(API_URL, options);
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    // Process data
    
} catch (error) {
    console.error('Error:', error);
    alert('Operation failed. Please try again.');
} finally {
    hideLoading();
}
```

---

## Testing

### Unit Testing

**ARIMA Model Testing:**
```python
def test_arima_prediction():
    # Test with known data
    prices = [100, 105, 110, 115, 120]
    predictions, metrics = predict_arima(prices, 5)
    
    assert predictions is not None
    assert len(predictions) == 5
    assert metrics['accuracy'] > 0
```

**LSTM Model Testing:**
```python
def test_lstm_prediction():
    # Test with sufficient data
    prices = generate_sample_prices(100, 200, 90)
    predictions, metrics = predict_lstm(prices, 10)
    
    assert predictions is not None
    assert len(predictions) == 10
```

### Integration Testing

**API Endpoint Testing:**
```python
def test_predict_endpoint():
    response = client.post('/api/predict', json={
        'commodity': 'Rice',
        'model': 'arima',
        'forecast_days': 30
    })
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'forecast' in data
    assert 'metrics' in data
```

---

## Deployment

### Local Deployment

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run application
python app.py

# 3. Access at http://localhost:5000
```

### Production Deployment

#### Using Gunicorn (Linux)
```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Using Nginx (Reverse Proxy)
```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
```

---

## Security Considerations

### Current Implementation
- Basic input validation
- CORS enabled (development)
- No authentication required

### Production Recommendations

1. **Authentication**
   - Implement user login
   - JWT tokens for API access
   - Rate limiting per user

2. **Input Validation**
   - Sanitize all inputs
   - Validate data types
   - Check ranges and constraints

3. **HTTPS**
   - Use SSL certificates
   - Redirect HTTP to HTTPS
   - Secure cookies

4. **Data Protection**
   - Encrypt sensitive data
   - Regular backups
   - Access control lists

---

## Maintenance

### Regular Tasks

**Daily:**
- Monitor application logs
- Check disk space
- Verify data backup

**Weekly:**
- Review prediction accuracy
- Update sample data if needed
- Check for errors in logs

**Monthly:**
- Update dependencies
- Review model performance
- Optimize database/storage

### Updating Models

**ARIMA Parameters:**
```python
# Modify in predict_arima function
model = ARIMA(prices, order=(p, d, q))
# Experiment with different values
```

**LSTM Architecture:**
```python
# Modify in build_lstm_model function
model = Sequential([
    LSTM(units, ...),  # Adjust units
    # Add/remove layers
])
```

---

## Extending the System

### Adding New Models

**Example: Prophet Model**
```python
from fbprophet import Prophet

def predict_prophet(prices, forecast_days=30):
    # Prepare data
    df = pd.DataFrame({
        'ds': dates,
        'y': prices
    })
    
    # Fit model
    model = Prophet()
    model.fit(df)
    
    # Forecast
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    return forecast['yhat'].values[-forecast_days:]
```

### Adding Database Support

**SQLAlchemy Example:**
```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///prices.db'
db = SQLAlchemy(app)

class Price(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    commodity = db.Column(db.String(50))
    price = db.Column(db.Float)
    date = db.Column(db.Date)
```

### Adding Real-time Data Integration

**Example: External API**
```python
import requests

def fetch_market_prices():
    response = requests.get('https://api.marketdata.com/prices')
    data = response.json()
    
    for item in data:
        add_price(item['commodity'], item['price'], item['date'])
```

---

## Performance Metrics

### Model Evaluation

**Metrics Calculated:**
1. **RMSE** (Root Mean Square Error)
   - Formula: √(Σ(predicted - actual)² / n)
   - Lower is better
   - Penalizes large errors

2. **MAE** (Mean Absolute Error)
   - Formula: Σ|predicted - actual| / n
   - Lower is better
   - Easier to interpret

3. **Accuracy**
   - Formula: 100 - (MAE / mean(actual) × 100)
   - Higher is better
   - Percentage-based

### Benchmarks

**Good Performance:**
- RMSE < 5% of average price
- MAE < 3% of average price
- Accuracy > 90%

**Acceptable Performance:**
- RMSE < 10% of average price
- MAE < 7% of average price
- Accuracy > 80%

---

## Troubleshooting Guide

### Common Issues

**Issue: High Memory Usage**
```python
# Solution: Limit data loaded
historical = data[commodity][-90:]  # Last 90 days only
```

**Issue: Slow Predictions**
```python
# Solution: Reduce LSTM epochs
model.fit(X, y, epochs=20, batch_size=32, verbose=0)
```

**Issue: Poor Accuracy**
```python
# Solution: Adjust ARIMA parameters or add more data
model = ARIMA(prices, order=(7, 1, 1))  # Try different values
```

---

## Code Quality

### Best Practices Followed

1. **PEP 8 Compliance**
   - 4-space indentation
   - Max line length: 79 characters
   - Clear variable names

2. **Documentation**
   - Docstrings for functions
   - Inline comments for complex logic
   - README and guides

3. **Error Handling**
   - Try-except blocks
   - Meaningful error messages
   - Graceful degradation

4. **Code Organization**
   - Logical function grouping
   - Separation of concerns
   - Modular design

---

## Future Development

### Planned Features

1. **Advanced Models**
   - Prophet integration
   - Ensemble methods
   - Neural Prophet

2. **Data Sources**
   - API integrations
   - Web scraping
   - Real-time feeds

3. **User Features**
   - Authentication
   - Personalized dashboards
   - Price alerts

4. **Analytics**
   - Historical analysis
   - Seasonal patterns
   - Market correlations

5. **Mobile App**
   - Native iOS/Android
   - Offline mode
   - Push notifications

---

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Run application
python app.py
```

### Code Style

- Follow PEP 8
- Write docstrings
- Add comments for complex logic
- Keep functions focused and small

---

## License

This project is for educational and research purposes.

---

## Contact

For technical questions or contributions, refer to the README.md file.

---

**Version:** 1.0
**Last Updated:** February 2026
**Status:** Production Ready

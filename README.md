# SmartMarketPriceTracker
This is a web-based system that uses machine learning (ARIMA and LSTM) to predict agricultural commodity prices.
# Smart Market Price Tracker for Rural Nigerian Traders

## Overview
A web-based AI-powered system that tracks and predicts agricultural commodity prices using machine learning algorithms (ARIMA and LSTM). This system is specifically designed to empower rural traders in Nigeria with reliable price forecasting capabilities.

## Features

### 1. AI-Powered Price Prediction
- **ARIMA Model**: Statistical time-series forecasting for short to medium-term predictions
- **LSTM Model**: Deep learning neural network for complex pattern recognition
- **Model Comparison**: Compare predictions from both models side-by-side
- Customizable forecast periods (7-90 days)

### 2. Interactive Visualizations
- Real-time price charts with historical and forecasted data
- Color-coded trend indicators
- Model performance metrics display
- Responsive design for mobile and desktop

### 3. Market Statistics
- Current, average, minimum, and maximum prices
- Price volatility analysis
- Trend indicators (upward/downward)
- Standard deviation and variance calculations

### 4. Data Management
- Add new price entries for any commodity
- Automatic data persistence
- Pre-loaded sample data for 6 major commodities
- Date-based price tracking

### 5. User-Friendly Interface
- Clean, intuitive design optimized for low-literacy users
- Visual indicators and icons
- Color-coded alerts and recommendations
- Mobile-responsive layout

## Technology Stack

### Backend
- **Python 3.8+**
- **Flask**: Web framework
- **Pandas & NumPy**: Data manipulation
- **Statsmodels**: ARIMA implementation
- **TensorFlow/Keras**: LSTM neural networks
- **Scikit-learn**: Data preprocessing and metrics

### Frontend
- **HTML5/CSS3**: Structure and styling
- **Bootstrap 5**: Responsive design
- **Chart.js**: Interactive visualizations
- **JavaScript (Vanilla)**: Client-side logic

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Setup Instructions

1. **Extract the project files to a directory**

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the application**
Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Making Price Predictions

1. **Select a Commodity**
   - Choose from the dropdown menu (Rice, Tomatoes, Onions, Yam, Beans, Maize)

2. **Choose Prediction Model**
   - ARIMA: Fast, statistical approach
   - LSTM: Advanced deep learning approach
   - Compare Both: See both predictions simultaneously

3. **Set Forecast Period**
   - Enter number of days (7-90)
   - Recommended: 30 days for balanced accuracy

4. **Click "Predict"**
   - System analyzes historical data
   - Generates forecasts
   - Displays results with metrics

### Understanding Results

#### Model Metrics
- **Accuracy**: Percentage of prediction reliability
- **RMSE** (Root Mean Square Error): Average prediction error in Naira
- **MAE** (Mean Absolute Error): Average absolute deviation

#### Chart Interpretation
- **Green Line**: Historical actual prices
- **Orange Dashed Line**: Forecasted prices
- **Shaded Area**: Confidence region

#### Market Insights
- Current vs. Predicted price comparison
- Expected percentage change
- Trading recommendations based on trends

### Adding New Price Data

1. Navigate to "Add New Price Data" section
2. Enter commodity name (existing or new)
3. Input price in Naira
4. Select date (defaults to today)
5. Click "Add"

### Viewing Statistics

1. Select commodity from statistics dropdown
2. Click "Show Statistics"
3. Review comprehensive metrics:
   - Current price
   - Average price
   - Price range (min/max)
   - Trend direction
   - Volatility percentage

## Machine Learning Models

### ARIMA (AutoRegressive Integrated Moving Average)
- **Type**: Statistical time-series model
- **Parameters**: (5, 1, 0) - optimized for agricultural prices
- **Best For**: Short-term predictions, stable trends
- **Advantages**: Fast, interpretable, works with limited data

### LSTM (Long Short-Term Memory)
- **Type**: Deep learning neural network
- **Architecture**: 2-layer LSTM with dropout regularization
- **Best For**: Complex patterns, long-term forecasts
- **Advantages**: Captures non-linear relationships, adapts to volatility

## Data Structure

### Price Data Format (JSON)
```json
{
  "Rice": [
    {
      "date": "2024-01-01",
      "price": 250.50
    },
    ...
  ],
  "Tomatoes": [...]
}
```

### Supported Commodities (Default)
- Rice
- Tomatoes
- Onions
- Yam
- Beans
- Maize

*Note: Users can add any commodity*

## API Endpoints

### GET /api/commodities
Returns list of all tracked commodities

### GET /api/prices/<commodity>
Returns historical price data for specified commodity

### POST /api/predict
Predicts future prices using specified model
```json
{
  "commodity": "Rice",
  "model": "arima",
  "forecast_days": 30
}
```

### POST /api/compare_models
Compares ARIMA and LSTM predictions

### POST /api/add_price
Adds new price entry
```json
{
  "commodity": "Rice",
  "price": 280.50,
  "date": "2024-02-07"
}
```

### GET /api/statistics/<commodity>
Returns statistical analysis of commodity prices

## Project Structure
```
smart-market-tracker/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── price_data.json        # Price data storage
├── models/                # Saved ML models
│
├── templates/
│   └── index.html         # Main web interface
│
├── static/ (optional)
│   ├── css/
│   ├── js/
│   └── images/
│
└── README.md              # This file
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   - Change port in app.py: `app.run(debug=True, port=5001)`

2. **Module Not Found**
   - Reinstall dependencies: `pip install -r requirements.txt`

3. **LSTM Training Slow**
   - Reduce epochs in `build_lstm_model()` function
   - Use ARIMA for faster results

4. **Prediction Accuracy Low**
   - Add more historical data
   - Adjust forecast period
   - Try different model

## Performance Considerations

### Optimization Tips
- ARIMA: Faster for quick predictions
- LSTM: More accurate for volatile markets
- Recommended data: Minimum 60 days for LSTM, 30 days for ARIMA

### Scalability
- Current: Supports multiple commodities
- Storage: JSON (easy to migrate to database)
- Future: Can integrate with PostgreSQL/MongoDB

## Future Enhancements

### Potential Features
1. Real-time data integration from market APIs
2. SMS/WhatsApp price alerts
3. Multi-language support (Hausa, Yoruba, Igbo)
4. Offline mode for rural areas
5. Export reports to PDF
6. User authentication and personalized dashboards
7. Integration with payment systems
8. Weather data correlation
9. Market location mapping
10. Community price sharing

## Research Context

This system implements the research objectives outlined in the academic document:
- Web-based interface for market metrics tracking
- Time-series machine learning for forecasting
- Model accuracy testing and validation

### Key Research Contributions
1. Addresses information asymmetry in rural markets
2. Reduces middlemen exploitation
3. Empowers low-literacy users with visual tools
4. Provides predictive capabilities beyond manual methods

## Academic References

The system is based on research examining:
- Agricultural price volatility in Nigeria
- Machine learning applications in commodity forecasting
- Digital platform adoption in rural markets
- Information access challenges in developing economies

## License

This project is developed for educational and research purposes to support rural agricultural traders in Nigeria.

## Support

For issues, questions, or contributions:
- Review the documentation
- Check troubleshooting section
- Examine code comments for implementation details

## Acknowledgments

Developed to address market price tracking challenges faced by rural Nigerian traders, implementing ARIMA and LSTM time-series forecasting models as specified in the research framework.

---

**Version**: 1.0
**Last Updated**: February 2026
**Status**: Production Ready

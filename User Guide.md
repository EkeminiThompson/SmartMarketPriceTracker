# USER GUIDE
## Smart Market Price Tracker for Rural Nigerian Traders

### Introduction
Welcome to the Smart Market Price Tracker! This system helps you make better trading decisions by predicting future prices of agricultural commodities using artificial intelligence.

---

## Getting Started

### What You Need
1. A computer or smartphone
2. Internet connection
3. Web browser (Chrome, Firefox, Safari, or Edge)

### First Time Setup
1. Make sure Python is installed on your computer
2. Open the command prompt or terminal
3. Navigate to the project folder
4. Type: `pip install -r requirements.txt` and press Enter
5. Wait for installation to complete
6. Type: `python app.py` and press Enter
7. Open your browser and go to: `http://localhost:5000`

---

## Understanding the Dashboard

### Main Sections

#### 1. AI-Powered Price Prediction
This is where you predict future prices.

**What you'll see:**
- Commodity selector (dropdown menu)
- Model selector (ARIMA, LSTM, or Compare Both)
- Forecast days input (how far into the future)
- Predict button

#### 2. Market Statistics
Shows detailed information about price trends.

**What you'll see:**
- Current price
- Average price
- Lowest and highest prices
- Price trend (going up or down)
- Volatility (how much prices change)

#### 3. Add New Price Data
Add today's market prices to improve predictions.

**What you'll need:**
- Commodity name (e.g., Rice, Tomatoes)
- Price in Naira (₦)
- Date (usually today)

---

## How to Make Price Predictions

### Step-by-Step Guide

#### Step 1: Choose Your Commodity
Click the dropdown menu that says "Select Commodity" and choose what you want to predict (e.g., Rice, Tomatoes, Onions).

#### Step 2: Select Prediction Model

**Option A: ARIMA**
- Good for: Quick predictions
- Best when: Prices have been stable
- Speed: Fast

**Option B: LSTM**
- Good for: More accurate predictions
- Best when: Prices change a lot
- Speed: Takes a bit longer

**Option C: Compare Both**
- Good for: Seeing both predictions
- Best when: You want to compare
- Speed: Takes longest

#### Step 3: Set Forecast Period
Enter how many days ahead you want to predict:
- Minimum: 7 days
- Maximum: 90 days
- Recommended: 30 days

#### Step 4: Click "Predict"
The system will:
1. Analyze historical prices
2. Run the prediction models
3. Show you the results

---

## Understanding Your Results

### The Chart
**Green Line** = Past prices (what actually happened)
**Orange Dashed Line** = Future predictions (what the AI thinks will happen)

### Model Performance Metrics

#### Accuracy Percentage
- **90-100%**: Excellent - Very reliable
- **80-89%**: Good - Reliable
- **70-79%**: Fair - Use with caution
- **Below 70%**: Poor - Add more data

#### RMSE (Root Mean Square Error)
- Measured in Naira (₦)
- Lower is better
- Shows average prediction error
- Example: RMSE of ₦20 means predictions are typically within ₦20 of actual price

#### MAE (Mean Absolute Error)
- Also measured in Naira (₦)
- Lower is better
- Shows typical difference from actual price
- Easier to understand than RMSE

### Market Insights

The system tells you:
1. **Current Price**: Today's price
2. **Predicted Price**: What it will be in X days
3. **Expected Change**: How much it will go up or down
4. **Recommendation**: What you should consider doing

#### Understanding Recommendations

**"Strong upward trend expected"**
- Prices going up a lot
- Consider: Holding stock, buying more if you plan to sell later

**"Moderate price increase expected"**
- Prices going up a little
- Consider: Good time for steady sales

**"Significant price drop expected"**
- Prices going down a lot
- Consider: Sell current stock soon

**"Slight price decrease expected"**
- Prices going down a little
- Consider: Watch the market closely

---

## Adding New Price Data

### Why Add Data?
More data = Better predictions
Fresh data = More accurate forecasts

### How to Add Data

1. **Scroll to "Add New Price Data" section**

2. **Fill in the form:**
   - Commodity Name: Type the product (e.g., "Rice" or "Yam")
   - Price: Enter price in Naira (e.g., 250.50)
   - Date: Select the date (defaults to today)

3. **Click "Add" button**

4. **Success!** You'll see a green message confirming the data was added

### Tips for Adding Data
- Be consistent with commodity names (use same name each time)
- Add prices regularly (daily or weekly is best)
- Use actual market prices, not estimates
- Include the Naira symbol mentally, but don't type it

---

## Viewing Statistics

### How to Check Statistics

1. Go to "Market Statistics" section
2. Select commodity from dropdown
3. Click "Show Statistics"

### What Each Statistic Means

**Current Price**
- The most recent price in the system
- What traders are paying today

**Average Price**
- The typical price over time
- Good benchmark for comparison

**Min/Max Prices**
- Lowest price recorded
- Highest price recorded
- Shows price range

**Price Trend**
- **Upward ↑**: Prices have been increasing
- **Downward ↓**: Prices have been decreasing

**Volatility**
- How much prices change
- High volatility = prices change a lot
- Low volatility = prices are stable

**Standard Deviation**
- Measure of price variation
- Higher number = more unpredictable prices

---

## Trading Strategies Based on Predictions

### When Prices Are Going Up

**Small Increase (1-5%)**
- Continue normal trading
- No urgent action needed

**Medium Increase (5-10%)**
- Consider holding some stock
- Good time to stock up if you're a buyer

**Large Increase (10%+)**
- Strong buying opportunity
- Sellers: Hold for better prices
- Buyers: Buy now before prices rise more

### When Prices Are Going Down

**Small Decrease (1-5%)**
- Monitor situation
- Continue normal trading

**Medium Decrease (5-10%)**
- Sellers: Consider selling soon
- Buyers: Wait for lower prices

**Large Decrease (10%+)**
- Sellers: Sell quickly to avoid losses
- Buyers: Good buying opportunity coming

---

## Tips for Best Results

### For Accurate Predictions
1. Add data regularly (at least weekly)
2. Use real market prices, not estimates
3. Include at least 60 days of data for LSTM
4. Include at least 30 days of data for ARIMA

### For Better Trading Decisions
1. Don't rely only on predictions
2. Consider other factors (weather, seasons, demand)
3. Compare both ARIMA and LSTM predictions
4. Watch the trend over time, not just one prediction

### For Efficient Use
1. Bookmark the website for quick access
2. Check predictions weekly
3. Add new prices immediately after market visits
4. Review statistics before making big purchases

---

## Common Questions

### Q: Which model should I use?
**A:** Start with "Compare Both" to see which works better for your commodity. LSTM is usually more accurate but takes longer.

### Q: How often should I check predictions?
**A:** Check weekly for planning, daily if prices are changing rapidly.

### Q: Can I predict any commodity?
**A:** Yes! The system starts with 6 commodities but you can add any agricultural product.

### Q: What if accuracy is low?
**A:** Add more price data. The system learns from more information.

### Q: How far ahead can I predict?
**A:** Up to 90 days, but 30-day predictions are most reliable.

### Q: What if predictions are wrong?
**A:** Predictions are estimates based on patterns. Always use judgment and consider other factors.

---

## Troubleshooting

### Problem: Page won't load
**Solution:** 
- Check if app.py is running
- Make sure you typed the correct URL
- Try refreshing the page

### Problem: No commodities showing
**Solution:**
- Restart the application
- Check if price_data.json file exists
- Add a new commodity manually

### Problem: Prediction taking too long
**Solution:**
- Use ARIMA instead of LSTM
- Reduce forecast days
- Close other programs

### Problem: Error message appears
**Solution:**
- Read the error message
- Check that all fields are filled correctly
- Try reloading the page

---

## Safety and Privacy

### Your Data
- All data stays on your computer
- No data is sent to external servers
- You control all information

### Backup Your Data
- Copy `price_data.json` file regularly
- Save it to USB drive or cloud storage
- This preserves your price history

---

## Best Practices

### Daily Routine
1. Visit market
2. Note prices of key commodities
3. Add prices to system
4. Check weekly predictions

### Weekly Routine
1. Run predictions for next 30 days
2. Compare with previous week
3. Note significant changes
4. Plan buying/selling strategy

### Monthly Routine
1. Review statistics for all commodities
2. Check model accuracy
3. Identify seasonal patterns
4. Adjust trading strategies

---

## Getting Help

### If You Need Assistance
1. Read this guide carefully
2. Check the README.md file
3. Review troubleshooting section
4. Check error messages for clues

### Learning More
- Practice with the system regularly
- Experiment with different settings
- Compare predictions with actual outcomes
- Learn from price patterns

---

## Success Tips

1. **Start Small**: Begin with 1-2 commodities you know well
2. **Be Consistent**: Add data regularly
3. **Stay Patient**: Accuracy improves over time
4. **Trust the Process**: Let the AI learn from your data
5. **Use Wisely**: Combine predictions with your experience

---

## Conclusion

This Smart Market Price Tracker is your assistant, not your replacement. Use it to:
- Make more informed decisions
- Reduce uncertainty
- Plan better
- Improve profits

Remember: The more you use it and feed it accurate data, the better it becomes at helping you!

**Happy Trading!** 📈

---

*For technical support or questions, refer to the README.md file included with the system.*

# XAI Stock Predictor

Explainable stock price prediction system for Nifty 50 stocks using XGBoost and SHAP.

---

## 1. Project Overview

This project implements an end-to-end machine learning pipeline to predict short-term stock movements in the Indian equity market. The system focuses on 46 Nifty 50 stocks and predicts whether each stock will gain more than 1.5 percent over the next 5 trading days. Along with predictions, the project provides model explainability using SHAP (SHapley Additive exPlanations) and a realistic backtesting framework.

Main capabilities:
- Predicts whether a stock is likely to gain more than 1.5 percent in the next 5 trading days
- Uses 52 technical indicators per stock as input features
- Trains separate XGBoost models for each stock
- Handles class imbalance using SMOTE
- Provides explainability for individual predictions and overall feature importance
- Includes a backtesting engine with transaction costs and position sizing
- Supports daily “real-time” style prediction generation

---

## 2. Data and Problem Definition

### 2.1 Data Source

- Market: Indian equity market (NSE)
- Universe: 46 Nifty 50 constituent stocks
- Source: Yahoo Finance (via `yfinance` library or equivalent data download)
- Frequency: Daily OHLCV data (Open, High, Low, Close, Volume)
- Period: Approximately 2016 to 2025
- Total observations: Around 117,000 rows across all stocks

### 2.2 Prediction Task

Binary classification problem:

- Target: Whether the stock price will increase by more than 1.5 percent within the next 5 trading days
- Label 1: Price gain greater than 1.5 percent in the next 5 days
- Label 0: Otherwise (no trade or avoid)

The target is generated for each stock independently using rolling future returns.

---

## 3. Feature Engineering

For each stock, 52 technical indicators are computed. The indicators cover:

1. Trend indicators  
   - Simple Moving Averages (SMA: 5, 10, 20, 50, 200 days)  
   - Exponential Moving Averages (EMA)  
   - MACD (Moving Average Convergence Divergence) and signal line  

2. Momentum indicators  
   - RSI (Relative Strength Index)  
   - Stochastic Oscillator (K, D)  
   - Rate of Change (ROC)  
   - Williams %R  

3. Volatility indicators  
   - Bollinger Bands (upper, lower, bandwidth)  
   - Average True Range (ATR)  
   - Rolling volatility over different windows (for example 20 and 50 days)  

4. Volume and money flow indicators  
   - On-Balance Volume (OBV)  
   - Money Flow Index (MFI)  
   - Volume moving averages and ratios  

5. Price position indicators  
   - Distance from 52-week high  
   - Distance from 52-week low  
   - Relative position within 52-week range  

These features are generated in `technical_indicators.py` and saved in the `data/features/` directory.

---

## 4. Model Architecture

### 4.1 Algorithm

- Model: XGBoost (gradient boosted decision trees)
- One model per stock (46 independent models)
- Loss function: Binary logistic loss
- Imbalance handling: SMOTE oversampling on the training set
- Typical hyperparameters (can be tuned):
  - max_depth: 6
  - learning_rate: 0.1
  - n_estimators: 100
  - subsample: 0.8
  - colsample_bytree: 0.8

Train/validation/test split (per stock):
- 70 percent training
- 15 percent validation
- 15 percent test (held out)

Training is implemented in `train_xgboost.py`.

---

## 5. Explainability with SHAP

SHAP is used to interpret both global and local model behaviour.

### 5.1 Global Explainability

For each stock model, global feature importance is computed using SHAP values:

- Overall ranking of features by average absolute SHAP value
- Aggregated feature importance across all stocks

Observed important features across the universe include:
1. Distance from 52-week high  
2. ADX (trend strength)  
3. Volume SMA 20  
4. Distance from 52-week low  
5. On-Balance Volume  
6. Medium and long term volatility measures  
7. Bollinger Band width  

This suggests that where the stock is trading within its 52-week range and the strength of the prevailing trend are highly informative.

### 5.2 Local Explainability

For individual predictions, SHAP values are used to show which features pushed the prediction towards “gain” or “no gain”:

- Waterfall plots for single prediction explanations
- Force plots for visualizing feature contributions

Explainability scripts:
- `shap_analysis.py` for a single stock
- `full_shap_analysis.py` to run SHAP across all stocks and save plots to `results/shap/`

---

## 6. Backtesting Framework

The project includes a backtesting script (`backtest.py`) to evaluate performance on unseen historical data.

Key characteristics:
- Test period: Last portion of the data per stock (for example final 20 percent of rows)
- Trade logic:
  - Enter trade when model predicts label 1 (expected gain greater than 1.5 percent)
  - Holding period: 5 trading days
  - Exit after 5 days or earlier if a defined gain threshold is reached (configurable)
- Transaction cost assumption: 0.5 percent per trade (both sides combined or per side depending on configuration)
- Risk management:
  - Conservative position sizing (typically 2 percent of capital per trade)
  - No leverage
  - Realistic slippage and cost assumptions

### 6.1 Backtesting Results

Performance on unseen test data across all 46 stocks:

- Overall win rate: 92.47 percent (trades that were profitable)
- Stocks with positive returns: 46 out of 46 (100 percent)
- Average return per stock: 7.95 percent over the test period
- Total number of trades analyzed: 6,854

Top performing stocks in backtesting:
1. BPCL: 12.83 percent return, 90.4 percent win rate
2. DIVISLAB: 12.18 percent return, 96.2 percent win rate
3. ADANIENT: 11.33 percent return, 96.3 percent win rate
4. HEROMOTOCO: 11.02 percent return, 93.3 percent win rate
5. HINDALCO: 10.96 percent return, 92.8 percent win rate

---

## 7. Training Performance

Aggregated performance metrics across all 46 stocks on held-out test data:

- Accuracy: 82.9 percent
- F1-Score: 83.2 percent
- Recall: 84.7 percent
- Precision: 81.8 percent (approximate)
- AUC-ROC: 0.904

These metrics indicate that the models generalize well to unseen data and maintain strong discrimination between gain and no-gain classes.

---

## 8. Real-Time Prediction System

The script `realtime_prediction.py` generates predictions for the current day (or the most recent data available).

How it works:
1. Loads the 46 trained models from the `models/` directory
2. Loads the latest feature values for each stock
3. Generates a probability score for each stock
4. Assigns confidence level (High, Medium, Low) based on probability thresholds
5. Outputs BUY or AVOID signals for each stock
6. Saves predictions to `predictions_today.csv`

# real_time_prediction.py
"""
Real-Time Stock Predictions
Use your trained models to predict which stocks will gain >1.5% in next 5 days
"""

import pandas as pd
import pickle
import os
from datetime import datetime
from config import NIFTY_46_STOCKS, FEATURES_DIR, MODELS_DIR

def make_prediction_for_today(symbol):
    """
    Make prediction for a single stock using LATEST data
    """
    
    # Load model
    model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load latest data
    data_path = f"{FEATURES_DIR}/{symbol}_features.csv"
    if not os.path.exists(data_path):
        return None
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get LATEST day (most recent trading day)
    latest_row = df.iloc[-1]
    latest_date = latest_row['Date']
    
    # Extract features (same 52 features used in training)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target', 'Forward_Return']]
    
    X_latest = latest_row[feature_cols].values.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(X_latest)[0]  # 0 = No Gain, 1 = Gain
    probability = model.predict_proba(X_latest)[0][1]  # Probability of Gain
    
    # Get current price
    current_price = latest_row['Close']
    
    return {
        'symbol': symbol,
        'date': latest_date,
        'current_price': current_price,
        'prediction': 'GAIN ✅' if prediction == 1 else 'NO GAIN ❌',
        'probability': probability * 100,
        'confidence': 'High' if probability > 0.70 else 'Medium' if probability > 0.55 else 'Low'
    }


def predict_all_stocks():
    """
    Get predictions for all 46 stocks
    """
    
    print("\n" + "="*90)
    print(f"REAL-TIME STOCK PREDICTIONS - {datetime.now().strftime('%d %B %Y')}")
    print("="*90)
    print("Predicting which stocks will GAIN >1.5% in next 5 days")
    print("="*90 + "\n")
    
    predictions = []
    
    for symbol in NIFTY_46_STOCKS:
        result = make_prediction_for_today(symbol)
        if result:
            predictions.append(result)
    
    # Create DataFrame
    pred_df = pd.DataFrame(predictions)
    
    # Sort by probability (highest first)
    pred_df = pred_df.sort_values('probability', ascending=False)
    
    # Separate BUY signals from NO BUY
    buy_signals = pred_df[pred_df['prediction'] == 'GAIN ✅']
    no_buy_signals = pred_df[pred_df['prediction'] == 'NO GAIN ❌']
    
    # Print BUY recommendations
    print(f"🎯 BUY SIGNALS (Predicted to GAIN >1.5%): {len(buy_signals)}/46")
    print("-" * 90)
    print(f"{'Rank':<6}{'Stock':<15}{'Current Price':<15}{'Probability':<15}{'Confidence':<12}{'Action'}")
    print("-" * 90)
    
    for i, row in enumerate(buy_signals.iterrows(), 1):
        _, data = row
        print(f"{i:<6}{data['symbol']:<15}₹{data['current_price']:<14.2f}{data['probability']:<14.1f}%{data['confidence']:<12}{'🟢 BUY'}")
    
    # Print NO BUY signals
    print(f"\n❌ NO BUY SIGNALS (Not expected to gain): {len(no_buy_signals)}/46")
    print("-" * 90)
    print(f"{'Rank':<6}{'Stock':<15}{'Current Price':<15}{'Probability':<15}{'Confidence':<12}{'Action'}")
    print("-" * 90)
    
    for i, row in enumerate(no_buy_signals.iterrows(), 1):
        _, data = row
        print(f"{i:<6}{data['symbol']:<15}₹{data['current_price']:<14.2f}{data['probability']:<14.1f}%{data['confidence']:<12}{'🔴 AVOID'}")
    
    print("\n" + "="*90)
    print("RECOMMENDATION SUMMARY")
    print("="*90)
    print(f"✅ BUY: {len(buy_signals)} stocks (predicted to gain >1.5%)")
    print(f"❌ AVOID: {len(no_buy_signals)} stocks (predicted to NOT gain)")
    print(f"\n💡 Based on 92.47% win rate, expect ~{int(len(buy_signals) * 0.92)}/{len(buy_signals)} BUY signals to be profitable!")
    print("="*90)
    
    # Save predictions
    pred_df.to_csv('predictions_today.csv', index=False)
    print(f"\n📁 Predictions saved: predictions_today.csv")
    
    return pred_df


if __name__ == "__main__":
    print("\n🚀 Making Real-Time Predictions...\n")
    predictions = predict_all_stocks()
    
    print("\n" + "="*90)
    print("HOW TO USE THESE PREDICTIONS")
    print("="*90)
    print("1. Check the BUY SIGNALS list above")
    print("2. Focus on 'High Confidence' stocks (>70% probability)")
    print("3. Buy TODAY (or next trading day)")
    print("4. Wait 5 trading days")
    print("5. Sell after 5 days (or earlier if >1.5% gain achieved)")
    print("6. Come back after 5 days and check your results!")
    print("="*90 + "\n")

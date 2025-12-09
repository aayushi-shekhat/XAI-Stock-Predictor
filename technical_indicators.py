# step2b_technical_indicators.py
"""
Step 2B: Technical Indicators
Calculates 30+ technical indicators for all 46 stocks
Uses ta-lib alternative (pandas-ta) for pure Python implementation
"""

import pandas as pd
import numpy as np
import os
from config import NIFTY_46_STOCKS, PROCESSED_DATA_DIR, FEATURES_DIR

# Technical Indicator Functions (Pure Python - No TA-Lib needed!)

def calculate_sma(df, column='Close', periods=[5, 10, 20, 50, 200]):
    """Simple Moving Averages"""
    for period in periods:
        df[f'SMA_{period}'] = df[column].rolling(window=period).mean()
    return df

def calculate_ema(df, column='Close', periods=[12, 26]):
    """Exponential Moving Averages"""
    for period in periods:
        df[f'EMA_{period}'] = df[column].ewm(span=period, adjust=False).mean()
    return df

def calculate_rsi(df, column='Close', period=14):
    """Relative Strength Index"""
    delta = df[column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, column='Close'):
    """MACD (Moving Average Convergence Divergence)"""
    ema_12 = df[column].ewm(span=12, adjust=False).mean()
    ema_26 = df[column].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def calculate_bollinger_bands(df, column='Close', period=20, std_dev=2):
    """Bollinger Bands"""
    df['BB_Middle'] = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (std * std_dev)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df[column] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def calculate_atr(df, period=14):
    """Average True Range (Volatility)"""
    # True Range already calculated in cleaning step
    df['ATR'] = df['True_Range'].rolling(window=period).mean()
    return df

def calculate_stochastic(df, period=14):
    """Stochastic Oscillator"""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    return df

def calculate_adx(df, period=14):
    """Average Directional Index (Trend Strength)"""
    # Plus/Minus Directional Movement
    df['Up_Move'] = df['High'] - df['High'].shift(1)
    df['Down_Move'] = df['Low'].shift(1) - df['Low']
    
    df['Plus_DM'] = np.where((df['Up_Move'] > df['Down_Move']) & (df['Up_Move'] > 0), df['Up_Move'], 0)
    df['Minus_DM'] = np.where((df['Down_Move'] > df['Up_Move']) & (df['Down_Move'] > 0), df['Down_Move'], 0)
    
    # Smoothed DM
    df['Plus_DM_Smooth'] = df['Plus_DM'].rolling(window=period).mean()
    df['Minus_DM_Smooth'] = df['Minus_DM'].rolling(window=period).mean()
    
    # Directional Indicators
    df['Plus_DI'] = 100 * df['Plus_DM_Smooth'] / df['ATR']
    df['Minus_DI'] = 100 * df['Minus_DM_Smooth'] / df['ATR']
    
    # ADX
    df['DX'] = 100 * np.abs(df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    # Clean up intermediate columns
    df = df.drop(['Up_Move', 'Down_Move', 'Plus_DM', 'Minus_DM', 
                  'Plus_DM_Smooth', 'Minus_DM_Smooth', 'DX'], axis=1)
    
    return df

def calculate_obv(df):
    """On-Balance Volume"""
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return df

def calculate_mfi(df, period=14):
    """Money Flow Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi_ratio = positive_mf / negative_mf
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    
    return df

def calculate_roc(df, column='Close', period=12):
    """Rate of Change (Momentum)"""
    df['ROC'] = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100
    return df

def calculate_williams_r(df, period=14):
    """Williams %R"""
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    return df

def calculate_cci(df, period=20):
    """Commodity Channel Index"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return df

def calculate_volume_features(df):
    """Volume-based features"""
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
    df['Volume_Change'] = df['Volume'].pct_change()
    return df

def calculate_price_features(df):
    """Additional price features"""
    # Distance from 52-week high/low
    df['Distance_52W_High'] = ((df['52W_High'] - df['Close']) / df['52W_High']) * 100
    df['Distance_52W_Low'] = ((df['Close'] - df['52W_Low']) / df['52W_Low']) * 100
    
    # Rolling volatility
    df['Volatility_20'] = df['Log_Return'].rolling(window=20).std() * np.sqrt(252)
    df['Volatility_50'] = df['Log_Return'].rolling(window=50).std() * np.sqrt(252)
    
    return df


def add_technical_indicators(symbol):
    """
    Add all technical indicators to a stock's cleaned data
    """
    
    print(f"  Adding indicators to {symbol}...", end=' ', flush=True)
    
    # Load cleaned data
    file_path = f"{PROCESSED_DATA_DIR}/{symbol}_cleaned.csv"
    if not os.path.exists(file_path):
        print("FAILED - File not found")
        return None
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate all indicators
    df = calculate_sma(df)
    df = calculate_ema(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_atr(df)
    df = calculate_stochastic(df)
    df = calculate_adx(df)
    df = calculate_obv(df)
    df = calculate_mfi(df)
    df = calculate_roc(df)
    df = calculate_williams_r(df)
    df = calculate_cci(df)
    df = calculate_volume_features(df)
    df = calculate_price_features(df)
    
    # Drop rows with NaN (due to rolling windows)
    initial_rows = len(df)
    df = df.dropna()
    final_rows = len(df)
    
    print(f"✓ {final_rows} rows ({len(df.columns)} features, dropped {initial_rows - final_rows} NaN rows)")
    
    return df


def process_all_stocks():
    """
    Add technical indicators to all 46 stocks
    """
    
    # Create output directory
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 2B: TECHNICAL INDICATORS")
    print("="*70)
    print(f"Total stocks: {len(NIFTY_46_STOCKS)}")
    print(f"Input:  {PROCESSED_DATA_DIR}/")
    print(f"Output: {FEATURES_DIR}/")
    print("="*70 + "\n")
    
    results = {}
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        
        try:
            df_features = add_technical_indicators(symbol)
            
            if df_features is not None:
                # Save
                output_path = f"{FEATURES_DIR}/{symbol}_features.csv"
                df_features.to_csv(output_path, index=False)
                results[symbol] = {
                    'rows': len(df_features),
                    'features': len(df_features.columns)
                }
        
        except Exception as e:
            print(f"  {symbol} - ERROR: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"Successfully processed: {len(results)}/46 stocks")
    
    if len(results) > 0:
        avg_rows = sum([r['rows'] for r in results.values()]) / len(results)
        avg_features = sum([r['features'] for r in results.values()]) / len(results)
        print(f"Average rows per stock: {avg_rows:.0f}")
        print(f"Average features per stock: {avg_features:.0f}")
        print(f"Total data points: {sum([r['rows'] for r in results.values()]):,}")
    
    print(f"\nFeature files saved in: {FEATURES_DIR}/")
    print("="*70)
    
    return results


if __name__ == "__main__":
    print("\nStarting technical indicator calculation...")
    print("This will add 30+ indicators to each stock (takes ~2 minutes)\n")
    
    results = process_all_stocks()
    
    # Show sample
    if results:
        first_stock = list(results.keys())[0]
        print(f"\nSample features from {first_stock}:")
        df_sample = pd.read_csv(f"{FEATURES_DIR}/{first_stock}_features.csv")
        print(f"\nColumns ({len(df_sample.columns)}):")
        for i, col in enumerate(df_sample.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nFirst 3 rows:")
        print(df_sample.head(3).to_string(index=False))
    
    print("\n✓ Ready for Step 2C: Statistical Features!")

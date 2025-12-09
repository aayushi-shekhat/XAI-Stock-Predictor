# step2a_data_cleaning.py
"""
Step 2A: Data Cleaning & Validation
- Load raw NSE data
- Handle missing values
- Remove duplicates
- Standardize columns
- Add basic derived features
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import NIFTY_46_STOCKS, RAW_DATA_DIR, PROCESSED_DATA_DIR

def load_raw_data(symbol):
    """Load raw CSV for a stock"""
    file_path = f"{RAW_DATA_DIR}/{symbol}.csv"
    if not os.path.exists(file_path):
        print(f"ERROR: {symbol}.csv not found!")
        return None
    
    df = pd.read_csv(file_path)
    return df


def clean_stock_data(symbol):
    """
    Clean and validate data for a single stock
    
    Steps:
    1. Load raw data
    2. Select essential columns
    3. Handle missing values
    4. Remove duplicates
    5. Sort by date
    6. Calculate basic features (returns, volatility)
    """
    
    print(f"  Processing {symbol}...", end=' ', flush=True)
    
    # Load data
    df = load_raw_data(symbol)
    if df is None:
        return None
    
    # Essential columns (some stocks have different column sets)
    essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    optional_cols = ['Turnover', 'VWAP', '52W_High', '52W_Low', 
                     'Deliverable_Qty', 'Delivery_Percent']
    
    # Keep only available columns
    cols_to_keep = [col for col in essential_cols + optional_cols if col in df.columns]
    df = df[cols_to_keep].copy()
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['Date'], keep='first')
    
    # Handle missing values
    # Forward fill (use previous day's value)
    df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].ffill()
    
    # Volume: fill with median volume
    if 'Volume' in df.columns:
        median_volume = df['Volume'].median()
        df['Volume'] = df['Volume'].fillna(median_volume)
    
    # VWAP: fill with Close if missing
    if 'VWAP' in df.columns:
        df['VWAP'] = df['VWAP'].fillna(df['Close'])
    
    # Remove any remaining rows with NaN in critical columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    
    # Data validation
    # 1. High should be >= Low
    df = df[df['High'] >= df['Low']]
    
    # 2. High should be >= Open and Close
    df = df[(df['High'] >= df['Open']) & (df['High'] >= df['Close'])]
    
    # 3. Low should be <= Open and Close
    df = df[(df['Low'] <= df['Open']) & (df['Low'] <= df['Close'])]
    
    # 4. Prices should be positive
    df = df[(df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)]
    
    # Calculate basic derived features
    
    # Daily returns
    df['Return'] = df['Close'].pct_change()
    
    # Log returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Intraday range
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100
    
    # True Range (for volatility)
    df['Prev_Close'] = df['Close'].shift(1)
    df['True_Range'] = df[['High', 'Prev_Close']].max(axis=1) - df[['Low', 'Prev_Close']].min(axis=1)
    
    # VWAP deviation (if VWAP available)
    if 'VWAP' in df.columns:
        df['VWAP_Deviation'] = ((df['Close'] - df['VWAP']) / df['VWAP']) * 100
    
    # Delivery percentage (if available)
    if 'Delivery_Percent' in df.columns:
        df['Delivery_Pct'] = df['Delivery_Percent']
    
    # Add stock symbol
    df['Symbol'] = symbol
    
    print(f"✓ {len(df)} rows cleaned")
    
    return df


def clean_all_stocks():
    """
    Clean data for all 46 stocks
    """
    
    # Create output directory
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 2A: DATA CLEANING & VALIDATION")
    print("="*70)
    print(f"Total stocks: {len(NIFTY_46_STOCKS)}")
    print(f"Input:  {RAW_DATA_DIR}/")
    print(f"Output: {PROCESSED_DATA_DIR}/")
    print("="*70 + "\n")
    
    results = {}
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        
        try:
            df_clean = clean_stock_data(symbol)
            
            if df_clean is not None:
                # Save cleaned data
                output_path = f"{PROCESSED_DATA_DIR}/{symbol}_cleaned.csv"
                df_clean.to_csv(output_path, index=False)
                results[symbol] = len(df_clean)
            else:
                print(f"  {symbol} - FAILED")
                
        except Exception as e:
            print(f"  {symbol} - ERROR: {str(e)[:50]}")
    
    # Summary
    print("\n" + "="*70)
    print("CLEANING COMPLETE!")
    print("="*70)
    print(f"Successfully cleaned: {len(results)}/46 stocks")
    
    if len(results) > 0:
        print(f"Average rows per stock: {sum(results.values()) / len(results):.0f}")
        print(f"Total clean data points: {sum(results.values()):,}")
    
    print(f"\nCleaned files saved in: {PROCESSED_DATA_DIR}/")
    print("="*70)
    
    return results


if __name__ == "__main__":
    print("\nStarting data cleaning...")
    print("This will validate and prepare data for feature engineering\n")
    
    results = clean_all_stocks()
    
    # Show sample cleaned data
    if results:
        first_stock = list(results.keys())[0]
        print(f"\nSample cleaned data from {first_stock}:")
        df_sample = pd.read_csv(f"{PROCESSED_DATA_DIR}/{first_stock}_cleaned.csv")
        print(df_sample.head(10).to_string(index=False))
        print(f"\nColumns: {', '.join(df_sample.columns)}")
        print(f"\nData types:\n{df_sample.dtypes}")
        print(f"\nMissing values:\n{df_sample.isnull().sum()}")
    
    print("\n✓ Ready for Step 2B: Technical Indicators!")

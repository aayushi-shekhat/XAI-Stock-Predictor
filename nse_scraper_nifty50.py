# nse_scraper_nifty46.py
"""
NSE Data Scraper - Final 46 Stocks
Updated to exclude 4 failed stocks (BAJAJFINSV, M&M, ULTRACEMCO, TITAN)
"""

from nsepython import *
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# FINAL 46 STOCKS (Successful Downloads Only)
NIFTY_46_STOCKS = [
    # Banking & Finance (9) - Removed: BAJAJFINSV
    'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN',
    'BAJFINANCE', 'INDUSINDBK', 'HDFCLIFE', 'SBILIFE',
    
    # IT (6)
    'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM',
    
    # FMCG (6)
    'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'TATACONSUM',
    
    # Pharma (5)
    'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP',
    
    # Auto (4) - Removed: M&M
    'MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'EICHERMOT',
    
    # Energy (5)
    'RELIANCE', 'ONGC', 'POWERGRID', 'NTPC', 'BPCL',
    
    # Infrastructure (3) - Removed: ULTRACEMCO
    'LT', 'ADANIENT', 'ADANIPORTS',
    
    # Metals (3)
    'HINDALCO', 'TATASTEEL', 'JSWSTEEL',
    
    # Telecom (1)
    'BHARTIARTL',
    
    # Others (4) - Removed: TITAN
    'ASIANPAINT', 'GRASIM', 'COALINDIA', 'HEROMOTOCO'
]

# Excluded stocks (for reference)
EXCLUDED_STOCKS = ['BAJAJFINSV', 'M&M', 'ULTRACEMCO', 'TITAN']


def download_stock_data(symbol, from_date, to_date):
    """
    Download historical data for a single stock
    
    Args:
        symbol: Stock symbol (e.g., 'HDFCBANK')
        from_date: Start date (datetime object)
        to_date: End date (datetime object)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # nsepython uses different date format
        from_date_str = from_date.strftime('%d-%m-%Y')
        to_date_str = to_date.strftime('%d-%m-%Y')
        
        # Fetch data using nsepython
        data = equity_history(symbol, 'EQ', from_date_str, to_date_str)
        
        if data is None or len(data) == 0:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Standardize column names
        column_mapping = {
            'CH_TIMESTAMP': 'Date',
            'CH_TRADE_HIGH_PRICE': 'High',
            'CH_TRADE_LOW_PRICE': 'Low',
            'CH_OPENING_PRICE': 'Open',
            'CH_CLOSING_PRICE': 'Close',
            'CH_TOT_TRADED_QTY': 'Volume',
            'CH_TOT_TRADED_VAL': 'Turnover',
            'VWAP': 'VWAP',
            'CH_52WEEK_HIGH_PRICE': '52W_High',
            'CH_52WEEK_LOW_PRICE': '52W_Low',
            'COP_DELIV_QTY': 'Deliverable_Qty',
            'COP_DELIV_PERC': 'Delivery_Percent'
        }
        
        # Rename columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Ensure Date column exists and is datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
        
        return df
        
    except Exception as e:
        print(f"      Error: {str(e)[:50]}")
        return None


def download_nifty46_data():
    """
    Download all 46 NIFTY stocks data
    """
    
    # Create output directory
    output_dir = 'data/nse_nifty50'  # Keep same dir name for consistency
    os.makedirs(output_dir, exist_ok=True)
    
    # Date range (10 years)
    to_date = datetime.now()
    from_date = datetime.now() - timedelta(days=3650)  # ~10 years
    
    # Results tracking
    results = {}
    failed = []
    
    # Header
    print("\n" + "="*70)
    print("NSE NIFTY 46 DATA DOWNLOADER (nsepython)")
    print("="*70)
    print(f"Total Stocks: {len(NIFTY_46_STOCKS)} (Curated list)")
    print(f"Excluded: {', '.join(EXCLUDED_STOCKS)}")
    print(f"Period: {from_date.strftime('%d-%m-%Y')} to {to_date.strftime('%d-%m-%Y')}")
    print(f"Output Directory: {output_dir}/")
    print("="*70 + "\n")
    
    # Download each stock
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] {symbol:<15} ", end='', flush=True)
        
        try:
            # Download data
            df = download_stock_data(symbol, from_date, to_date)
            
            if df is not None and len(df) > 0:
                # Save to CSV
                output_path = f"{output_dir}/{symbol}.csv"
                df.to_csv(output_path, index=False)
                
                results[symbol] = len(df)
                print(f"SUCCESS {len(df):4d} rows  ({df['Date'].min().date()} to {df['Date'].max().date()})")
                
            else:
                failed.append(symbol)
                print("FAILED - No data")
            
        except Exception as e:
            failed.append(symbol)
            print(f"FAILED - {str(e)[:40]}")
        
        # Wait between requests
        time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"Successful: {len(results)}/46 stocks")
    print(f"Failed: {len(failed)}/46 stocks")
    
    if len(results) > 0:
        print(f"\nAverage rows per stock: {sum(results.values()) / len(results):.0f}")
        print(f"Total data points: {sum(results.values()):,}")
    
    if failed:
        print(f"\nFailed stocks: {', '.join(failed)}")
        print("(Unexpected - these should all work!)")
    
    print(f"\nAll CSV files saved in: {output_dir}/")
    print("="*70 + "\n")
    
    return results, failed


if __name__ == "__main__":
    print("\nStarting NIFTY 46 data download using nsepython...")
    print("This library handles NSE authentication automatically!")
    print("Working with curated 46-stock list (excluded 4 problematic stocks)")
    print("Estimated time: 2-3 minutes\n")
    
    # Download all data
    results, failed = download_nifty46_data()
    
    # Show sample data
    if results:
        first_stock = list(results.keys())[0]
        print(f"\nSample data from {first_stock}:")
        try:
            df_sample = pd.read_csv(f'data/nse_nifty50/{first_stock}.csv')
            print(df_sample.head().to_string(index=False))
            print(f"\nColumns: {', '.join(df_sample.columns)}")
        except:
            pass
    
    print("\nAll done! Ready for feature engineering!")

# step2c_create_targets.py
"""
Step 2C: Create Target Labels
Creates binary classification target:
- 1 (Gain): Price increases >1.5% in next 5 days
- 0 (No Gain): Price does NOT increase >1.5% in next 5 days
"""

import pandas as pd
import numpy as np
import os
from config import NIFTY_46_STOCKS, FEATURES_DIR

# Configuration
PREDICTION_HORIZON = 5  # days
THRESHOLD = 1.5  # % gain threshold


def create_target_labels(symbol):
    """
    Create binary target labels for a stock
    
    Target Logic:
    - Look ahead 5 days
    - Calculate forward return: (Close_future - Close_today) / Close_today * 100
    - Label = 1 if forward_return >= 1.5%, else 0
    """
    
    print(f"  Creating labels for {symbol}...", end=' ', flush=True)
    
    # Load feature data
    file_path = f"{FEATURES_DIR}/{symbol}_features.csv"
    if not os.path.exists(file_path):
        print("FAILED - File not found")
        return None
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate forward return (5 days ahead)
    df['Close_Future'] = df['Close'].shift(-PREDICTION_HORIZON)
    df['Forward_Return'] = ((df['Close_Future'] - df['Close']) / df['Close']) * 100
    
    # Create binary target
    df['Target'] = (df['Forward_Return'] >= THRESHOLD).astype(int)
    
    # Remove last 5 rows (no future data available)
    df_labeled = df[:-PREDICTION_HORIZON].copy()
    
    # Drop intermediate columns
    df_labeled = df_labeled.drop(['Close_Future'], axis=1)
    
    # Calculate class distribution
    class_counts = df_labeled['Target'].value_counts()
    total = len(df_labeled)
    
    class_0 = class_counts.get(0, 0)
    class_1 = class_counts.get(1, 0)
    
    balance = (class_1 / total * 100) if total > 0 else 0
    
    print(f"✓ {total} rows | Class 0: {class_0} ({class_0/total*100:.1f}%) | Class 1: {class_1} ({balance:.1f}%)")
    
    return df_labeled


def process_all_stocks():
    """
    Create target labels for all 46 stocks
    """
    
    print("\n" + "="*80)
    print("STEP 2C: CREATE TARGET LABELS")
    print("="*80)
    print(f"Total stocks: {len(NIFTY_46_STOCKS)}")
    print(f"Prediction horizon: {PREDICTION_HORIZON} days")
    print(f"Gain threshold: >={THRESHOLD}%")
    print(f"Input/Output: {FEATURES_DIR}/")
    print("="*80 + "\n")
    
    results = {}
    all_targets = []
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        
        try:
            df_labeled = create_target_labels(symbol)
            
            if df_labeled is not None:
                # Save labeled data (overwrites feature file with added Target column)
                output_path = f"{FEATURES_DIR}/{symbol}_features.csv"
                df_labeled.to_csv(output_path, index=False)
                
                # Track statistics
                results[symbol] = {
                    'rows': len(df_labeled),
                    'class_0': (df_labeled['Target'] == 0).sum(),
                    'class_1': (df_labeled['Target'] == 1).sum()
                }
                
                all_targets.extend(df_labeled['Target'].tolist())
        
        except Exception as e:
            print(f"  {symbol} - ERROR: {str(e)[:60]}")
    
    # Overall statistics
    print("\n" + "="*80)
    print("TARGET LABEL CREATION COMPLETE!")
    print("="*80)
    print(f"Successfully labeled: {len(results)}/46 stocks")
    
    if len(results) > 0:
        total_rows = sum([r['rows'] for r in results.values()])
        total_class_0 = sum([r['class_0'] for r in results.values()])
        total_class_1 = sum([r['class_1'] for r in results.values()])
        
        print(f"\nAggregate Statistics:")
        print(f"  Total samples: {total_rows:,}")
        print(f"  Class 0 (No Gain): {total_class_0:,} ({total_class_0/total_rows*100:.1f}%)")
        print(f"  Class 1 (Gain):    {total_class_1:,} ({total_class_1/total_rows*100:.1f}%)")
        print(f"  Class ratio: {total_class_0/total_class_1:.2f}:1")
        
        # Check balance
        minority_pct = min(total_class_0, total_class_1) / total_rows * 100
        if minority_pct < 30:
            print(f"\n⚠️  WARNING: Imbalanced dataset (minority class: {minority_pct:.1f}%)")
            print(f"   Recommendation: Use SMOTE or class weights during training")
        else:
            print(f"\n✓ Dataset is reasonably balanced (minority: {minority_pct:.1f}%)")
    
    print(f"\nLabeled files saved in: {FEATURES_DIR}/")
    print(f"Each file now has {53 if results else 52} columns (added 'Target' + 'Forward_Return')")
    print("="*80)
    
    return results


def analyze_sample_stock(symbol='HDFCBANK'):
    """
    Detailed analysis of one stock's labels
    """
    
    file_path = f"{FEATURES_DIR}/{symbol}_features.csv"
    if not os.path.exists(file_path):
        print(f"{symbol} file not found!")
        return
    
    df = pd.read_csv(file_path)
    
    print(f"\n" + "="*80)
    print(f"DETAILED ANALYSIS: {symbol}")
    print("="*80)
    
    # Basic stats
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {', '.join(df.columns.tolist())}")
    
    # Target distribution
    print(f"\nTarget Distribution:")
    target_counts = df['Target'].value_counts().sort_index()
    for label, count in target_counts.items():
        pct = count / len(df) * 100
        print(f"  Class {label} ({'Gain' if label == 1 else 'No Gain'}): {count:,} samples ({pct:.1f}%)")
    
    # Forward return statistics
    print(f"\nForward Return (5-day) Statistics:")
    print(f"  Mean:   {df['Forward_Return'].mean():.3f}%")
    print(f"  Median: {df['Forward_Return'].median():.3f}%")
    print(f"  Std:    {df['Forward_Return'].std():.3f}%")
    print(f"  Min:    {df['Forward_Return'].min():.3f}%")
    print(f"  Max:    {df['Forward_Return'].max():.3f}%")
    
    # Sample predictions
    print(f"\nSample Predictions (last 10 days):")
    print(df[['Date', 'Close', 'Forward_Return', 'Target']].tail(10).to_string(index=False))
    
    # High gain examples
    high_gains = df[df['Forward_Return'] >= 5.0][['Date', 'Close', 'Forward_Return', 'Target']].head(5)
    if len(high_gains) > 0:
        print(f"\nExample High Gains (>5%):")
        print(high_gains.to_string(index=False))
    
    # Large losses
    large_losses = df[df['Forward_Return'] <= -5.0][['Date', 'Close', 'Forward_Return', 'Target']].head(5)
    if len(large_losses) > 0:
        print(f"\nExample Large Losses (<-5%):")
        print(large_losses.to_string(index=False))
    
    print("="*80)


if __name__ == "__main__":
    print("\nStarting target label creation...")
    print("This will create binary labels for all 46 stocks\n")
    
    # Create labels
    results = process_all_stocks()
    
    # Detailed analysis of sample stock
    if results:
        analyze_sample_stock('HDFCBANK')
    
    print("\n✓✓✓ DATA PREPARATION COMPLETE! ✓✓✓")
    print("\nNext Steps:")
    print("  1. Step 3: Train XGBoost models (46 models)")
    print("  2. Step 4: SHAP explainability analysis")
    print("  3. Step 5: Backtesting & evaluation")
    print("  4. Step 6: Deploy & showcase\n")

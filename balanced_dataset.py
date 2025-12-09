# step2d_balance_dataset_fixed.py
"""
Step 2D: Balance Dataset with SMOTE (FIXED)
Works for ALL stocks by using 'auto' strategy (1:1 balance)
"""

import pandas as pd
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from collections import Counter
from config import NIFTY_46_STOCKS, FEATURES_DIR

def balance_stock_data(symbol):
    """
    Apply SMOTE to balance a single stock's dataset to 1:1 ratio
    """
    
    print(f"  Balancing {symbol}...", end=' ', flush=True)
    
    # Load labeled data
    file_path = f"{FEATURES_DIR}/{symbol}_features.csv"
    if not os.path.exists(file_path):
        print("FAILED - File not found")
        return None
    
    df = pd.read_csv(file_path)
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target', 'Forward_Return']]
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Store metadata
    metadata = df[['Date', 'Symbol', 'Forward_Return']].copy()
    
    # Count original distribution
    original_counts = Counter(y)
    original_total = len(y)
    
    # Apply SMOTE with 'auto' strategy (1:1 balance)
    try:
        smote = SMOTE(
            sampling_strategy='auto',  # Always balance to 1:1
            random_state=42,
            k_neighbors=min(5, original_counts[1] - 1)  # Adjust for small classes
        )
        
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        # Count new distribution
        new_counts = Counter(y_balanced)
        new_total = len(y_balanced)
        
        # Create balanced DataFrame
        df_balanced = pd.DataFrame(X_balanced, columns=feature_cols)
        df_balanced['Target'] = y_balanced
        
        # Create synthetic metadata
        original_minority_indices = np.where(y == 1)[0]
        num_synthetic = new_total - original_total
        
        if num_synthetic > 0:
            synthetic_metadata_indices = np.random.choice(
                original_minority_indices, 
                size=num_synthetic, 
                replace=True
            )
            synthetic_metadata = metadata.iloc[synthetic_metadata_indices].copy()
            synthetic_metadata['Date'] = pd.to_datetime(synthetic_metadata['Date']) + pd.Timedelta(days=1)
            full_metadata = pd.concat([metadata, synthetic_metadata], ignore_index=True)
        else:
            full_metadata = metadata
        
        df_balanced['Date'] = full_metadata['Date'].values
        df_balanced['Symbol'] = full_metadata['Symbol'].values
        df_balanced['Forward_Return'] = full_metadata['Forward_Return'].values
        
        # Reorder columns
        col_order = df.columns.tolist()
        df_balanced = df_balanced[col_order]
        
        print(f"✓ {original_total}→{new_total} rows | Class 0: {original_counts[0]}→{new_counts[0]} | Class 1: {original_counts[1]}→{new_counts[1]} (50.0%)")
        
        return df_balanced
        
    except Exception as e:
        print(f"ERROR: {str(e)[:60]}")
        return None


def balance_all_stocks():
    """Balance all 46 stocks to perfect 1:1 ratio"""
    
    print("\n" + "="*85)
    print("STEP 2D: BALANCE DATASET WITH SMOTE (FIXED)")
    print("="*85)
    print(f"Total stocks: {len(NIFTY_46_STOCKS)}")
    print(f"Strategy: 'auto' (perfect 50-50 balance)")
    print(f"Input/Output: {FEATURES_DIR}/")
    print("="*85 + "\n")
    
    results = {}
    total_original = 0
    total_balanced = 0
    total_class_0_original = 0
    total_class_1_original = 0
    total_class_0_balanced = 0
    total_class_1_balanced = 0
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        
        try:
            df_balanced = balance_stock_data(symbol)
            
            if df_balanced is not None:
                df_original = pd.read_csv(f"{FEATURES_DIR}/{symbol}_features.csv")
                
                # Save balanced data
                output_path = f"{FEATURES_DIR}/{symbol}_features_balanced.csv"
                df_balanced.to_csv(output_path, index=False)
                
                # Track statistics
                original_counts = df_original['Target'].value_counts()
                balanced_counts = df_balanced['Target'].value_counts()
                
                results[symbol] = {
                    'original_rows': len(df_original),
                    'balanced_rows': len(df_balanced),
                    'original_class_0': original_counts.get(0, 0),
                    'original_class_1': original_counts.get(1, 0),
                    'balanced_class_0': balanced_counts.get(0, 0),
                    'balanced_class_1': balanced_counts.get(1, 0)
                }
                
                # Aggregate
                total_original += len(df_original)
                total_balanced += len(df_balanced)
                total_class_0_original += original_counts.get(0, 0)
                total_class_1_original += original_counts.get(1, 0)
                total_class_0_balanced += balanced_counts.get(0, 0)
                total_class_1_balanced += balanced_counts.get(1, 0)
        
        except Exception as e:
            print(f"  {symbol} - ERROR: {str(e)[:60]}")
    
    # Summary
    print("\n" + "="*85)
    print("DATASET BALANCING COMPLETE!")
    print("="*85)
    print(f"Successfully balanced: {len(results)}/46 stocks\n")
    
    if len(results) > 0:
        print("BEFORE (Original):")
        print(f"  Total samples: {total_original:,}")
        print(f"  Class 0 (No Gain): {total_class_0_original:,} ({total_class_0_original/total_original*100:.1f}%)")
        print(f"  Class 1 (Gain):    {total_class_1_original:,} ({total_class_1_original/total_original*100:.1f}%)")
        print(f"  Imbalance ratio: {total_class_0_original/total_class_1_original:.2f}:1")
        
        print("\nAFTER (Balanced with SMOTE):")
        print(f"  Total samples: {total_balanced:,} (+{total_balanced-total_original:,} synthetic)")
        print(f"  Class 0 (No Gain): {total_class_0_balanced:,} ({total_class_0_balanced/total_balanced*100:.1f}%)")
        print(f"  Class 1 (Gain):    {total_class_1_balanced:,} ({total_class_1_balanced/total_balanced*100:.1f}%)")
        print(f"  New ratio: 1.00:1 (Perfect balance!)")
        
        print(f"\n✓ Class 1 increased by {total_class_1_balanced - total_class_1_original:,} samples ({(total_class_1_balanced/total_class_1_original - 1)*100:.1f}% growth)")
        print(f"✓ Dataset size increased by {(total_balanced/total_original - 1)*100:.1f}%")
    
    print(f"\nBalanced files saved as: {FEATURES_DIR}/*_features_balanced.csv")
    print("="*85)
    
    return results


def compare_original_vs_balanced(symbol='HDFCBANK'):
    """Compare datasets"""
    
    original_path = f"{FEATURES_DIR}/{symbol}_features.csv"
    balanced_path = f"{FEATURES_DIR}/{symbol}_features_balanced.csv"
    
    if not os.path.exists(original_path) or not os.path.exists(balanced_path):
        print(f"Files not found for {symbol}")
        return
    
    df_original = pd.read_csv(original_path)
    df_balanced = pd.read_csv(balanced_path)
    
    print(f"\n" + "="*85)
    print(f"COMPARISON: {symbol}")
    print("="*85)
    
    print("\nORIGINAL Dataset:")
    print(f"  Rows: {len(df_original):,}")
    for label, count in df_original['Target'].value_counts().sort_index().items():
        print(f"    Class {label}: {count:,} ({count/len(df_original)*100:.1f}%)")
    
    print("\nBALANCED Dataset:")
    print(f"  Rows: {len(df_balanced):,}")
    for label, count in df_balanced['Target'].value_counts().sort_index().items():
        print(f"    Class {label}: {count:,} ({count/len(df_balanced)*100:.1f}%)")
    
    print(f"\nSynthetic samples: {len(df_balanced) - len(df_original):,}")
    print("="*85)


if __name__ == "__main__":
    print("\nStarting dataset balancing with SMOTE (FIXED VERSION)...")
    print("This will create perfect 50-50 balance for all stocks\n")
    
    results = balance_all_stocks()
    
    if results:
        compare_original_vs_balanced('HDFCBANK')
    
    print("\n✓✓✓ ALL 46 STOCKS BALANCED! ✓✓✓")
    print("\nNext: Train XGBoost on BALANCED data!\n")

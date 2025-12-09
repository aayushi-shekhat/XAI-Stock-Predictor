# step4b_full_shap_analysis.py
"""
Extended SHAP Analysis - ALL 46 Stocks
- Feature importance for every stock (fast)
- Detailed plots only for top performers (slow)
- Comprehensive aggregate insights
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from config import NIFTY_46_STOCKS, FEATURES_DIR, MODELS_DIR, RESULTS_DIR

SHAP_DIR = f"{RESULTS_DIR}/shap"
os.makedirs(SHAP_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_and_data(symbol):
    """Load trained model and test data"""
    
    model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
    data_path = f"{FEATURES_DIR}/{symbol}_features_balanced.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target', 'Forward_Return']]
    
    X = df[feature_cols]
    y = df['Target']
    
    return model, X, feature_cols


def calculate_feature_importance_fast(symbol):
    """
    Fast feature importance calculation (no plots)
    Returns importance DataFrame only
    """
    
    print(f"  [{symbol:<15s}] ", end='', flush=True)
    
    model, X, feature_cols = load_model_and_data(symbol)
    
    if model is None:
        print("FAILED")
        return None
    
    # Sample for speed (200 samples is enough for importance)
    X_sample = X.sample(n=min(200, len(X)), random_state=42)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_shap,
        'symbol': symbol
    }).sort_values('importance', ascending=False)
    
    # Print top 3
    top_3 = importance_df.head(3)['feature'].tolist()
    print(f"✓ Top 3: {', '.join(top_3[:3])}")
    
    return importance_df


def create_detailed_plots(symbol):
    """
    Create detailed SHAP plots for a single stock
    (Only for top performers to save time)
    """
    
    model, X, feature_cols = load_model_and_data(symbol)
    
    if model is None:
        return
    
    X_sample = X.sample(n=min(500, len(X)), random_state=42)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 1. Feature Importance Bar Chart
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    top_15 = importance_df.head(15)
    plt.barh(range(len(top_15)), top_15['importance'], color='steelblue')
    plt.yticks(range(len(top_15)), top_15['feature'])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title(f'{symbol} - Top 15 Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/{symbol}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Summary Plot (Beeswarm)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
    plt.title(f'{symbol} - SHAP Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{SHAP_DIR}/{symbol}_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Waterfall Plot (Sample Prediction)
    gain_indices = np.where(model.predict(X_sample) == 1)[0]
    if len(gain_indices) > 0:
        idx = gain_indices[0]
        explanation = shap.Explanation(
            values=shap_values[idx],
            base_values=explainer.expected_value,
            data=X_sample.iloc[idx].values,
            feature_names=X_sample.columns.tolist()
        )
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f'{symbol} - Prediction Explanation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{SHAP_DIR}/{symbol}_waterfall.png", dpi=300, bbox_inches='tight')
        plt.close()


def analyze_all_46_stocks(detailed_plots_for_top_n=5):
    """
    Analyze ALL 46 stocks:
    - Feature importance for all (fast)
    - Detailed plots only for top N (slow)
    """
    
    print("\n" + "="*90)
    print("EXTENDED SHAP ANALYSIS - ALL 46 STOCKS")
    print("="*90)
    print(f"Feature importance: ALL 46 stocks (~5 minutes)")
    print(f"Detailed plots: Top {detailed_plots_for_top_n} performers (~2 minutes)")
    print(f"Output: {SHAP_DIR}/")
    print("="*90 + "\n")
    
    # Get top performers
    try:
        import json
        with open(f"{RESULTS_DIR}/training_results_baseline.json", 'r') as f:
            results = json.load(f)
        
        top_stocks = sorted(results, key=lambda x: x['f1_score'], reverse=True)[:detailed_plots_for_top_n]
        top_symbols = [s['symbol'] for s in top_stocks]
        
        print(f"Top {detailed_plots_for_top_n} Performers (will get detailed plots):")
        for i, s in enumerate(top_stocks, 1):
            print(f"  {i}. {s['symbol']}: F1={s['f1_score']:.3f}, Acc={s['accuracy']:.3f}")
        print()
    except:
        top_symbols = NIFTY_46_STOCKS[:detailed_plots_for_top_n]
    
    # PART 1: Feature importance for ALL 46 stocks
    print("PART 1: Calculating feature importance for ALL 46 stocks...")
    print("-" * 90)
    
    all_importance = []
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        importance_df = calculate_feature_importance_fast(symbol)
        
        if importance_df is not None:
            all_importance.append(importance_df)
    
    # PART 2: Detailed plots for top performers
    print("\n" + "-" * 90)
    print(f"PART 2: Creating detailed plots for top {detailed_plots_for_top_n} stocks...")
    print("-" * 90 + "\n")
    
    for i, symbol in enumerate(top_symbols, 1):
        print(f"[{i}/{detailed_plots_for_top_n}] Creating plots for {symbol}...")
        create_detailed_plots(symbol)
    
    # PART 3: Aggregate Analysis
    print("\n" + "="*90)
    print("PART 3: AGGREGATE ANALYSIS (ALL 46 STOCKS)")
    print("="*90 + "\n")
    
    if len(all_importance) > 0:
        # Combine all importance DataFrames
        combined_importance = pd.concat(all_importance, ignore_index=True)
        
        # Average importance per feature
        avg_importance = combined_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        print("Top 20 Most Important Features (Averaged Across ALL 46 Stocks):")
        for i, (feature, importance) in enumerate(avg_importance.head(20).items(), 1):
            print(f"  {i:2d}. {feature:<25s} → {importance:.4f}")
        
        # Plot aggregate importance (Top 20)
        plt.figure(figsize=(12, 10))
        top_20 = avg_importance.head(20)
        plt.barh(range(len(top_20)), top_20.values, color='coral')
        plt.yticks(range(len(top_20)), top_20.index)
        plt.xlabel('Average SHAP Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Top 20 Features Across ALL 46 Stocks', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{SHAP_DIR}/aggregate_all_46_stocks.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        avg_importance.to_csv(f"{SHAP_DIR}/feature_importance_all_46_stocks.csv")
        combined_importance.to_csv(f"{SHAP_DIR}/feature_importance_per_stock.csv", index=False)
        
        # Sector-wise analysis
        print("\n" + "-" * 90)
        print("SECTOR-WISE TOP FEATURES:")
        print("-" * 90)
        
        sector_map = {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'BAJFINANCE', 'INDUSINDBK'],
            'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'TATACONSUM'],
            'Metals': ['HINDALCO', 'TATASTEEL', 'JSWSTEEL']
        }
        
        for sector, stocks in sector_map.items():
            sector_data = combined_importance[combined_importance['symbol'].isin(stocks)]
            if len(sector_data) > 0:
                sector_avg = sector_data.groupby('feature')['importance'].mean().sort_values(ascending=False)
                print(f"\n{sector} Sector (Top 5):")
                for i, (feature, importance) in enumerate(sector_avg.head(5).items(), 1):
                    print(f"  {i}. {feature:<25s} → {importance:.4f}")
    
    print("\n" + "="*90)
    print("EXTENDED SHAP ANALYSIS COMPLETE!")
    print("="*90)
    print(f"\nGenerated Files:")
    print(f"  - Feature importance: ALL 46 stocks")
    print(f"  - Detailed plots: Top {detailed_plots_for_top_n} stocks ({detailed_plots_for_top_n * 3} images)")
    print(f"  - Aggregate analysis: ALL 46 stocks combined")
    print(f"  - Sector-wise rankings")
    print(f"\nFiles saved in: {SHAP_DIR}/")
    print("="*90)
    
    return combined_importance


if __name__ == "__main__":
    print("\nExtended SHAP Analysis - ALL 46 STOCKS")
    print("This will analyze feature importance for every stock\n")
    
    # Analyze all stocks
    # detailed_plots_for_top_n: How many stocks get full plots (default: 5)
    combined_importance = analyze_all_46_stocks(detailed_plots_for_top_n=5)
    
    print("\n✓✓✓ EXTENDED ANALYSIS COMPLETE! ✓✓✓")
    print("\nYou now have:")
    print("  ✅ Feature importance for ALL 46 stocks")
    print("  ✅ Detailed plots for top 5 performers")
    print("  ✅ Aggregate insights across all stocks")
    print("  ✅ Sector-wise feature rankings\n")

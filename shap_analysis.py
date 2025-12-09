# step4_shap_analysis.py
"""
Step 4: SHAP Explainability Analysis
- Explains XGBoost predictions using SHAP values
- Generates feature importance rankings
- Creates visualizations for top performers
- Shows why model predicts Gain vs No Gain
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from config import NIFTY_46_STOCKS, FEATURES_DIR, MODELS_DIR, RESULTS_DIR

# Create SHAP output directory
SHAP_DIR = f"{RESULTS_DIR}/shap"
os.makedirs(SHAP_DIR, exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_and_data(symbol):
    """Load trained model and test data for a stock"""
    
    # Load model
    model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found for {symbol}")
        return None, None, None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load balanced data
    data_path = f"{FEATURES_DIR}/{symbol}_features_balanced.csv"
    if not os.path.exists(data_path):
        print(f"ERROR: Data not found for {symbol}")
        return None, None, None
    
    df = pd.read_csv(data_path)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target', 'Forward_Return']]
    
    X = df[feature_cols]
    y = df['Target']
    
    return model, X, feature_cols


def calculate_shap_values(model, X, sample_size=500):
    """
    Calculate SHAP values for a model
    Uses TreeExplainer for XGBoost (fast and accurate)
    """
    
    # Sample data if too large (for speed)
    if len(X) > sample_size:
        X_sample = X.sample(n=sample_size, random_state=42)
    else:
        X_sample = X
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    return explainer, shap_values, X_sample


def plot_feature_importance(shap_values, feature_names, symbol, top_n=15):
    """
    Plot top N most important features
    """
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_shap
    }).sort_values('importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Mean |SHAP Value| (Average Impact on Prediction)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{symbol} - Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Save
    plt.savefig(f"{SHAP_DIR}/{symbol}_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df


def plot_shap_summary(shap_values, X_sample, symbol, top_n=20):
    """
    SHAP summary plot (beeswarm) - shows feature impact and distribution
    """
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, max_display=top_n, show=False)
    plt.title(f'{symbol} - SHAP Summary (Impact & Distribution)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save
    plt.savefig(f"{SHAP_DIR}/{symbol}_shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_waterfall(explainer, shap_values, X_sample, symbol, prediction_idx=0):
    """
    Waterfall plot for a single prediction
    Shows how each feature contributes to the final prediction
    """
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[prediction_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[prediction_idx].values,
        feature_names=X_sample.columns.tolist()
    )
    
    # Plot
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=15, show=False)
    plt.title(f'{symbol} - Prediction Explanation (Sample)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(f"{SHAP_DIR}/{symbol}_waterfall.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_shap_force(explainer, shap_values, X_sample, symbol, prediction_idx=0):
    """
    Force plot for a single prediction (alternative to waterfall)
    """
    
    # Generate force plot
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[prediction_idx],
        X_sample.iloc[prediction_idx],
        matplotlib=True,
        show=False
    )
    
    plt.title(f'{symbol} - Force Plot (Sample Prediction)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(f"{SHAP_DIR}/{symbol}_force_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def analyze_single_stock(symbol, create_plots=True):
    """
    Complete SHAP analysis for a single stock
    """
    
    print(f"  Analyzing {symbol}...", end=' ', flush=True)
    
    # Load model and data
    model, X, feature_cols = load_model_and_data(symbol)
    
    if model is None:
        print("FAILED")
        return None
    
    # Calculate SHAP values
    explainer, shap_values, X_sample = calculate_shap_values(model, X, sample_size=500)
    
    # Feature importance
    importance_df = plot_feature_importance(shap_values, feature_cols, symbol, top_n=15)
    
    if create_plots:
        # Summary plot
        plot_shap_summary(shap_values, X_sample, symbol, top_n=20)
        
        # Waterfall for a "Gain" prediction
        gain_indices = np.where(model.predict(X_sample) == 1)[0]
        if len(gain_indices) > 0:
            plot_shap_waterfall(explainer, shap_values, X_sample, symbol, prediction_idx=gain_indices[0])
    
    print(f"✓ Top features: {', '.join(importance_df.head(3)['feature'].tolist())}")
    
    return importance_df


def analyze_all_stocks(top_performers=5):
    """
    Analyze top performing stocks with detailed visualizations
    """
    
    print("\n" + "="*90)
    print("STEP 4: SHAP EXPLAINABILITY ANALYSIS")
    print("="*90)
    print(f"Analyzing: Top {top_performers} performers + aggregated insights")
    print(f"Output directory: {SHAP_DIR}/")
    print("="*90 + "\n")
    
    # Get top performers from training results
    try:
        import json
        with open(f"{RESULTS_DIR}/training_results_baseline.json", 'r') as f:
            results = json.load(f)
        
        # Sort by F1-score
        top_stocks = sorted(results, key=lambda x: x['f1_score'], reverse=True)[:top_performers]
        top_symbols = [s['symbol'] for s in top_stocks]
        
        print("Top Performers (by F1-Score):")
        for i, s in enumerate(top_stocks, 1):
            print(f"  {i}. {s['symbol']}: F1={s['f1_score']:.3f}, Acc={s['accuracy']:.3f}")
        print()
        
    except:
        # Fallback to first 5 stocks
        top_symbols = NIFTY_46_STOCKS[:top_performers]
        print(f"Using first {top_performers} stocks for analysis\n")
    
    # Analyze each top performer
    all_importance = []
    
    for i, symbol in enumerate(top_symbols, 1):
        print(f"[{i}/{top_performers}] ", end='')
        importance_df = analyze_single_stock(symbol, create_plots=True)
        
        if importance_df is not None:
            importance_df['symbol'] = symbol
            all_importance.append(importance_df)
    
    # Aggregate feature importance across all stocks
    print("\n" + "="*90)
    print("AGGREGATING INSIGHTS...")
    print("="*90)
    
    if len(all_importance) > 0:
        # Combine all importance DataFrames
        combined_importance = pd.concat(all_importance, ignore_index=True)
        
        # Calculate average importance per feature
        avg_importance = combined_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        print("\nTop 10 Most Important Features (Averaged Across All Stocks):")
        for i, (feature, importance) in enumerate(avg_importance.head(10).items(), 1):
            print(f"  {i:2d}. {feature:<20s} → {importance:.4f}")
        
        # Plot aggregate importance
        plt.figure(figsize=(12, 8))
        top_15 = avg_importance.head(15)
        plt.barh(range(len(top_15)), top_15.values, color='coral')
        plt.yticks(range(len(top_15)), top_15.index)
        plt.xlabel('Average SHAP Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top 15 Features Across Top {top_performers} Stocks', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{SHAP_DIR}/aggregate_feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save aggregate importance
        avg_importance.to_csv(f"{SHAP_DIR}/aggregate_feature_importance.csv")
        
        print(f"\nAggregate plot saved: {SHAP_DIR}/aggregate_feature_importance.png")
    
    print("\n" + "="*90)
    print("SHAP ANALYSIS COMPLETE!")
    print("="*90)
    print(f"Generated visualizations: {SHAP_DIR}/")
    print("  - Feature importance plots (per stock)")
    print("  - SHAP summary plots (per stock)")
    print("  - Waterfall plots (per stock)")
    print("  - Aggregate feature importance (all stocks)")
    print("="*90)


def quick_analysis(symbol):
    """
    Quick SHAP analysis for any stock (for testing/demo)
    """
    
    print(f"\nQuick SHAP Analysis: {symbol}")
    print("="*50)
    
    model, X, feature_cols = load_model_and_data(symbol)
    
    if model is None:
        return
    
    # Calculate SHAP
    explainer, shap_values, X_sample = calculate_shap_values(model, X, sample_size=200)
    
    # Feature importance
    importance_df = plot_feature_importance(shap_values, feature_cols, symbol, top_n=10)
    
    print("\nTop 10 Features:")
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\nPlots saved in: {SHAP_DIR}/")
    print("="*50)


if __name__ == "__main__":
    print("\nStarting SHAP Explainability Analysis...")
    print("This will explain WHY the model predicts Gain vs No Gain\n")
    
    # Analyze top 5 performers
    analyze_all_stocks(top_performers=5)
    
    print("\n✓✓✓ SHAP ANALYSIS COMPLETE! ✓✓✓")
    print("\nNext Steps:")
    print("  1. Review generated plots in results/shap/")
    print("  2. Step 5: Backtesting (test on unseen data)")
    print("  3. Step 6: Deploy & Showcase\n")

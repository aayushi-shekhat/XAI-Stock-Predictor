# step3_train_xgboost.py
"""
Step 3: Train XGBoost Models
- Trains 46 stock-specific models on balanced data
- Uses Optuna for hyperparameter optimization
- Evaluates performance with comprehensive metrics
- Saves best models for deployment
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import optuna
from config import NIFTY_46_STOCKS, FEATURES_DIR, MODELS_DIR, RESULTS_DIR

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_balanced_data(symbol):
    """Load balanced dataset for a stock"""
    
    file_path = f"{FEATURES_DIR}/{symbol}_features_balanced.csv"
    if not os.path.exists(file_path):
        print(f"ERROR: {symbol}_features_balanced.csv not found!")
        return None, None, None, None, None, None
    
    df = pd.read_csv(file_path)
    
    # Features (exclude non-feature columns)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target', 'Forward_Return']]
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Train/Val/Test split (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=42,
        stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost_simple(symbol, X_train, y_train, X_val, y_val):
    """
    Train XGBoost with default hyperparameters (fast baseline)
    """
    
    model = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.03,
        n_estimators=800,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=1,
        scale_pos_weight=1.0,  # Already balanced with SMOTE
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50,
        verbosity=0
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def train_xgboost_optimized(symbol, X_train, y_train, X_val, y_val, n_trials=50):
    """
    Train XGBoost with Optuna hyperparameter optimization (better but slower)
    """
    
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': 1.0,
            'random_state': 42,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 30,
            'verbosity': 0
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Optimize for F1-score (balanced metric)
        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        
        return f1
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize', study_name=f'{symbol}_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    # Train final model with best params
    best_params = study.best_params
    best_params.update({
        'scale_pos_weight': 1.0,
        'random_state': 42,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 30,
        'verbosity': 0
    })
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model, best_params


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation
    """
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class metrics
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp),
            'fn': int(fn), 'tp': int(tp)
        },
        'class_0': {
            'precision': class_report['0']['precision'],
            'recall': class_report['0']['recall'],
            'f1_score': class_report['0']['f1-score']
        },
        'class_1': {
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score']
        }
    }
    
    return results


def train_single_stock(symbol, optimize=False):
    """
    Train model for a single stock
    """
    
    print(f"  Training {symbol}...", end=' ', flush=True)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_balanced_data(symbol)
    
    if X_train is None:
        print("FAILED - Data not found")
        return None
    
    # Train model
    try:
        if optimize:
            model, best_params = train_xgboost_optimized(symbol, X_train, y_train, X_val, y_val, n_trials=30)
        else:
            model = train_xgboost_simple(symbol, X_train, y_train, X_val, y_val)
            best_params = None
        
        # Evaluate
        results = evaluate_model(model, X_test, y_test)
        
        # Save model
        model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save results
        results['symbol'] = symbol
        results['train_samples'] = len(X_train)
        results['test_samples'] = len(X_test)
        results['best_params'] = best_params
        
        print(f"✓ Acc: {results['accuracy']:.3f} | F1: {results['f1_score']:.3f} | Recall(1): {results['recall']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")
        return None


def train_all_stocks(optimize=False):
    """
    Train models for all 46 stocks
    """
    
    print("\n" + "="*90)
    print(f"STEP 3: TRAIN XGBOOST MODELS ({'OPTIMIZED' if optimize else 'BASELINE'})")
    print("="*90)
    print(f"Total stocks: {len(NIFTY_46_STOCKS)}")
    print(f"Optimization: {'ON (Optuna, 30 trials per stock)' if optimize else 'OFF (Default params)'}")
    print(f"Training data: Balanced (50-50 split)")
    print(f"Models directory: {MODELS_DIR}/")
    print("="*90 + "\n")
    
    all_results = []
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        
        results = train_single_stock(symbol, optimize=optimize)
        
        if results:
            all_results.append(results)
    
    # Summary statistics
    print("\n" + "="*90)
    print("TRAINING COMPLETE!")
    print("="*90)
    print(f"Successfully trained: {len(all_results)}/46 models\n")
    
    if len(all_results) > 0:
        avg_accuracy = np.mean([r['accuracy'] for r in all_results])
        avg_f1 = np.mean([r['f1_score'] for r in all_results])
        avg_recall_1 = np.mean([r['recall'] for r in all_results])
        avg_auc = np.mean([r['auc_roc'] for r in all_results])
        
        print("Average Performance:")
        print(f"  Accuracy:        {avg_accuracy:.3f}")
        print(f"  F1-Score:        {avg_f1:.3f}")
        print(f"  Recall (Class 1): {avg_recall_1:.3f}")
        print(f"  AUC-ROC:         {avg_auc:.3f}")
        
        # Best performing stocks
        best_f1 = sorted(all_results, key=lambda x: x['f1_score'], reverse=True)[:5]
        print("\nTop 5 Performers (by F1-Score):")
        for i, r in enumerate(best_f1, 1):
            print(f"  {i}. {r['symbol']}: F1={r['f1_score']:.3f}, Acc={r['accuracy']:.3f}, Recall={r['recall']:.3f}")
        
        # Save all results
        results_file = f"{RESULTS_DIR}/training_results_{'optimized' if optimize else 'baseline'}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved: {results_file}")
        print(f"Models saved: {MODELS_DIR}/*_model.pkl")
    
    print("="*90)
    
    return all_results


if __name__ == "__main__":
    import time
    
    print("\nStarting XGBoost training...")
    print("Training on BALANCED dataset (50-50 split)")
    print("\nChoice:")
    print("  1. BASELINE (Fast, ~10 minutes for 46 stocks)")
    print("  2. OPTIMIZED (Better, ~30 minutes for 46 stocks)\n")
    
    # For this run, use BASELINE (faster)
    # Change to optimize=True for better performance
    
    start_time = time.time()
    results = train_all_stocks(optimize=False)  # Set to True for optimization
    elapsed = time.time() - start_time
    
    print(f"\n✓✓✓ TRAINING COMPLETE in {elapsed/60:.1f} minutes! ✓✓✓")
    print("\nNext Steps:")
    print("  1. Step 4: SHAP Explainability (understand predictions)")
    print("  2. Step 5: Backtesting (test on real unseen data)")
    print("  3. Step 6: Deploy & Showcase\n")

# step5_backtest.py
"""
Step 5: Backtesting & Real-World Validation
- Tests models on ORIGINAL (unbalanced) data
- Simulates real trading with transaction costs
- Calculates portfolio returns and metrics
- Compares against buy-and-hold strategy
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from config import NIFTY_46_STOCKS, FEATURES_DIR, MODELS_DIR, RESULTS_DIR

# Backtest configuration
TRANSACTION_COST = 0.005  # 0.5% per trade (brokerage + taxes)
INITIAL_CAPITAL = 100000  # ₹1,00,000 starting capital
POSITION_SIZE = 0.02      # 2% of capital per trade (risk management)

# Create backtest directory
BACKTEST_DIR = f"{RESULTS_DIR}/backtest"
os.makedirs(BACKTEST_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_original_data(symbol):
    """
    Load ORIGINAL unbalanced data (real-world distribution)
    NOT the balanced SMOTE data used for training
    """
    
    file_path = f"{FEATURES_DIR}/{symbol}_features.csv"  # Original file
    
    if not os.path.exists(file_path):
        print(f"ERROR: Original data not found for {symbol}")
        return None
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df


def load_model(symbol):
    """Load trained model"""
    
    model_path = f"{MODELS_DIR}/{symbol}_model.pkl"
    
    if not os.path.exists(model_path):
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def backtest_single_stock(symbol, test_split=0.20):
    """
    Backtest a single stock on unseen data
    
    Args:
        symbol: Stock ticker
        test_split: Use last 20% of data as unseen test set
    
    Returns:
        Dict with backtest results
    """
    
    print(f"  Backtesting {symbol}...", end=' ', flush=True)
    
    # Load original (unbalanced) data
    df = load_original_data(symbol)
    model = load_model(symbol)
    
    if df is None or model is None:
        print("FAILED - Data/Model not found")
        return None
    
    # Split data (last 20% is unseen test data)
    split_idx = int(len(df) * (1 - test_split))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    # Features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Symbol', 'Target', 'Forward_Return']]
    
    X_test = df_test[feature_cols]
    y_test = df_test['Target'].values
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Add predictions to dataframe
    df_test['Prediction'] = y_pred
    df_test['Pred_Proba'] = y_pred_proba
    
    # Simulate trading
    trades = []
    capital = INITIAL_CAPITAL
    positions = []
    
    for i in range(len(df_test) - 5):  # -5 because we need 5 days forward return
        
        date = df_test.iloc[i]['Date']
        close_price = df_test.iloc[i]['Close']
        prediction = df_test.iloc[i]['Prediction']
        forward_return = df_test.iloc[i]['Forward_Return']
        actual_target = df_test.iloc[i]['Target']
        
        # Trading logic: Buy if model predicts Gain (1)
        if prediction == 1:
            
            # Calculate position size
            position_value = capital * POSITION_SIZE
            shares = position_value / close_price
            
            # Entry cost (with transaction cost)
            entry_cost = position_value * (1 + TRANSACTION_COST)
            
            # Exit after 5 days (or use forward_return)
            exit_value = position_value * (1 + forward_return / 100)
            exit_cost = exit_value * TRANSACTION_COST
            
            # Net profit/loss
            pnl = exit_value - entry_cost - exit_cost
            pnl_pct = (pnl / entry_cost) * 100
            
            # Update capital
            capital += pnl
            
            # Record trade
            trades.append({
                'date': date,
                'price': close_price,
                'prediction': prediction,
                'actual': actual_target,
                'forward_return': forward_return,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'capital': capital,
                'correct': (prediction == actual_target)
            })
    
    if len(trades) == 0:
        print("No trades executed")
        return None
    
    # Calculate metrics
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = (winning_trades / total_trades) * 100
    
    total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    avg_pnl_per_trade = trades_df['pnl'].mean()
    
    # Buy-and-hold comparison
    buy_hold_return = ((df_test.iloc[-1]['Close'] - df_test.iloc[0]['Close']) / df_test.iloc[0]['Close']) * 100
    
    # Accuracy on test set
    correct_predictions = trades_df['correct'].sum()
    accuracy = (correct_predictions / total_trades) * 100
    
    results = {
        'symbol': symbol,
        'test_samples': len(df_test),
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'accuracy': accuracy,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'outperformance': total_return - buy_hold_return,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'final_capital': capital,
        'max_capital': trades_df['capital'].max(),
        'min_capital': trades_df['capital'].min()
    }
    
    print(f"✓ Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Return: {total_return:.2f}% vs Buy-Hold: {buy_hold_return:.2f}%")
    
    return results, trades_df


def backtest_all_stocks(test_split=0.20):
    """
    Backtest all 46 stocks
    """
    
    print("\n" + "="*100)
    print("STEP 5: BACKTESTING ON REAL UNSEEN DATA")
    print("="*100)
    print(f"Test data: Last {test_split*100:.0f}% of each stock (unseen during training)")
    print(f"Trading simulation: {POSITION_SIZE*100:.0f}% position size, {TRANSACTION_COST*100:.2f}% transaction cost")
    print(f"Initial capital: ₹{INITIAL_CAPITAL:,}")
    print("="*100 + "\n")
    
    all_results = []
    all_trades = []
    
    for i, symbol in enumerate(NIFTY_46_STOCKS, 1):
        print(f"[{i:2d}/46] ", end='')
        
        result = backtest_single_stock(symbol, test_split=test_split)
        
        if result is not None:
            results, trades_df = result
            all_results.append(results)
            
            trades_df['symbol'] = symbol
            all_trades.append(trades_df)
    
    # Aggregate results
    print("\n" + "="*100)
    print("BACKTESTING COMPLETE!")
    print("="*100)
    print(f"Successfully backtested: {len(all_results)}/46 stocks\n")
    
    if len(all_results) > 0:
        
        # Overall statistics
        total_trades = sum([r['total_trades'] for r in all_results])
        total_winning = sum([r['winning_trades'] for r in all_results])
        overall_win_rate = (total_winning / total_trades) * 100
        
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_buy_hold = np.mean([r['buy_hold_return'] for r in all_results])
        avg_outperformance = avg_return - avg_buy_hold
        
        median_return = np.median([r['total_return'] for r in all_results])
        
        positive_returns = len([r for r in all_results if r['total_return'] > 0])
        beat_buy_hold = len([r for r in all_results if r['outperformance'] > 0])
        
        print("AGGREGATE PERFORMANCE:")
        print(f"  Total trades executed: {total_trades:,}")
        print(f"  Overall win rate: {overall_win_rate:.2f}%")
        print(f"  Stocks with positive returns: {positive_returns}/46 ({positive_returns/46*100:.1f}%)")
        print(f"  Stocks beating buy-and-hold: {beat_buy_hold}/46 ({beat_buy_hold/46*100:.1f}%)")
        print(f"\n  Average return: {avg_return:.2f}%")
        print(f"  Average buy-and-hold: {avg_buy_hold:.2f}%")
        print(f"  Average outperformance: {avg_outperformance:+.2f}%")
        print(f"  Median return: {median_return:.2f}%")
        
        # Top performers
        top_performers = sorted(all_results, key=lambda x: x['total_return'], reverse=True)[:10]
        print("\nTOP 10 PERFORMERS (by Return):")
        for i, r in enumerate(top_performers, 1):
            print(f"  {i:2d}. {r['symbol']:<15s} → Return: {r['total_return']:+7.2f}% | Win Rate: {r['win_rate']:5.1f}% | Trades: {r['total_trades']:3d}")
        
        # Worst performers
        worst_performers = sorted(all_results, key=lambda x: x['total_return'])[:5]
        print("\nWORST 5 PERFORMERS (by Return):")
        for i, r in enumerate(worst_performers, 1):
            print(f"  {i}. {r['symbol']:<15s} → Return: {r['total_return']:+7.2f}% | Win Rate: {r['win_rate']:5.1f}% | Trades: {r['total_trades']:3d}")
        
        # Save results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{BACKTEST_DIR}/backtest_results.csv", index=False)
        
        # Save all trades
        if len(all_trades) > 0:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
            all_trades_df.to_csv(f"{BACKTEST_DIR}/all_trades.csv", index=False)
        
        # Create visualizations
        create_backtest_visualizations(results_df)
        
        print(f"\nResults saved: {BACKTEST_DIR}/backtest_results.csv")
        print(f"All trades saved: {BACKTEST_DIR}/all_trades.csv")
    
    print("="*100)
    
    return all_results


def create_backtest_visualizations(results_df):
    """
    Create visualization plots for backtest results
    """
    
    # 1. Return Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results_df['total_return'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
    plt.xlabel('Return (%)', fontsize=12)
    plt.ylabel('Number of Stocks', fontsize=12)
    plt.title('Distribution of Returns', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 2. Model Return vs Buy-and-Hold
    plt.subplot(1, 2, 2)
    plt.scatter(results_df['buy_hold_return'], results_df['total_return'], 
                alpha=0.6, s=100, c='coral', edgecolors='black')
    
    # Diagonal line (equal performance)
    max_val = max(results_df['buy_hold_return'].max(), results_df['total_return'].max())
    min_val = min(results_df['buy_hold_return'].min(), results_df['total_return'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance')
    
    plt.xlabel('Buy-and-Hold Return (%)', fontsize=12)
    plt.ylabel('Model Return (%)', fontsize=12)
    plt.title('Model vs Buy-and-Hold Strategy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{BACKTEST_DIR}/return_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Win Rate vs Return
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['win_rate'], results_df['total_return'], 
                alpha=0.6, s=100, c='green', edgecolors='black')
    
    for i, row in results_df.iterrows():
        if row['total_return'] > 50 or row['total_return'] < -20:
            plt.annotate(row['symbol'], (row['win_rate'], row['total_return']), 
                        fontsize=8, alpha=0.7)
    
    plt.xlabel('Win Rate (%)', fontsize=12)
    plt.ylabel('Total Return (%)', fontsize=12)
    plt.title('Win Rate vs Return', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{BACKTEST_DIR}/win_rate_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved: {BACKTEST_DIR}/")


if __name__ == "__main__":
    print("\nStarting Backtesting on REAL UNSEEN DATA...")
    print("This will test if models actually make money in real-world conditions\n")
    
    # Run backtest (using last 20% of data as unseen test)
    results = backtest_all_stocks(test_split=0.20)
    
    print("\n✓✓✓ BACKTESTING COMPLETE! ✓✓✓")
    print("\nNext Steps:")
    print("  1. Review backtest results in results/backtest/")
    print("  2. Step 6: Deploy & Showcase (create dashboard)")
    print("  3. Upload to GitHub with impressive README\n")

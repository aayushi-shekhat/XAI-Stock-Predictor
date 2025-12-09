# config.py
"""
XAI Stock Predictor V2.0 - Project Configuration
Final list of 46 NIFTY stocks (curated for quality)
"""

# Final 46 stocks (successfully downloaded)
NIFTY_46_STOCKS = [
    # Banking & Finance (9)
    'HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN',
    'BAJFINANCE', 'INDUSINDBK', 'HDFCLIFE', 'SBILIFE',
    
    # IT (6)
    'TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM',
    
    # FMCG (6)
    'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'TATACONSUM',
    
    # Pharma (5)
    'SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP',
    
    # Auto (4)
    'MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'EICHERMOT',
    
    # Energy (5)
    'RELIANCE', 'ONGC', 'POWERGRID', 'NTPC', 'BPCL',
    
    # Infrastructure (3)
    'LT', 'ADANIENT', 'ADANIPORTS',
    
    # Metals (3)
    'HINDALCO', 'TATASTEEL', 'JSWSTEEL',
    
    # Telecom (1)
    'BHARTIARTL',
    
    # Others (4)
    'ASIANPAINT', 'GRASIM', 'COALINDIA', 'HEROMOTOCO'
]

# Excluded stocks (download issues)
EXCLUDED_STOCKS = ['BAJAJFINSV', 'M&M', 'ULTRACEMCO', 'TITAN']

# Sector mapping
SECTOR_MAP = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'BAJFINANCE', 'INDUSINDBK'],
    'Insurance': ['HDFCLIFE', 'SBILIFE'],
    'IT': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM'],
    'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'TATACONSUM'],
    'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'APOLLOHOSP'],
    'Auto': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'EICHERMOT'],
    'Energy': ['RELIANCE', 'ONGC', 'POWERGRID', 'NTPC', 'BPCL'],
    'Infrastructure': ['LT', 'ADANIENT', 'ADANIPORTS'],
    'Metals': ['HINDALCO', 'TATASTEEL', 'JSWSTEEL'],
    'Telecom': ['BHARTIARTL'],
    'Others': ['ASIANPAINT', 'GRASIM', 'COALINDIA', 'HEROMOTOCO']
}

# Project metadata
PROJECT_NAME = "XAI-Driven Stock Market Prediction"
VERSION = "2.0"
AUTHOR = "Aayushi Shekhat"
DATA_START_DATE = "2015-11-27"
DATA_END_DATE = "2025-11-24"
TOTAL_STOCKS = len(NIFTY_46_STOCKS)

# Data directories
RAW_DATA_DIR = "data/nse_nifty50"
PROCESSED_DATA_DIR = "data/processed"
FEATURES_DIR = "data/features"
MODELS_DIR = "models"
RESULTS_DIR = "results"
SHAP_DIR = "results/shap"

# Model parameters (to be optimized)
PREDICTION_HORIZON = 5  # days
THRESHOLD = 1.5  # % gain threshold
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature engineering
NUM_TECHNICAL_INDICATORS = 30
NUM_STATISTICAL_FEATURES = 10
NUM_MICROSTRUCTURE_FEATURES = 15
TOTAL_FEATURES = 70

# XGBoost hyperparameters (initial - will be tuned)
XGBOOST_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.03,
    'n_estimators': 800,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'scale_pos_weight': 1.94,
    'random_state': 42
}

if __name__ == "__main__":
    print(f"\n{PROJECT_NAME} v{VERSION}")
    print(f"="*60)
    print(f"Stocks: {TOTAL_STOCKS}")
    print(f"Sectors: {len(SECTOR_MAP)}")
    print(f"Period: {DATA_START_DATE} to {DATA_END_DATE}")
    print(f"Features: {TOTAL_FEATURES}")
    print(f"="*60)
    
    print("\nStock breakdown by sector:")
    for sector, stocks in SECTOR_MAP.items():
        print(f"  {sector:15s}: {len(stocks):2d} stocks")
    
    print(f"\nExcluded: {', '.join(EXCLUDED_STOCKS)}")

FEATURES_DIR = "data/features"

MODELS_DIR = "models"
RESULTS_DIR = "results"
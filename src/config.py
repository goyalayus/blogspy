# src/config.py

import pathlib

# --- Project Root ---
ROOT_DIR = pathlib.Path(__file__).parent.parent

# --- Data Directories ---
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Output Directories ---
OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = OUTPUT_DIR / "reports"

# --- Raw Data Files ---
CORPORATE_SITES_FILE = RAW_DATA_DIR / "ranked_domains.csv"
PERSONAL_SITES_FILE = RAW_DATA_DIR / "searchmysite_urls.csv"

# --- Processed Data Files (Cache & Temp) ---
ENRICHED_DATA_FILE = PROCESSED_DATA_DIR / "enriched_data.parquet"
TEMP_JSONL_FILE = PROCESSED_DATA_DIR / \
    "temp_enriched_data.jsonl"  # New temp file

# --- Label Mapping ---
LABEL_MAPPING = {"corporate_seo": 0, "personal_blog": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

# --- Model Training Parameters ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# --- LightGBM model parameters ---
# [MEMORY-FIX] Parameters have been tuned to be more memory-efficient.
# max_bin is the most critical parameter for memory usage.
# num_leaves also helps control model complexity and memory.
LGBM_PARAMS = {
    "objective":        "binary",
    "boosting_type":    "gbdt",
    "learning_rate":    0.08,
    "n_estimators":     400,
    "num_leaves":       21,     # REDUCED from 31
    "max_depth": -1,
    "max_bin":          63,     # DRASTICALLY REDUCED from 127
    "feature_fraction": 0.7,    # 70 % columns / tree
    "bagging_fraction": 0.7,    # 70 % rows / tree
    "bagging_freq":     1,
    "min_data_in_leaf": 20,
    "n_jobs":           1,      # single-threaded avoids OpenMP crashes
    "verbosity": -1,
}

# --- Data Enrichment ---
MAX_WORKERS = 40
REQUEST_TIMEOUT = 10
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

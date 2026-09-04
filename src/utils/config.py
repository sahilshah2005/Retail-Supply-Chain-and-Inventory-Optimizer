"""
Configuration constants for Retail Supply Chain and Inventory Optimizer.
"""
import os

# ── Project Metadata ─────────────────────────────────────────────────────────
APP_NAME = "Retail Supply Chain and Inventory Optimizer"
APP_VERSION = "2.0.0"
APP_ICON = "📦"

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
DEFAULT_DATA_FILE = os.path.join(DATA_DIR, "retail_sales.csv")

# ── Forecasting Defaults ─────────────────────────────────────────────────────
RANDOM_SEED = 42
DEFAULT_FORECAST_HORIZON = 4  # weeks
MIN_FORECAST_HORIZON = 2
MAX_FORECAST_HORIZON = 12
DATASET_FREQUENCY = "W"  # Weekly

# Feature engineering parameters
LAG_PERIODS = [1, 2, 4]
ROLLING_WINDOWS = [4]  # 4-week rolling stats
# Note: With ~18 data points per product, we keep rolling windows small
# to avoid losing too many rows to NaN

# ── Inventory Defaults ───────────────────────────────────────────────────────
DEFAULT_LEAD_TIME_WEEKS = 2
DEFAULT_SERVICE_LEVEL = 0.95
DEFAULT_ORDERING_COST = 500.0  # ₹ per order
DEFAULT_HOLDING_COST_PCT = 0.20  # 20% of unit cost per year
DEFAULT_CURRENT_INVENTORY = 100  # Default simulation value (units)

# ── Service Level Z-Scores ───────────────────────────────────────────────────
# Standard normal z-scores for common service levels
SERVICE_LEVEL_Z = {
    0.80: 0.842,
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}

# ── ABC Classification Thresholds ────────────────────────────────────────────
# Cumulative value percentage thresholds
# A: top items contributing to ~70% of total value
# B: next items contributing to ~20% of total value (70-90% cumulative)
# C: remaining items contributing to ~10% of total value (90-100% cumulative)
ABC_A_THRESHOLD = 0.70
ABC_B_THRESHOLD = 0.90
# C = everything above B threshold

# ── Model Comparison ─────────────────────────────────────────────────────────
VALIDATION_METRIC = "MAE"  # Primary metric for model selection (lower is better)

# ── Prediction Interval ──────────────────────────────────────────────────────
PREDICTION_INTERVAL_LEVEL = 0.90  # 90% prediction interval
# Z-score for 90% prediction interval (two-tailed: 5% each side)
PREDICTION_INTERVAL_Z = 1.645

# src/sp100rank/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"

# --- Time window ---
# Two distinct dates:
#
# TRAIN_START  : the start of all data ingest. Stable, never changes.
#
# FROZEN_DATA_END : the end-date used during model DEVELOPMENT. Walk-
#                   forward CV, hyperparameter tuning, SHAP, regime
#                   breakdown, and hold-out evaluation are all anchored
#                   to this date for reproducibility. The exact value
#                   here is what makes "Phase 5 numbers" replicable.
#                   DO NOT change this once the project's results are
#                   reported in the writeup — changing it would
#                   invalidate every reported IC number.
#
# Production scoring uses TODAY (not FROZEN_DATA_END). The download
# function defaults end=TODAY when end is None, allowing GitHub
# Actions runs to fetch fresh data without code changes.
TRAIN_START      = "2018-01-01"
FROZEN_DATA_END  = "2026-03-31"   # development snapshot — see above

SPX_TICKER = "^GSPC"

# -- Labels / Horizon --
HORIZON_DAYS = 20
EXECUTION_LAG = 1
EMBARGO_DAYS = HORIZON_DAYS

# -- Walk-forward CV --
# Sizes expressed in TRADING DAYS, not calendar units. The fold
# generator works on integer positions in the sorted-dates array;
# converting calendar lengths to trading-day counts here means the
# generator never has to think about holidays.
#
# Conversions used:
#   1 trading year ≈ 252 days
#   1 trading month ≈ 21 days
N_FOLDS              = 5
INITIAL_TRAIN_DAYS   = 756        # ~3 trading years
TEST_DAYS            = 126        # ~6 trading months


# -- Reproducibility --
RANDOM_SEED = 42



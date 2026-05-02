# src/sp100rank/config.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
DOCS_DIR = PROJECT_ROOT / "docs"

TRAIN_START = "2018-01-01"
DATA_END = "2026-03-31"

SPX_TICKER = "^GSPC"

# -- Labels / Horizon --
HORIZON_DAYS = 20
EXECUTION_LAG = 1
EMBARGO_DAYS = HORIZON_DAYS

# -- Walk-forward CV --
N_FOLDS = 5
INITIAL_TRAIN_YEARS = 3.0
TEST_PERIOD_MONTHS = 3


# -- Reproducibility --
RANDOM_SEED = 42



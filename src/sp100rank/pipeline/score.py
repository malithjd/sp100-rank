# src/sp100rank/pipeline/score.py
"""
Production scoring entry point.

Run via: `uv run python -m sp100rank.pipeline.score`

What it does:
  1. Refresh data from yfinance through today (via download_universe).
  2. Rebuild the clean panel.
  3. Build features for the latest available date.
  4. Load the latest model checkpoint.
  5. Score all stocks for the latest date.
  6. Write a ranked watchlist to outputs/scores/scores_YYYY-MM-DD.csv.

Designed to be:
  - Idempotent: running twice on the same day produces the same output.
  - Resumable: if data download fails on ticker N, retry only N.
  - Minimally fast: ~3-5 min total (data download dominates).
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sp100rank.config import (
    CHECKPOINT_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)
from sp100rank.data.clean import build_clean_panel
from sp100rank.data.ingest import download_universe
from sp100rank.features.build import build_features_and_labels


# Outputs go to a project-root directory. Created at runtime; commit-allowed.
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "scores"


def latest_checkpoint_path(model_name: str = "rf") -> Path:
    """Path to the model the production pipeline should score against.

    Production cadence:
      - retrain.yml workflow saves a NEW timestamped pickle and copies
        it to 'latest.pkl' in CHECKPOINTS_DIR/<model_name>/.
      - score.yml fetches 'rf_latest.pkl' from the latest GitHub
        Release into the same path.

    Local fallback for first-run-before-any-retrain:
      Use fold5.pkl from Phase 4 walk-forward training. This is the
      most-recent-data fold's RF model, an OK proxy until retraining
      runs.
    """
    latest = CHECKPOINT_DIR / model_name / "latest.pkl"
    if latest.exists():
        return latest
    fallback = CHECKPOINT_DIR / model_name / "fold5.pkl"
    if not fallback.exists():
        raise FileNotFoundError(
            f"Neither {latest} nor {fallback} exists. "
            f"Run retrain.py at least once before scoring, "
            f"or run the Phase 4 training to produce fold5.pkl."
        )
    return fallback


def score_latest(model_name: str = "rf") -> pd.DataFrame:
    """Score all 100 stocks for the latest date with valid features.

    Returns a DataFrame with one row per stock:
      ticker | predicted_rank | watchlist_signal | score_date

    Sorted descending by predicted_rank.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting score run.")

    # 1. Refresh data through today.
    print("Refreshing data...")
    download_universe()

    # 2. Rebuild clean panel.
    print("Rebuilding clean panel...")
    panel = build_clean_panel()

    # 3. Compute features.
    print("Computing features...")
    X_all, _ = build_features_and_labels()

    # 4. Restrict to selected feature set.
    selected = json.loads(
        (PROCESSED_DATA_DIR / "selected_features.json").read_text()
    )["selected_features"]
    X_all = X_all[selected]

    # 5. Find the latest date with feature coverage.
    # We need at least 95/100 stocks with all features non-NaN.
    valid_per_date = X_all.notna().all(axis=1).groupby(level="date").sum()
    eligible_dates = valid_per_date[valid_per_date >= 95].index
    if len(eligible_dates) == 0:
        raise RuntimeError(
            "No date has features for at least 95 stocks. "
            "Check data ingest and feature warmup periods."
        )
    score_date = eligible_dates.max()
    print(f"Scoring on: {score_date.date()}")

    # 6. Load model.
    ckpt_path = latest_checkpoint_path(model_name)
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        model = pickle.load(f)

    # 7. Predict.
    X_score = X_all.xs(score_date, level="date").dropna()
    preds = model.predict(X_score)

    # 8. Format output.
    df = pd.DataFrame({
        "ticker": X_score.index,
        "predicted_rank": preds,
    })
    df = df.sort_values("predicted_rank", ascending=False).reset_index(drop=True)

    # Watchlist tag:
    #   Top 20% (rank 0.80-1.00) → BUY
    #   Bottom 20% (rank 0.00-0.20) → AVOID
    #   Middle 60% → HOLD
    n = len(df)
    df["watchlist_signal"] = "HOLD"
    df.loc[df.index < n * 0.20, "watchlist_signal"] = "BUY"
    df.loc[df.index >= n * 0.80, "watchlist_signal"] = "AVOID"
    df["score_date"] = score_date.date()

    return df


def main() -> int:
    """Shell entry point. Returns 0 on success, 1 on any error."""
    try:
        scores = score_latest()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        score_date = scores["score_date"].iloc[0]
        out_path = OUTPUTS_DIR / f"scores_{score_date}.csv"
        scores.to_csv(out_path, index=False)
        print(f"Wrote {len(scores)} scores to {out_path}")
        print("\nTop 5 BUY signals:")
        print(
            scores[scores["watchlist_signal"] == "BUY"]
            .head(5)
            .to_string(index=False)
        )
        print("\nTop 5 AVOID signals:")
        print(
            scores[scores["watchlist_signal"] == "AVOID"]
            .head(5)
            .to_string(index=False)
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
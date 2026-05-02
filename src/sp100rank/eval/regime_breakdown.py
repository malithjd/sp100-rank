# src/sp100rank/eval/regime_breakdown.py
"""
Compute IC stratified by calendar year and by regime tags.

For each model, we have predictions across all 5 test folds (Feb 2021
to Nov 2023). We want to slice the IC by:
  1. Calendar year — clean reporting unit, easy to interpret.
  2. Trend regime (bull/bear) — does the model work in both?
  3. Volatility regime (high/low) — does it work in stress periods?
  4. Combined trend × vol cells — the 4-quadrant picture.

We use per-DATE IC (not pooled) and aggregate by mean within each
slice. Reasoning identical to the headline IC analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sp100rank.config import PROCESSED_DATA_DIR
from sp100rank.data.clean import load_clean_panel
from sp100rank.eval.metrics import daily_rank_ic
from sp100rank.eval.regimes import tag_regimes
from sp100rank.eval.walkforward import walk_forward_folds
from sp100rank.features.build import build_features_and_labels
from sp100rank.models.registry import all_model_names


def compute_per_date_ic_all_models() -> pd.DataFrame:
    """Compute per-date IC for every (model, fold) combination.

    Returns a DataFrame with columns:
      [date, model, fold_id, ic, year, trend_regime, vol_regime]

    Long-format. Easy to groupby on whatever stratification you want.
    """
    panel = load_clean_panel()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    folds = list(walk_forward_folds(all_dates))
    regimes = tag_regimes(panel)

    rows = []
    for model_name in all_model_names():
        pred_dir = PROCESSED_DATA_DIR / "predictions" / model_name
        for f in folds:
            preds = pd.read_parquet(pred_dir / f"fold{f.fold_id}.parquet")["pred"]
            _, y_test = build_features_and_labels(start=f.test_start, end=f.test_end)
            ic = daily_rank_ic(preds, y_test).dropna()
            for date_idx, val in ic.items():
                rows.append({
                    "date": date_idx,
                    "model": model_name,
                    "fold_id": f.fold_id,
                    "ic": val,
                })

    df = pd.DataFrame(rows)
    df["year"] = df["date"].dt.year
    df = df.merge(regimes, left_on="date", right_index=True, how="left")
    return df


def summarize_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Mean IC per (model, year). Pivot to model × year matrix."""
    return (
        df.groupby(["model", "year"])["ic"]
          .mean()
          .unstack("year")
          .round(4)
    )


def summarize_by_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Mean IC per (model, trend_regime). Bull vs bear."""
    return (
        df.groupby(["model", "trend_regime"])["ic"]
          .agg(["mean", "count"])
          .round(4)
    )


def summarize_by_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Mean IC per (model, vol_regime). High-vol vs low-vol."""
    return (
        df.groupby(["model", "vol_regime"])["ic"]
          .agg(["mean", "count"])
          .round(4)
    )


def summarize_by_trend_x_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Mean IC per (model, trend × vol). The 4-cell picture."""
    cell_means = (
        df.groupby(["model", "trend_regime", "vol_regime"])["ic"]
          .mean()
          .unstack(["trend_regime", "vol_regime"])
          .round(4)
    )
    return cell_means
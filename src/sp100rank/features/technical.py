# src/sp100rank/features/technical.py
"""
Technical features for cross-sectional return prediction.

Each feature is implemented as a function that takes a single ticker's
time series and returns a Series. The top-level `compute_all_features`
applies them all per-ticker via groupby.

CAUSAL CONTRACT — read this before adding features:
  Every feature value at date t depends ONLY on data through close
  of t. No `.shift(-N)`, no centered rolling windows, no `.expanding()`
  reaching past t. The no-lookahead test in tests/ enforces this.

  If you need future data, you're writing a LABEL, not a feature.
  Labels go in labels.py.
"""

import pandas as pd
import numpy as np


def compute_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply all features per ticker. Returns one row per (date, ticker).

    Currently empty — features are added in subsequent commits, one
    at a time, each verified by the no-lookahead test.
    """
    # Sort once at the top so groupby + rolling don't have to think.
    prices = prices.sort_index()

    # Per-ticker feature application. group_keys=False keeps the
    # output's index aligned with the input's (date, ticker).
    out = (
        prices.groupby(level="ticker", group_keys=False)
              .apply(_features_for_one_ticker, include_groups=False)
    )
    return out


def _features_for_one_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features for a single ticker's time series.

    Stub returns an empty DataFrame indexed like the input. Feature
    columns get added in subsequent commits. The empty-output return
    path lets the no-lookahead test run end-to-end before any
    features exist — verifying the test infrastructure itself.
    """
    return pd.DataFrame(index=df.index)
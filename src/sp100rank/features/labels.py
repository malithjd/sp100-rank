# src/sp100rank/features/labels.py
"""
Cross-sectional forward-return rank label.

Distinct from features because labels USE FUTURE DATA. Keeping them
in a separate module makes the boundary explicit:
  - technical.py  : strictly causal — never .shift(-N), never future
  - labels.py     : the only place future shifts are allowed

Per ADR-003 (realistic execution alignment), the label uses a 1-day
execution lag: we observe close of t, fill at close of t+1, exit at
close of t+1+horizon. The forward return is close_{t+1+h}/close_{t+1}-1.

This is the prediction target for all four models.
"""

from __future__ import annotations

import pandas as pd

from sp100rank.config import HORIZON_DAYS, EXECUTION_LAG
from sp100rank.data.universe import is_index


def forward_return(
    prices: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    execution_lag: int = EXECUTION_LAG,
) -> pd.Series:
    close = prices["adj_close"].sort_index()

    g = close.groupby(level="ticker")


    fill_close = g.shift(-execution_lag)
    
    exit_close = g.shift(-(execution_lag + horizon))

    fwd_ret = (exit_close / fill_close) - 1.0
    fwd_ret.name = "fwd_return"
    return fwd_ret


def cross_sectional_rank_label(
    prices: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    execution_lag: int = EXECUTION_LAG,
) -> pd.Series:
    
    fwd_ret = forward_return(prices, horizon, execution_lag)

    tickers = fwd_ret.index.get_level_values("ticker")
    is_equity = ~tickers.map(is_index)
    fwd_ret_equity = fwd_ret[is_equity]

    # Rank per date. groupby(level='date') groups all tickers on the
    # same trading day; .rank(pct=True) normalizes ranks to [0, 1].
    label = fwd_ret_equity.groupby(level="date").rank(
        method="average",
        pct=True,
    )
    label.name = "y_rank"
    return label
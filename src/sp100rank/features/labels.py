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
    """Realized forward return per (date, ticker), using close-to-close.

    Timeline:
        t            : observe close (used by features)
        t + lag      : fill trade at close
        t + lag + h  : close trade at close

        forward_return_t = close_{t + lag + h} / close_{t + lag} - 1

    With default lag=1 and h=20:
        forward_return_t = close_{t+21} / close_{t+1} - 1

    Why 1-day lag (vs 0-day same-bar trade):
        Same-bar trading would compute close_{t+h}/close_t, implicitly
        executing AT today's close using TODAY's close as a feature.
        That's same-bar leakage. The 1-day lag enforces "observe
        today, trade tomorrow" — the realistic case.

    Parameters
    ----------
    prices : MultiIndex (date, ticker) panel with at least 'adj_close'.
    horizon : trading days held. 20 ≈ one calendar month.
    execution_lag : trading days between observation and fill. 1 = T+1.

    Returns
    -------
    Series indexed by (date, ticker), values are realized returns.
    Last (lag + horizon) rows per ticker are NaN — those positions
    can't be evaluated yet because the future hasn't happened.
    """
    close = prices["adj_close"].sort_index()

    # shift(-N) within each ticker brings close_{t+N} into the row
    # indexed by t. Negative shifts are forward-looking — that's
    # exactly what labels need. groupby(ticker) is critical: without
    # it, the shift would cross ticker boundaries and produce
    # nonsense (e.g., AAPL's last day "shifting in" the first day of
    # ABT, the alphabetically next ticker).
    g = close.groupby(level="ticker")

    # close_{t+lag} — what we'd buy at
    fill_close = g.shift(-execution_lag)
    # close_{t+lag+horizon} — what we'd sell at
    exit_close = g.shift(-(execution_lag + horizon))

    fwd_ret = (exit_close / fill_close) - 1.0
    fwd_ret.name = "fwd_return"
    return fwd_ret


def cross_sectional_rank_label(
    prices: pd.DataFrame,
    horizon: int = HORIZON_DAYS,
    execution_lag: int = EXECUTION_LAG,
) -> pd.Series:
    """Cross-sectional percentile rank of forward return, per date.

    Steps:
        1. Compute forward_return per (date, ticker).
        2. EXCLUDE the index ticker (^GSPC) — it's a proxy, not a
           prediction target.
        3. Per date, rank stocks by forward_return; normalize to [0, 1].

    Returns a Series indexed by (date, ticker), values in [0, 1].
    NaN rows where forward window extends past the data.

    Why pct=True: divides each rank by the count of valid (non-NaN)
    ranks within the date. Result: [0, 1] regardless of how many
    stocks have valid forward returns on that date. Robust to
    missing names.

    Why method='average': handles ties symmetrically. If two stocks
    have identical forward returns, both get the same average rank
    rather than arbitrary tie-breaking.
    """
    fwd_ret = forward_return(prices, horizon, execution_lag)

    # Exclude the market index from the prediction cross-section.
    # ^GSPC is a feature input (for beta) but not a target.
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
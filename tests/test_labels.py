# tests/test_labels.py
"""
Tests for the cross-sectional rank label.

Two contracts to verify:
  1. The label uses the CORRECT future window (close_{t+1+h}/close_{t+1}-1).
  2. The label is in [0, 1] and uniformly distributed within each date.

Both are easy to mis-implement and silent if wrong.
"""

import numpy as np
import pandas as pd
import pytest

from sp100rank.features.labels import forward_return, cross_sectional_rank_label


def test_forward_return_alignment(synthetic_panel):
    """
    Manual sanity check: pick one ticker, one date, manually compute
    the forward return, verify it matches the label module's output.

    Using close-to-close logic with lag=1, horizon=20:
        on date t, label_t = close_{t+21} / close_{t+1} - 1
    """
    panel = synthetic_panel
    ticker = "AAA"
    close = panel.xs(ticker, level="ticker")["adj_close"]

    # Pick a date with at least 21 trading days remaining.
    # synthetic_panel has 200 dates, so date index 50 leaves 150 ahead.
    t_idx = 50
    t_date = close.index[t_idx]

    # Manual computation: observe at t, fill at t+1, exit at t+21.
    expected_fill = close.iloc[t_idx + 1]
    expected_exit = close.iloc[t_idx + 1 + 20]
    expected_return = expected_exit / expected_fill - 1.0

    # Module computation
    fr = forward_return(panel, horizon=20, execution_lag=1)
    actual_return = fr.loc[(t_date, ticker)]

    assert np.isclose(actual_return, expected_return, rtol=1e-12), (
        f"Mismatch at {t_date} {ticker}: "
        f"expected {expected_return}, got {actual_return}"
    )


def test_label_range_and_uniformity(synthetic_panel):
    """
    Cross-sectional ranks must be in [0, 1]. Within each date, the
    distribution of ranks should be uniform — that's the definition
    of a percentile rank.

    With 5 tickers in the fixture, ranks per date should be exactly
    {0.2, 0.4, 0.6, 0.8, 1.0} (using pct=True with method='average').
    """
    panel = synthetic_panel
    label = cross_sectional_rank_label(panel, horizon=20, execution_lag=1)
    label = label.dropna()

    # Range check
    assert label.min() > 0.0, f"Min rank should be > 0, got {label.min()}"
    assert label.max() <= 1.0, f"Max rank should be ≤ 1, got {label.max()}"

    # Per-date uniformity: for any date, the set of ranks should be
    # close to {0.2, 0.4, 0.6, 0.8, 1.0} (5 tickers, no ties expected
    # because synthetic returns are random floats).
    sample_date = label.index.get_level_values("date").unique()[10]
    ranks_on_date = label.xs(sample_date, level="date").sort_values().values
    expected = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    np.testing.assert_allclose(
        ranks_on_date, expected, atol=1e-9,
        err_msg=f"Ranks on {sample_date} should be uniform: got {ranks_on_date}"
    )


def test_label_excludes_no_future_rows(synthetic_panel):
    """
    The last (lag + horizon) rows of each ticker should have NaN
    labels — there isn't enough future data to compute them.

    With lag=1, horizon=20, that's the last 21 rows per ticker.
    """
    panel = synthetic_panel
    label = cross_sectional_rank_label(panel, horizon=20, execution_lag=1)

    last_dates = panel.index.get_level_values("date").unique()[-21:]
    for d in last_dates:
        try:
            on_date = label.xs(d, level="date")
            # All NaN expected
            assert on_date.isna().all(), (
                f"Date {d} should have all NaN labels (insufficient future), "
                f"got {on_date.dropna().values}"
            )
        except KeyError:
            # Date not in label index at all — that's also acceptable
            # (groupby+rank may drop fully-NaN dates).
            pass
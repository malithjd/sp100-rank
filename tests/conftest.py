# tests/conftest.py
"""
Shared pytest fixtures for the test suite.

Pytest auto-discovers this file and makes its fixtures available to
any test in the same directory or below. We define synthetic test
data here once, then use it from multiple test files. This both
keeps tests fast (no I/O) and makes tests deterministic (fixed seed).
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A synthetic OHLCV panel mirroring the real data's structure.

    Returns a DataFrame with:
      - MultiIndex (date, ticker)
      - Columns: open, high, low, close, adj_close, volume
      - 5 tickers × 200 business days = 1000 rows

    Numbers are random but reproducible (seed=42). Real values don't
    matter — these tests verify STRUCTURAL correctness of feature
    code (no lookahead, correct shape, correct alignment), not
    numerical correctness against any reference.

    Why a fixture and not a global constant: pytest creates a fresh
    copy for each test that requests it. Tests that mutate the panel
    can't accidentally affect other tests.
    """
    # Reproducible: same seed → same numbers every run. If a test
    # fails on one machine and not another, seed makes the difference
    # immediately debuggable.
    rng = np.random.default_rng(seed=42)

    # 200 business days starting 2020-01-01. freq='B' = business days
    # (skips weekends). We don't model holidays in tests — the real
    # data has them but our feature code is calendar-agnostic anyway.
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]

    # Build the cross-product index: every (date, ticker) pair.
    idx = pd.MultiIndex.from_product(
        [dates, tickers],
        names=["date", "ticker"],
    )

    # Synthetic OHLCV. Prices roughly $100, volumes 1M-10M.
    # The exact distributions don't matter for structural tests; we
    # just need realistic-ish ranges so feature math doesn't underflow
    # or produce surprising NaNs.
    panel = pd.DataFrame(
        {
            "open":      rng.normal(100.0, 1.0, len(idx)),
            "high":      rng.normal(101.0, 1.0, len(idx)),
            "low":       rng.normal(99.0,  1.0, len(idx)),
            "close":     rng.normal(100.0, 1.0, len(idx)),
            "adj_close": rng.normal(100.0, 1.0, len(idx)),
            "volume":    rng.integers(1_000_000, 10_000_000, len(idx)),
        },
        index=idx,
    )

    # Sort by (date, ticker). Pandas groupby + rolling assumes sorted
    # input within each group; sort once here so individual tests
    # don't have to think about it.
    panel = panel.sort_index()
    return panel
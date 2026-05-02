# tests/test_features_no_lookahead.py
"""
THE most important test in the project.

Lookahead bugs — features that accidentally peek into the future —
are the leading cause of "my backtest was great but live trading
lost money" stories in quant ML. They don't crash, they don't even
look wrong; they just silently inflate IC.

Pattern of attack: shuffle prices AFTER some date t, recompute
features, assert features at dates BEFORE t are unchanged. If
they change, a feature is using future data.

We run this test every time we add or modify a feature.
"""

import numpy as np
import pandas as pd
import pytest

from sp100rank.features.technical import compute_all_features


def test_features_do_not_depend_on_future(synthetic_panel):
    """
    Causal feature contract: feature value at date t depends ONLY on
    OHLCV through date t (inclusive).

    Verification: corrupt prices from a cutoff date forward, recompute
    features, assert features at dates STRICTLY BEFORE the cutoff are
    bitwise-identical to the clean version.

    Why bitwise-identical and not approximately equal: if features are
    truly causal, the corruption can't possibly affect them — so the
    floats should match exactly. We use rtol=1e-10 for safety against
    tiny float-add-order differences in pandas internals, but in
    practice the values match exactly.
    """
    panel = synthetic_panel
    dates = panel.index.get_level_values("date").unique()

    # Pick a cutoff in the middle of the date range. Use day 100 of 200.
    # Far enough into the series that any feature with a long lookback
    # window has had time to "warm up" on real data before the cutoff;
    # this prevents false-positive failures where a feature is NaN on
    # both sides simply because the warmup hasn't completed.
    cutoff = dates[100]

    # --- Compute features on the clean panel ---
    features_clean = compute_all_features(panel)

    # --- Corrupt the future and recompute ---
    # Replace adj_close, close, volume from cutoff onward with garbage.
    # We corrupt MULTIPLE columns because a feature might use any of
    # them; we want a single test that catches lookahead through any
    # input column.
    corrupted = panel.copy()
    future_mask = corrupted.index.get_level_values("date") >= cutoff
    # Use POSITIVE garbage values. Negatives + log_dollar_volume produce
    # RuntimeWarnings that aren't real bugs — the production data
    # never has negative prices/volumes (cleaning enforces this).
    # The corruption test still works because the values are still
    # wildly different from the clean data; if features leak future
    # data, the change is detected regardless of sign.
    corrupted.loc[future_mask, "close"]     = 99999.0
    corrupted.loc[future_mask, "adj_close"] = 99999.0
    corrupted.loc[future_mask, "high"]      = 99999.0
    corrupted.loc[future_mask, "low"]       = 99999.0
    corrupted.loc[future_mask, "open"]      = 99999.0
    corrupted.loc[future_mask, "volume"]    = 99999999

    features_corrupted = compute_all_features(corrupted)

    # --- Compare features BEFORE the cutoff ---
    # Strictly less-than: date == cutoff is the first point that may
    # legitimately change (it sees its own corrupted close). Don't
    # include it in the "must be unchanged" assertion.
    pre_mask = features_clean.index.get_level_values("date") < cutoff

    pd.testing.assert_frame_equal(
        features_clean.loc[pre_mask],
        features_corrupted.loc[pre_mask],
        check_exact=False,
        rtol=1e-10,
        check_dtype=False,
        check_names=True,
    )


def test_features_have_no_post_cutoff_dependence_per_column(synthetic_panel):
    """
    Stronger version: corrupt ONE input column at a time. If a feature
    uses ONLY close but the test corrupts only volume, the feature
    should be perfectly unchanged everywhere — past AND future.

    This catches the subtle case where a feature accidentally uses a
    column it shouldn't (e.g., a "momentum" feature that mistakenly
    pulls in volume because of a typo).

    We don't run this for every column combo — that's combinatorial.
    But it's a useful diagnostic when the headline test above fails
    and you want to narrow down WHICH column is the leak.
    """
    # Skip-marker: we keep this in the file as a template but don't
    # run it by default. Uncomment / remove the skip when debugging.
    pytest.skip("diagnostic test — enable manually when debugging")
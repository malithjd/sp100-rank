# src/sp100rank/eval/walkforward.py
"""
Walk-forward cross-validation fold generator.

Produces (train_start, train_end, test_start, test_end) tuples that
walk forward through time. Expanding training window: each fold's
train starts at the very first date and grows to absorb the previous
fold's test period.

Per ADR-002, this is our PRIMARY evaluation scheme. Per ADR-003, the
embargo equals the label horizon (20 trading days).

Why expanding (not rolling) windows:
  1. Mirrors deployment — in production we accumulate data forever,
     never forget the past, so fold structure matches.
  2. More training data in later folds helps stabilize learning;
     rolling windows can starve the model in early folds.

Why an embargo:
  Labels at date `train_end` use prices through `train_end + lag + h`.
  If we tested on `train_end + 1`, that test row's features-and-label
  would share information with rows already in train. The embargo
  enforces strict separation. See López de Prado (2018) ch. 7 for
  the original argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from sp100rank.config import (
    N_FOLDS,
    EMBARGO_DAYS,
    INITIAL_TRAIN_DAYS,
    TEST_DAYS,
)


@dataclass
class Fold:
    """Walk-forward fold boundaries.

    All attributes are pandas Timestamps. Inclusive on both ends.
    Use these to slice your features/labels DataFrames:
        train = df.loc[fold.train_start : fold.train_end]
        test  = df.loc[fold.test_start  : fold.test_end ]
    """
    fold_id: int
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp


def walk_forward_folds(
    dates: pd.DatetimeIndex,
    n_folds: int = N_FOLDS,
    embargo_days: int = EMBARGO_DAYS,
    initial_train_days: int = INITIAL_TRAIN_DAYS,
    test_days: int = TEST_DAYS,
) -> Iterator[Fold]:
    """Generate walk-forward folds with embargo.

    Parameters
    ----------
    dates : sorted DatetimeIndex
        All trading dates in the dataset. We use INTEGER POSITIONS in
        this array — never calendar offsets — so weekends and
        holidays don't introduce slippage.
    n_folds : int
        Number of folds to produce. With 8 years of data and the
        defaults below, 5 is the sweet spot.
    embargo_days : int
        Trading-day gap between train_end and test_start. Set this
        to your label horizon (20 here, matching HORIZON_DAYS).
    initial_train_days : int
        Length of fold 1's training window. 756 ≈ 3 years.
    test_days : int
        Length of EACH fold's test period. 126 ≈ 6 months.

    Yields
    ------
    Fold dataclasses, in chronological order.

    Implementation note — why integer indices vs date arithmetic:
        pd.DateOffset(months=6) is a CALENDAR offset; on a panel with
        ~252 trading days/year it produces inconsistent fold sizes
        (some folds get 122 days, some 130). Integer positions on the
        sorted-dates array give exact, reproducible fold sizes.
    """
    # Coerce + sort. Defensive against callers passing unsorted dates;
    # cheap and prevents subtle off-by-one errors downstream.
    dates = pd.DatetimeIndex(sorted(set(dates)))
    n = len(dates)

    # First fold's boundaries.
    train_end_idx   = initial_train_days - 1
    test_start_idx  = train_end_idx + 1 + embargo_days
    test_end_idx    = test_start_idx + test_days - 1

    for k in range(n_folds):
        # Bounds check — if the test period would run off the end of
        # the data, stop yielding rather than truncate silently.
        # Better to surface "you asked for 5 folds but only 4 fit"
        # than to give a fold with a 30-day test window when the
        # caller specified 6 months.
        if test_end_idx >= n:
            # We could raise here. Yielding only what fits is more
            # forgiving and matches sklearn's CV behavior.
            return

        yield Fold(
            fold_id     = k + 1,
            train_start = dates[0],                    # expanding
            train_end   = dates[train_end_idx],
            test_start  = dates[test_start_idx],
            test_end    = dates[test_end_idx],
        )

        # Slide forward: training absorbs the previous test period
        # (no embargo on training side — only between train and test).
        # Next test period begins one embargo past the new train_end.
        train_end_idx  = test_end_idx
        test_start_idx = train_end_idx + 1 + embargo_days
        test_end_idx   = test_start_idx + test_days - 1
# tests/test_walkforward.py
"""
Tests for walk-forward fold generation.

Critical invariants:
  1. Embargo: test_start - train_end >= embargo_days (in TRADING days).
  2. Order: each fold is chronologically after the previous.
  3. No overlap between train and test of the same fold.
  4. Test sets across folds are non-overlapping.

Bugs in any of these silently invalidate evaluation. Fast tests
(< 1 second total) provide cheap insurance.
"""

import pandas as pd
import pytest

from sp100rank.eval.walkforward import walk_forward_folds, Fold


@pytest.fixture
def long_date_range():
    """Synthetic ~8 years of business days, matching real data shape."""
    return pd.bdate_range("2018-01-01", "2026-03-31")


def test_fold_count(long_date_range):
    """With our defaults on 8 years, we should get exactly 5 folds."""
    folds = list(walk_forward_folds(long_date_range, n_folds=5))
    assert len(folds) == 5, f"Expected 5 folds, got {len(folds)}"


def test_train_ends_before_test_starts(long_date_range):
    """train_end < test_start, with at least embargo_days gap."""
    embargo = 20
    folds = list(walk_forward_folds(long_date_range, embargo_days=embargo))

    for f in folds:
        # Find integer positions of train_end and test_start in the
        # date series. The gap should be exactly embargo + 1 trading
        # days (test_start = train_end + 1 + embargo, both inclusive).
        train_end_idx = long_date_range.get_loc(f.train_end)
        test_start_idx = long_date_range.get_loc(f.test_start)
        gap = test_start_idx - train_end_idx
        assert gap >= embargo + 1, (
            f"Fold {f.fold_id}: gap of {gap} trading days < embargo {embargo} + 1"
        )


def test_train_test_no_overlap(long_date_range):
    """Within a fold, train and test must not share any dates."""
    for f in walk_forward_folds(long_date_range):
        assert f.train_end < f.test_start, (
            f"Fold {f.fold_id}: train_end={f.train_end} >= test_start={f.test_start}"
        )


def test_test_sets_non_overlapping(long_date_range):
    """No two folds' test windows may overlap.

    Otherwise we'd be averaging IC over the same dates multiple times,
    inflating apparent stability via repeated counting.
    """
    folds = list(walk_forward_folds(long_date_range))
    for i, f1 in enumerate(folds):
        for f2 in folds[i + 1:]:
            # f2 comes later. Its test_start must be after f1's
            # test_end.
            assert f2.test_start > f1.test_end, (
                f"Folds {f1.fold_id} and {f2.fold_id} overlap: "
                f"f1 test ends {f1.test_end}, f2 test starts {f2.test_start}"
            )


def test_chronological_order(long_date_range):
    """Folds yielded in ascending fold_id, with monotonically advancing
    test windows."""
    folds = list(walk_forward_folds(long_date_range))
    for i in range(len(folds) - 1):
        assert folds[i].fold_id < folds[i + 1].fold_id
        assert folds[i].test_start < folds[i + 1].test_start


def test_expanding_train_window(long_date_range):
    """Each fold's train_start must equal the first date (expanding
    window). train_end advances with each fold."""
    first_date = long_date_range[0]
    folds = list(walk_forward_folds(long_date_range))

    for f in folds:
        assert f.train_start == first_date, (
            f"Fold {f.fold_id}: expanding window should start at {first_date}, "
            f"got {f.train_start}"
        )

    # train_ends advance monotonically
    for i in range(len(folds) - 1):
        assert folds[i].train_end < folds[i + 1].train_end


def test_handles_insufficient_data():
    """If the date range is too short for the requested folds, yield
    fewer folds rather than error."""
    # 2 years of data, 5 folds requested with 3-year initial train.
    short = pd.bdate_range("2018-01-01", "2020-01-01")
    folds = list(walk_forward_folds(
        short, n_folds=5, initial_train_days=756, test_days=126
    ))
    # Initial train alone (756 days) > 2 years of bdates (~520) — no
    # folds should fit at all.
    assert len(folds) == 0, (
        f"Insufficient data should yield 0 folds, got {len(folds)}"
    )
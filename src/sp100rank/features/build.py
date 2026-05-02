# src/sp100rank/features/build.py
"""
Assemble feature matrix and label series, sliced to a date window.

Pipeline-orchestration helper used by both feature selection and
model training. Single function — keep it simple.
"""

from __future__ import annotations

import pandas as pd

from sp100rank.data.clean import load_clean_panel
from sp100rank.features.technical import compute_all_features
from sp100rank.features.labels import cross_sectional_rank_label


def build_features_and_labels(
    start: pd.Timestamp | None = None,
    end:   pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute features + label, optionally sliced to a date window.

    Parameters
    ----------
    start, end : optional inclusive date bounds. If None, the full
        panel is returned. Used to slice to a fold's train period.

    Returns
    -------
    (X, y) : tuple of (DataFrame with feature columns, Series with
    label). Both indexed by (date, ticker) and aligned. Rows with
    any NaN are NOT dropped here — caller decides.
    """
    panel = load_clean_panel()
    X = compute_all_features(panel)
    y = cross_sectional_rank_label(panel)

    if start is not None or end is not None:
        # Slice both X and y to the date window. Inclusive on both sides.
        # Use IndexSlice for clarity on MultiIndex slicing.
        idx = pd.IndexSlice
        X = X.loc[idx[start:end, :], :]
        y = y.loc[idx[start:end, :]]

    # Align: y is missing for SPX (we excluded it). Inner-join so X
    # only contains rows that have labels too.
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    return X, y
# src/sp100rank/features/select.py
"""
Feature selection via gain importance from a single LightGBM fit
on Fold 1 training data.

Per ADR-004:
  - Fit LightGBM ONCE on Fold 1 train.
  - Rank features by 'gain' importance.
  - Keep the top 8 of 12.
  - Use this list across ALL models and ALL folds.

Why this approach is defensible:
  - No information leak from later folds (selection touches only
    Fold 1 train).
  - Fold 1 IC is slightly optimistic (acknowledged in writeup);
    Folds 2-5 are unbiased.
  - LightGBM is fast and handles non-linear interactions, so the
    selected set is appropriate for the other tree models too.
    For the linear baseline, the same set is used — slight handicap
    for the linear model, but consistency across models matters more.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from sp100rank.config import RANDOM_SEED


def select_features_by_gain(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    keep_top_k: int = 8,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], pd.Series]:
    """Fit LightGBM, rank features by total gain, return top-k.

    Parameters
    ----------
    X_train : DataFrame with feature columns. (date, ticker)-indexed.
    y_train : Series with label. Same index as X_train.
    keep_top_k : how many features to retain. Per ADR-004, 8.
    seed : reproducibility.

    Returns
    -------
    (selected_features, importance_series) :
      - selected_features: list of column names, ordered by gain (most first).
      - importance_series: full feature ranking (all 12), indexed by name.
        Useful for the writeup table.
    """
    aligned = pd.concat(
        {"y": y_train, **{c: X_train[c] for c in X_train.columns}},
        axis=1,
    ).dropna()

    y = aligned["y"]
    X = aligned[X_train.columns]

    print(f"  Fitting LightGBM on {len(X):,} rows × {X.shape[1]} features...")

    model = lgb.LGBMRegressor(
        n_estimators       = 300,
        learning_rate      = 0.05,
        max_depth          = 6,
        num_leaves         = 31,
        min_child_samples  = 50,
        feature_fraction   = 0.9,
        bagging_fraction   = 0.9,
        bagging_freq       = 5,
        random_state       = seed,
        n_jobs             = -1,
        verbose            = -1,
    )
    model.fit(X, y)

    # 'gain' importance from the booster. .feature_importance(type='gain')
    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X.columns,
        name="gain_importance",
    ).sort_values(ascending=False)

    selected = importance.head(keep_top_k).index.tolist()
    return selected, importance
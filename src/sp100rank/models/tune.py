# src/sp100rank/models/tune.py
"""
Hyperparameter tuning on Fold 1 train data only.

Per ADR-004's reasoning extended: selecting hyperparameters on later
folds would let us peek at their test sets via the model decision
itself, biasing reported IC. Fold 1 train is the most-past data;
tuning only there keeps the rest unbiased.

We split Fold 1's training period into:
  - inner_train: first 80% of Fold 1 train (chronological)
  - inner_val:   last 20% of Fold 1 train

The split has its own embargo (matching label horizon) so labels
in inner_train don't peek into inner_val.

For each candidate hyperparameter combination, we fit on inner_train,
predict on inner_val, score by mean Rank IC, and keep the winner.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from sp100rank.config import EMBARGO_DAYS
from sp100rank.eval.metrics import daily_rank_ic, summarize_ic
from sp100rank.models.registry import GRIDS, make_model


def _inner_train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_fraction: float = 0.2,
    embargo_days: int = EMBARGO_DAYS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split (X, y) into inner_train / inner_val along the date axis.

    Embargo applied between the two — same logic as outer walk-forward.

    Returns (X_tr, y_tr, X_val, y_val), all date-sorted.
    """
    # Get unique sorted dates from the index.
    dates = X.index.get_level_values("date").unique().sort_values()
    n = len(dates)

    val_n = int(round(n * val_fraction))
    train_end_idx  = n - val_n - embargo_days - 1
    val_start_idx  = train_end_idx + 1 + embargo_days

    if train_end_idx <= 0 or val_start_idx >= n:
        raise ValueError(
            f"Insufficient dates to split: have {n}, need at least "
            f"~{embargo_days + val_n + 1}."
        )

    train_end_date  = dates[train_end_idx]
    val_start_date  = dates[val_start_idx]
    val_end_date    = dates[-1]

    idx = pd.IndexSlice
    X_tr  = X.loc[idx[: train_end_date, :], :]
    y_tr  = y.loc[idx[: train_end_date, :]]
    X_val = X.loc[idx[val_start_date : val_end_date, :], :]
    y_val = y.loc[idx[val_start_date : val_end_date, :]]

    return X_tr, y_tr, X_val, y_val


def tune_model(
    model_name: str,
    X_fold1_train: pd.DataFrame,
    y_fold1_train: pd.Series,
    grid: dict[str, list] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Grid search on Fold 1 train (inner train/val split).

    Parameters
    ----------
    model_name : 'linear' | 'rf' | 'xgb' | 'lgb'
    X_fold1_train, y_fold1_train : feature matrix and label, restricted
        to Fold 1's training window. (date, ticker)-indexed.
    grid : optional override; defaults to GRIDS[model_name].

    Returns
    -------
    (best_params, results_df) :
      - best_params: dict of the winning hyperparameter combination
      - results_df: full grid with mean IC, ICIR per combo (for
        the writeup table)
    """
    if grid is None:
        grid = GRIDS[model_name]

    X_tr, y_tr, X_val, y_val = _inner_train_val_split(
        X_fold1_train, y_fold1_train,
    )
    if verbose:
        print(f"  inner_train: {X_tr.shape}, inner_val: {X_val.shape}")

    # Cartesian product over the grid.
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    results = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        model = make_model(model_name, **params)
        model.fit(X_tr, y_tr)

        preds = pd.Series(
            model.predict(X_val),
            index=X_val.index,
            name="pred",
        )
        ic_series = daily_rank_ic(preds, y_val)
        summary = summarize_ic(ic_series)
        results.append({**params, **summary})

        if verbose:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            print(f"    {params_str}  →  "
                  f"mean_IC={summary['mean_ic']:.4f}, "
                  f"ICIR={summary['icir']:.3f}")

    results_df = pd.DataFrame(results)
    # Pick by ICIR (mean IC penalized by stability) rather than raw mean.
    # Avoids picking models that won by luck in a few volatile dates.
    best_row = results_df.loc[results_df["icir"].idxmax()]

    # Cast numpy types back to native Python. sklearn's strict type
    # checking rejects np.float64 for int-typed parameters; passing
    # native types avoids this.
    best_params = {}
    for k in keys:
        v = best_row[k]
        if hasattr(v, "item"):  # numpy scalar
            v = v.item()
        # If the original grid had ints, keep them as int (pandas may
        # have promoted to float in the mixed-dtype results frame).
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        best_params[k] = v

    if verbose:
        print(f"  → best: {best_params} (ICIR={best_row['icir']:.3f})")

    return best_params, results_df
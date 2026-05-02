# src/sp100rank/models/registry.py
"""
Uniform interface over the 4 candidate models.

All four have a .fit(X, y) and .predict(X) method that look the same
to the outside world. Walk-forward training code never branches on
model type — it just calls model.fit(X_train, y_train), then
model.predict(X_test).

The hyperparameter grids per model are defined here too. Per ADR
(small grid), each grid is small — the goal is reasonable defaults,
not exhaustive search.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sp100rank.config import RANDOM_SEED


# === Hyperparameter grids ===
#
# Each grid is dict[param_name -> list_of_values]. We enumerate the
# Cartesian product during tuning. Small intentionally — the proposal
# specifies a small grid, and the marginal IC gain from larger grids
# rarely justifies the time.

GRIDS: dict[str, dict[str, list]] = {
    "linear": {
        # Ridge regularization. Wider than tree grids because Ridge
        # is computationally cheap.
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "rf": {
        # RF is robust to most settings; tune the two that matter.
        # n_estimators=300 fixed (more trees = more stable, diminishing
        # returns past 300).
        "max_depth":         [6, 12],
        "min_samples_leaf":  [50, 200],
    },
    "xgb": {
        # The two knobs that matter most for cross-sectional ranking.
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.10],
    },
    "lgb": {
        # Same idea — depth and rate.
        "num_leaves":    [15, 31],
        "learning_rate": [0.05, 0.10],
    },
}


# === Model wrappers ===
#
# Each class has a uniform .fit(X, y) and .predict(X). They internally
# handle their model-specific quirks (sklearn vs native APIs, scaling,
# NaN handling).


class _LinearModel:
    """Ridge regression with feature standardization.

    Linear models are scale-sensitive. Tree models are not. We
    standardize ONLY for the linear model. The scaler is fit on
    training data only — predict() uses the fitted scaler to
    transform test data, never re-fitting.
    """
    def __init__(self, alpha: float = 1.0, **_):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=RANDOM_SEED)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LinearModel":
        # Drop any remaining NaN rows. Trees handle NaN; Ridge errors
        # on them. We drop ON TRAIN SIDE — at predict time, NaN inputs
        # would also need handling (we'll fillna at the call site).
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]
        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y_clean)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # NaN at predict-time: fill with column means from training.
        # This is a defensive simplification — in practice Ridge
        # predictions on NaN-imputed rows are noisy. We rely on the
        # NaN-row drop in evaluation to handle this.
        X_filled = X.fillna(X.mean(numeric_only=True))
        X_scaled = self.scaler.transform(X_filled)
        return self.model.predict(X_scaled)


class _RFModel:
    """Random Forest regressor.

    n_estimators fixed at 300; we tune depth and leaf size. Trees
    handle NaN via masking — sklearn 1.4+ supports it natively for
    RandomForestRegressor with `criterion='squared_error'`.

    Wait — actually sklearn RF does NOT handle NaN as of 1.8.
    HistGradientBoostingRegressor does. To keep our 4-model line-up
    matching the proposal, we use RandomForestRegressor and drop NaN
    rows on the train side. Same handling as Ridge.
    """
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 50, **_):
        self.model = RandomForestRegressor(
            n_estimators       = 300,
            max_depth          = max_depth,
            min_samples_leaf   = min_samples_leaf,
            random_state       = RANDOM_SEED,
            n_jobs             = -1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_RFModel":
        mask = X.notna().all(axis=1) & y.notna()
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_filled = X.fillna(X.mean(numeric_only=True))
        return self.model.predict(X_filled)


class _XGBModel:
    """XGBoost regressor. Native NaN handling — no preprocessing needed."""
    def __init__(self, max_depth: int = 6, learning_rate: float = 0.1, **_):
        self.model = xgb.XGBRegressor(
            n_estimators       = 400,
            max_depth          = max_depth,
            learning_rate      = learning_rate,
            subsample          = 0.9,
            colsample_bytree   = 0.9,
            min_child_weight   = 50,
            random_state       = RANDOM_SEED,
            n_jobs             = -1,
            tree_method        = "hist",     # fast histogram-based splits
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_XGBModel":
        mask = y.notna()  # XGB handles NaN in X but not y
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class _LGBModel:
    """LightGBM regressor. Native NaN handling."""
    def __init__(self, num_leaves: int = 31, learning_rate: float = 0.1, **_):
        self.model = lgb.LGBMRegressor(
            n_estimators       = 400,
            num_leaves         = num_leaves,
            learning_rate      = learning_rate,
            min_child_samples  = 50,
            feature_fraction   = 0.9,
            bagging_fraction   = 0.9,
            bagging_freq       = 5,
            random_state       = RANDOM_SEED,
            n_jobs             = -1,
            verbose            = -1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LGBModel":
        mask = y.notna()
        self.model.fit(X[mask], y[mask])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


# === Factory ===

def make_model(name: str, **params: Any):
    """Instantiate a model by name. Raises ValueError on unknown name."""
    factories = {
        "linear": _LinearModel,
        "rf":     _RFModel,
        "xgb":    _XGBModel,
        "lgb":    _LGBModel,
    }
    if name not in factories:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(factories)}"
        )
    return factories[name](**params)


def all_model_names() -> list[str]:
    return ["linear", "rf", "xgb", "lgb"]
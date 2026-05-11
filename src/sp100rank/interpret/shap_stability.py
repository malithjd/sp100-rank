# src/sp100rank/interpret/shap_stability.py
"""
SHAP stability analysis across walk-forward folds.

For each fold's trained Random Forest model, compute mean absolute
SHAP value per feature on the test set. The output table (features ×
folds) shows whether feature importance is stable through regime
shifts or whether the model fundamentally relies on different
signals in different periods.

Stable features (consistent rank across folds) are robust signals.
Features that swing wildly (rank #1 in one fold, #6 in another) are
regime-dependent — useful only when the model can correctly recognize
the current regime.

Reference: Bryzgalova, Pelger & Zhu (2023, J. of Finance) document
similar analysis for tree-based asset pricing models.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from sp100rank.config import (
    CHECKPOINT_DIR, PROCESSED_DATA_DIR, RANDOM_SEED,
)
from sp100rank.eval.walkforward import walk_forward_folds
from sp100rank.features.build import build_features_and_labels
from sp100rank.data.clean import load_clean_panel


def load_model(model_name: str, fold_id: int):
    """Load a pickled model from checkpoints/."""
    path = CHECKPOINT_DIR / model_name / f"fold{fold_id}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_shap_stability(
    model_name: str = "rf",
    sample_size: int = 5000,
) -> pd.DataFrame:
    """Compute mean |SHAP| per feature per fold.

    Parameters
    ----------
    model_name : default "rf" (the leader from Phase 4).
    sample_size : per-fold SHAP is computed on at most this many rows.
        SHAP is expensive; 5000 is enough for stable mean |SHAP| but
        ~10x faster than the full ~12,500-row test set.

    Returns
    -------
    DataFrame with rows = features, columns = fold_id, values = mean
    |SHAP| on that fold's test data.
    """
    panel = load_clean_panel()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    folds = list(walk_forward_folds(all_dates))

    selected = json.loads(
        (PROCESSED_DATA_DIR / "selected_features.json").read_text()
    )["selected_features"]

    fold_columns = []
    for f in folds:
        print(f"  Fold {f.fold_id}: loading model + computing SHAP...")
        wrapper = load_model(model_name, f.fold_id)
        # The model wrapper has .model (the actual sklearn/XGB/LGB model).
        # For RF: wrapper.model is RandomForestRegressor.
        underlying = wrapper.model

        # Build the test features for this fold.
        X_test, _ = build_features_and_labels(
            start=f.test_start, end=f.test_end,
        )
        X_test = X_test[selected]
        
        X_test_clean = X_test.dropna()
        if len(X_test_clean) > sample_size:
            X_test_clean = X_test_clean.sample(
                n=sample_size, random_state=RANDOM_SEED,
            )

        
        explainer = shap.TreeExplainer(
            underlying,
            feature_perturbation="tree_path_dependent",
        )
        shap_values = explainer.shap_values(X_test_clean)

        mean_abs = np.abs(shap_values).mean(axis=0)
        fold_columns.append(
            pd.Series(mean_abs, index=X_test_clean.columns,
                      name=f"fold_{f.fold_id}")
        )

    return pd.concat(fold_columns, axis=1)


def to_rank_table(stability_df: pd.DataFrame) -> pd.DataFrame:
    """Convert mean |SHAP| values to within-fold ranks.

    Rank 1 = most important in that fold. The plot of this is the
    headline interpretability figure.
    """
    return stability_df.rank(axis=0, method="min", ascending=False).astype(int)
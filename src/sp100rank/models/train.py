# src/sp100rank/models/train.py
"""
Walk-forward model training.

For each (model, fold) pair:
  1. Fit on fold.train_start → fold.train_end.
  2. Predict on fold.test_start → fold.test_end.
  3. Save predictions to data/processed/predictions/{model}/fold{k}.parquet
  4. Save trained model to models/checkpoints/{model}/fold{k}.pkl

The same hyperparameters (chosen via tune_model on Fold 1) are used
across all 5 folds for a given model. This is consistent with our
no-future-leakage discipline.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from sp100rank.config import CHECKPOINT_DIR, PROCESSED_DATA_DIR
from sp100rank.eval.walkforward import Fold, walk_forward_folds
from sp100rank.features.build import build_features_and_labels
from sp100rank.models.registry import make_model


def train_model_walkforward(
    model_name: str,
    best_params: dict,
    folds: list[Fold],
    selected_features: list[str],
    verbose: bool = True,
) -> dict[int, pd.Series]:
    """Train a single model across all folds; return per-fold predictions.

    Parameters
    ----------
    model_name : 'linear' | 'rf' | 'xgb' | 'lgb'
    best_params : hyperparameters from tuning step
    folds : list of Fold dataclasses (typically 5)
    selected_features : 8-name feature list from selection step

    Returns
    -------
    dict mapping fold_id -> Series of test-period predictions.
    """
    # Set up output directories. mkdir parents=True is idempotent.
    pred_dir = PROCESSED_DATA_DIR / "predictions" / model_name
    ckpt_dir = CHECKPOINT_DIR / model_name
    pred_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_predictions = {}

    for fold in folds:
        if verbose:
            print(f"  Fold {fold.fold_id}: train [{fold.train_start.date()} → "
                  f"{fold.train_end.date()}], test [{fold.test_start.date()} → "
                  f"{fold.test_end.date()}]")

        # Build train and test (X, y) sliced to fold boundaries.
        X_train, y_train = build_features_and_labels(
            start=fold.train_start, end=fold.train_end,
        )
        X_test, y_test = build_features_and_labels(
            start=fold.test_start, end=fold.test_end,
        )

        # Restrict to selected features only.
        X_train = X_train[selected_features]
        X_test  = X_test[selected_features]

        # Fit on train, predict on test.
        model = make_model(model_name, **best_params)
        model.fit(X_train, y_train)
        preds = pd.Series(
            model.predict(X_test),
            index=X_test.index,
            name="pred",
        )

        # Persist predictions and model.
        preds.to_frame().to_parquet(pred_dir / f"fold{fold.fold_id}.parquet")
        with open(ckpt_dir / f"fold{fold.fold_id}.pkl", "wb") as f:
            pickle.dump(model, f)

        fold_predictions[fold.fold_id] = preds

        if verbose:
            print(f"    saved predictions ({len(preds):,} rows) and checkpoint")

    return fold_predictions
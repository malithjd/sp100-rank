# src/sp100rank/pipeline/retrain.py
"""
Production retraining entry point.

Run via: `uv run python -m sp100rank.pipeline.retrain`

What it does:
  1. Refresh all data from yfinance.
  2. Rebuild the clean panel.
  3. Train the lead model (RF) on all data through the most recent
     date with valid (feature, label) pairs.
  4. Save a timestamped pickle AND update 'latest.pkl'.
  5. Print the path so the GitHub workflow can capture and upload to
     a Release.

Cadence: per ADR-009 and proposal Task 5, this runs on manual trigger
(workflow_dispatch) or quarterly via cron when enabled. Scoring fetches
'latest.pkl' to pick up the new model on its next run.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sp100rank.config import CHECKPOINT_DIR, PROCESSED_DATA_DIR
from sp100rank.data.clean import build_clean_panel
from sp100rank.data.ingest import download_universe
from sp100rank.features.build import build_features_and_labels
from sp100rank.models.registry import make_model


def retrain(model_name: str = "rf") -> Path:
    """Retrain on all available data; return path to new checkpoint.

    The training data is everything from TRAIN_START through the latest
    date for which we have BOTH features (warmup completed) AND labels
    (forward window completes). The trailing ~21 days have no label
    yet and are excluded.
    """
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting retrain.")

    # 1. Refresh data.
    print("Refreshing data...")
    download_universe()
    panel = build_clean_panel()

    # 2. Build features and labels.
    print("Computing features and labels...")
    X, y = build_features_and_labels()

    # 3. Restrict to selected features.
    selected = json.loads(
        (PROCESSED_DATA_DIR / "selected_features.json").read_text()
    )["selected_features"]
    X = X[selected]

    # 4. Drop rows with NaN labels (trailing ~21 dates per ticker).
    aligned = pd.concat({"y": y, **{c: X[c] for c in X.columns}}, axis=1).dropna()
    y_train = aligned["y"]
    X_train = aligned[X.columns]
    print(f"Training shape: {X_train.shape}")

    # 5. Load tuned hyperparameters.
    hp = json.loads(
        (PROCESSED_DATA_DIR / "tuned_hyperparameters.json").read_text()
    )[model_name]

    # Cast numpy/float-int back to native Python int (sklearn strict).
    hp_clean = {}
    for k, v in hp.items():
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        hp_clean[k] = v
    print(f"Hyperparameters: {hp_clean}")

    # 6. Fit.
    model = make_model(model_name, **hp_clean)
    model.fit(X_train, y_train)

    # 7. Save.
    ckpt_dir = CHECKPOINT_DIR / model_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_path = ckpt_dir / f"retrained_{timestamp}.pkl"
    latest_path = ckpt_dir / "latest.pkl"

    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)
    shutil.copy(ckpt_path, latest_path)
    print(f"Saved: {ckpt_path}")
    print(f"Updated: {latest_path}")

    return ckpt_path


def main() -> int:
    try:
        ckpt = retrain()
        # Print path so GitHub workflow can capture for Release upload.
        print(f"CHECKPOINT_PATH={ckpt}")
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
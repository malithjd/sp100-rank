# src/sp100rank/data/clean.py
"""
Clean the raw OHLCV panel: drop zero-volume rows, forward-fill isolated
price gaps, flag and drop extreme price spikes.

The cleaning is intentionally CONSERVATIVE. We'd rather lose a few rows
than silently keep bad data. Every drop is logged; the writeup will
include the count.

Outputs are written to data/processed/clean_panel.parquet so subsequent
phases (features, labels) read from a single canonical file.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sp100rank.config import PROCESSED_DATA_DIR
from sp100rank.data.ingest import load_raw_panel
from sp100rank.data.universe import is_index


# Thresholds. Document these in the ADR if changed.
ZERO_VOL_TOLERANCE = 0       # drop rows with volume == 0 (equity only)
SPIKE_RETURN_THRESHOLD = 0.50  # one-day return >50% in absolute value
                                # is flagged as a spike. Real names rarely
                                # do this without it being a data error
                                # OR a major corporate event (in which
                                # case the day is junk for our features).
MAX_FFILL_DAYS = 2            # if a price gap is longer than 2 days,
                                # we treat it as data missing-ness, not
                                # a hold. Multi-day fills can fabricate
                                # signal where there is none.


def clean_panel(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all cleaning steps. Return (cleaned panel, stats dict).

    Stats dict has counts of what we dropped/filled. Useful for the
    writeup's "data quality" subsection.
    """
    stats = {
        "input_rows": len(panel),
        "zero_volume_dropped": 0,
        "isolated_gaps_filled": 0,
        "spike_rows_dropped": 0,
        "tickers_with_spikes": [],
    }

    # Work on a copy so the input is untouched (callers may want both).
    df = panel.copy().sort_index()

    # --- Step 1: drop zero-volume rows for EQUITIES only.
    # Indices like ^GSPC have a 'volume' that's actually a sum across
    # constituents and can legitimately be zero on weird days. We don't
    # use volume on indices anyway, so just don't drop their rows.
    is_equity = ~df.index.get_level_values("ticker").map(is_index)
    zero_vol_mask = (df["volume"] <= ZERO_VOL_TOLERANCE) & is_equity
    stats["zero_volume_dropped"] = int(zero_vol_mask.sum())
    df = df[~zero_vol_mask]

    # --- Step 2: forward-fill isolated gaps in adj_close.
    # We do this PER TICKER (groupby) so we never fill across ticker
    # boundaries. Limit=MAX_FFILL_DAYS prevents long-fill fabrication.
    #
    # Subtle: ffill only fills NaNs that are ALREADY there. If a ticker
    # is genuinely missing a date (no row at all for that day), ffill
    # won't create a row. That's intentional — we don't manufacture
    # rows. The cross-section just has one fewer name on that date.
    before_na = df["adj_close"].isna().sum()
    df["adj_close"] = (
        df.groupby(level="ticker")["adj_close"]
          .ffill(limit=MAX_FFILL_DAYS)
    )
    after_na = df["adj_close"].isna().sum()
    stats["isolated_gaps_filled"] = int(before_na - after_na)

    # --- Step 3: flag and drop extreme price spikes.
    # Compute one-day adjusted return per ticker.
    daily_ret = (
        df.groupby(level="ticker")["adj_close"]
          .pct_change()
    )
    spike_mask = daily_ret.abs() > SPIKE_RETURN_THRESHOLD
    # Don't drop spikes on indices (very rare anyway, and we don't want
    # to lose ^GSPC observations).
    spike_mask = spike_mask & is_equity
    stats["spike_rows_dropped"] = int(spike_mask.sum())
    if spike_mask.any():
        # Record which tickers had spikes for the audit notebook.
        spike_tickers = (
            df[spike_mask]
            .index.get_level_values("ticker")
            .unique()
            .tolist()
        )
        stats["tickers_with_spikes"] = spike_tickers
    df = df[~spike_mask]

    # --- Step 4: drop any remaining NaN adj_close rows.
    # These are ticker-date pairs where forward fill couldn't recover
    # the price (e.g., gap longer than MAX_FFILL_DAYS). Cleaner to
    # drop than to carry NaNs into feature computation.
    nan_mask = df["adj_close"].isna()
    df = df[~nan_mask]

    stats["output_rows"] = len(df)
    stats["dropped_total"] = stats["input_rows"] - stats["output_rows"]

    return df, stats


def build_clean_panel(write: bool = True) -> pd.DataFrame:
    """Top-level entry: load raw, clean, optionally write to disk.

    Idempotent — running twice produces the same output (the source
    raw files don't change).
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_panel()
    clean, stats = clean_panel(raw)

    print("Cleaning summary:")
    for k, v in stats.items():
        if isinstance(v, list) and len(v) > 8:
            print(f"  {k}: {v[:8]}... (+{len(v)-8} more)")
        else:
            print(f"  {k}: {v}")

    if write:
        out_path = PROCESSED_DATA_DIR / "clean_panel.parquet"
        clean.to_parquet(out_path)
        print(f"\nWrote {len(clean):,} rows to {out_path}")

    return clean


def load_clean_panel() -> pd.DataFrame:
    """Read the cached cleaned panel."""
    path = PROCESSED_DATA_DIR / "clean_panel.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No clean panel at {path}. Run build_clean_panel() first."
        )
    return pd.read_parquet(path)


if __name__ == "__main__":
    build_clean_panel()
# src/sp100rank/data/ingest.py
"""
Download adjusted OHLCV data from yfinance for the project universe
and cache to Parquet. Designed to be RESUMABLE — if a download fails
halfway, rerunning picks up where it left off.

Why per-ticker files (not one big file): if yfinance fails on ticker
N of 101, we keep the first N-1 files and only retry the failed ones.
The combined panel is built at the end via concat.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from sp100rank.config import RAW_DATA_DIR, TRAIN_START, FROZEN_DATA_END
from sp100rank.data.universe import all_tickers


_YF_COLUMN_RENAME = {
    "Open":      "open",
    "High":      "high",
    "Low":       "low",
    "Close":     "close",
    "Adj Close": "adj_close",
    "Volume":    "volume",
}

def _ticker_path(ticker: str) -> Path:
    """Filesystems handle '-' fine but '^' is best avoided. We
    replace it with '_idx_' for filename safety. The original ticker
    is preserved in the DataFrame's column.
    """
    safe = ticker.replace("^", "_idx_")
    return RAW_DATA_DIR / f"{safe}.parquet"

def _download_one(
    ticker: str,
    start: str,
    end: str,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                # auto_adjust=False keeps unadjusted Open/High/Low/Close
                # AND a separate Adj Close column. We need both: unadjusted
                # for volume (which isn't adjusted), adjusted for returns.
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
                threads=False,
            )

            if df is None or df.empty:
                raise RuntimeError(f"empty frame returned for {ticker}")

            return df

        except Exception as e:
            last_err = e
            if attempt < retries:
                wait = sleep_seconds * (2 ** (attempt - 1))
                print(f"  retry {attempt}/{retries} for {ticker} after {wait:.1f}s: {e}")
                time.sleep(wait)

    raise RuntimeError(f"failed to download {ticker} after {retries} retries: {last_err}")


def _normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Standardize a single-ticker yfinance frame to our schema.

    Output columns: open, high, low, close, adj_close, volume, ticker.
    Index: DatetimeIndex named 'date' (timezone-stripped — yfinance
    sometimes returns tz-aware indexes which pandas merges hate).
    """
    df = df.rename(columns=_YF_COLUMN_RENAME)
    expected = ["open", "high", "low", "close", "adj_close", "volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing columns {missing}; got {list(df.columns)}")
    df = df[expected].copy()

    # Strip timezone if present.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    df["ticker"] = ticker
    return df


def download_universe(
    start: str = TRAIN_START,
    end: str | None = None,
    refresh: bool = False,
) -> dict[str, str]:
    """Download all tickers, caching one Parquet per ticker.

    Parameters
    ----------
    start, end : ISO date strings (inclusive on start, exclusive on end
        per yfinance convention). yfinance's `end` is exclusive; if you
        want data through 2026-03-31, pass end='2026-04-01'. I add one
        day inside this function so callers can pass intuitive dates.
    refresh : if False, skip tickers whose cache file already exists.
        Set True when you suspect data corruption or a yfinance update.

    Returns
    -------
    dict mapping ticker -> status. Statuses: 'ok', 'cached', 'failed'.
    Inspect this to find tickers that need a manual retry.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        print(f"end=None → using today: {end}")
    # yfinance treats `end` as exclusive. Bump by one day so the user
    # provides the last date they want INCLUDED.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    statuses: dict[str, str] = {}
    tickers = all_tickers()

    for i, ticker in enumerate(tickers, start=1):
        path = _ticker_path(ticker)
        prefix = f"[{i:>3}/{len(tickers)}] {ticker:<8}"

        if path.exists() and not refresh:
            try:
                cached = pd.read_parquet(path)
                last_date = cached.index.max()
                today = pd.Timestamp.today().normalize()
                days_behind = (today - last_date).days
                # 7-day tolerance handles weekends + holidays without
                # forcing weekday-by-weekday re-downloads.
                if days_behind <= 7:
                    print(f"{prefix} cached ({last_date.date()}) -> {path.name}")
                    statuses[ticker] = "cached"
                    continue
                else:
                    print(f"{prefix} cache stale ({days_behind} days behind), refetching...")
            except Exception:
                # Corrupt cache → just redownload.
                print(f"{prefix} cache unreadable, refetching...")

        try:
            print(f"{prefix} downloading...", end=" ", flush=True)
            df = _download_one(ticker, start=start, end=end_exclusive)
            df = _normalize(df, ticker)
            df.to_parquet(path)
            print(f"ok ({len(df)} rows)")
            statuses[ticker] = "ok"
            # Throttle: yfinance starts rate-limiting around 60 req/min.
            # Half a second between bulk requests seemed safe.
            time.sleep(0.5)
        except Exception as e:
            print(f"FAILED: {e}")
            statuses[ticker] = "failed"

    # Summary
    print("\n" + "=" * 60)
    n_ok      = sum(1 for s in statuses.values() if s == "ok")
    n_cached  = sum(1 for s in statuses.values() if s == "cached")
    n_failed  = sum(1 for s in statuses.values() if s == "failed")
    print(f"Downloaded: {n_ok} | Cached: {n_cached} | Failed: {n_failed}")
    if n_failed:
        failed = [t for t, s in statuses.items() if s == "failed"]
        print(f"Failed tickers: {failed}")
        print("Re-run download_universe(refresh=True) to retry, or call _download_one() manually.")

    return statuses


def load_raw_panel() -> pd.DataFrame:
    """Read all per-ticker Parquets and concatenate into one panel.

    Returns a frame with MultiIndex (date, ticker) and columns
    open/high/low/close/adj_close/volume. Sorted by (date, ticker).

    This is the raw, uncleaned panel. Use load_clean_panel() for the
    cleaned version once cleaning is implemented.
    """
    files = sorted(RAW_DATA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No raw data found in {RAW_DATA_DIR}. "
            f"Run download_universe() first."
        )

    frames = [pd.read_parquet(f) for f in files]
    panel = pd.concat(frames)

    panel = panel.set_index("ticker", append=True)
    # Reorder so date is the outer level.
    panel = panel.reorder_levels(["date", "ticker"]).sort_index()

    return panel


if __name__ == "__main__":
    # Allow running this module directly: `uv run python -m sp100rank.data.ingest`
    download_universe()


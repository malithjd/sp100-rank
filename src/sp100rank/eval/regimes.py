# src/sp100rank/eval/regimes.py
"""
Rule-based regime tagging using SPX as the market proxy.

Two axes, defined to be:
  - REPRODUCIBLE: rules based on observable price data, not hand-picked
    date ranges.
  - CAUSAL: a date's regime is determined by data up to and including
    that date — never by future events.
  - ORTHOGONAL: trend and volatility classifications are independent
    enough to give 4 cells.

Why rule-based and not "calendar regimes" (e.g., '2020 crash period'):
  Hand-picked date windows invite the criticism "you defined regimes
  AFTER seeing where the model performed." Rule-based tagging removes
  that degree of freedom.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from sp100rank.config import SPX_TICKER


def tag_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute trend and volatility regime tags per date.

    Parameters
    ----------
    panel : the full clean panel with MultiIndex (date, ticker).
        Must contain ^GSPC for the SPX series.

    Returns
    -------
    DataFrame indexed by date with columns:
      - trend_regime: "bull" if SPX > 200-day MA, "bear" otherwise.
      - vol_regime:   "high_vol" / "low_vol" based on rolling realized
                      vol vs its in-sample median.
    Both are causal — they only use SPX data through that date.
    """
    spx_close = (
        panel.xs(SPX_TICKER, level="ticker")["adj_close"]
        .sort_index()
    )

    # === Trend regime: SPX vs its own 200-day MA ===
    # 200-day MA at date t uses prices from t-199 through t. Causal.
    ma200 = spx_close.rolling(window=200, min_periods=200).mean()
    trend = pd.Series("bull", index=spx_close.index, name="trend_regime")
    trend[spx_close < ma200] = "bear"
    # Warmup: first 200 days have no MA. Mark as None.
    trend[ma200.isna()] = None

    # === Volatility regime: 60-day realized vol vs its rolling median ===
    # We use a rolling median rather than a fixed threshold to avoid
    # arbitrary cutoffs. The median is computed over an EXPANDING
    # window (all data up to date t), so the threshold itself is
    # causal but adapts as more data accumulates.
    daily_ret = spx_close.pct_change()
    rv60 = daily_ret.rolling(window=60, min_periods=60).std(ddof=0)
    rv60_median_expanding = rv60.expanding(min_periods=200).median()

    vol = pd.Series("low_vol", index=spx_close.index, name="vol_regime")
    vol[rv60 > rv60_median_expanding] = "high_vol"
    vol[rv60.isna() | rv60_median_expanding.isna()] = None

    return pd.DataFrame({
        "trend_regime": trend,
        "vol_regime":   vol,
    })


def join_regimes_to_predictions(
    predictions: pd.Series,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Attach regime tags to a predictions Series for groupby analysis.

    Returns a DataFrame indexed by (date, ticker) with columns
    pred + trend_regime + vol_regime.
    """
    regimes = tag_regimes(panel)
    pred_df = predictions.to_frame(name="pred")
    # Reset to a date column for merging, then re-set index.
    pred_df = pred_df.reset_index()
    pred_df = pred_df.merge(regimes, left_on="date", right_index=True, how="left")
    pred_df = pred_df.set_index(["date", "ticker"])
    return pred_df
# src/sp100rank/eval/metrics.py
"""
Information Coefficient (IC) metrics.

The cross-sectional ranking task evaluates one number per DATE: how
well does the model rank stocks against each other on that day? We
compute that per date, then summarize across dates.

Why per-date and not pooled across (date, ticker):
  - Pooling correlates predictions and labels across the whole panel.
    That includes between-date variation, which has nothing to do
    with the model's cross-sectional discrimination ability. Pooled
    IC inflates the metric.
  - Per-date IC gives a TIME SERIES of skill. ICIR (= mean/std of
    that series) is a Sharpe-ratio-analog for alpha — a far more
    informative number than pooled IC alone.

Why Spearman / Rank IC and not Pearson:
  - Our LABELS are already ranks (uniform on [0,1]). Pearson on rank-
    label is equivalent to Spearman up to ties; both are robust.
  - We use scipy.spearmanr because it handles ties via average-rank
    automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def daily_rank_ic(
    predictions: pd.Series,
    labels: pd.Series,
    min_cross_section: int = 10,
) -> pd.Series:
    """Spearman correlation between predictions and labels, per date.

    Both inputs are Series with a (date, ticker) MultiIndex. Inner-join
    on the shared index drops any row missing from either side.

    Parameters
    ----------
    predictions : (date, ticker) -> predicted rank/score
    labels      : (date, ticker) -> realized rank
    min_cross_section : minimum tickers needed on a date to compute a
        meaningful Spearman. With 100 tickers in our universe this
        won't trip in practice; defensive against edge dates.

    Returns
    -------
    Series indexed by date, values are per-date Spearman IC.
    """
    df = pd.concat({"pred": predictions, "y": labels}, axis=1).dropna()

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < min_cross_section:
            return np.nan
        # spearmanr returns a SignificanceResult tuple in modern scipy;
        # we want the correlation coefficient, .statistic.
        result = spearmanr(group["pred"], group["y"])
        return result.statistic

    return df.groupby(level="date").apply(_ic)


def summarize_ic(ic: pd.Series) -> dict:
    """Reduce a per-date IC series to scalar diagnostics.

    Returns a dict with:
      mean_ic   : average per-date IC
      std_ic    : std of per-date IC across the test period
      icir      : mean / std — a Sharpe-ratio-analog for alpha
      t_stat    : tests H0: mean IC = 0; useful for "is this real?"
      hit_rate  : fraction of dates with positive IC
      n_days    : number of evaluated dates

    Reporting note: ICIR is typically annualized in industry
    (multiply by sqrt(periods_per_year)). We return the raw ICIR
    here; the writeup handles annualization on top.
    """
    ic = ic.dropna()
    if len(ic) == 0:
        return {
            "mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan,
            "t_stat": np.nan, "hit_rate": np.nan, "n_days": 0,
        }
    mean = ic.mean()
    std = ic.std(ddof=1)
    icir = mean / std if std > 0 else np.nan
    # One-sample t-stat for H0: mean = 0. n-1 degrees of freedom.
    tstat = mean / (std / np.sqrt(len(ic))) if std > 0 else np.nan
    return {
        "mean_ic":  mean,
        "std_ic":   std,
        "icir":     icir,
        "t_stat":   tstat,
        "hit_rate": (ic > 0).mean(),
        "n_days":   len(ic),
    }
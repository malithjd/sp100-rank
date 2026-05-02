# src/sp100rank/eval/portfolio.py
"""
Decile portfolio simulation from cross-sectional predictions.

Strategy:
  - At each rebalance date, sort the 100 stocks by predicted rank.
  - Long the top decile (10 stocks), equal-weighted.
  - Short the bottom decile (10 stocks), equal-weighted.
  - Hold for `rebalance_every` days, then re-pick deciles.

This translates the cross-sectional ranking signal into a tradeable
strategy. The IC's "real-world" implication is what the simulation
delivers: Sharpe ratio, max drawdown, transaction-cost sensitivity.

Why non-overlapping holds (rebalance_every == horizon):
  Daily rebalancing on a 20-day horizon prediction would have
  overlapping holdings (20 different prediction "vintages" active
  simultaneously). Non-overlapping is the conservative / honest
  setup: we get one independent observation per holding period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sp100rank.config import PROCESSED_DATA_DIR
from sp100rank.data.clean import load_clean_panel
from sp100rank.eval.walkforward import walk_forward_folds
from sp100rank.features.labels import forward_return
from sp100rank.models.registry import all_model_names


def simulate_decile_portfolio(
    predictions: pd.Series,
    panel: pd.DataFrame,
    n_deciles: int = 10,
    rebalance_every: int = 20,
    bps_round_trip: float = 0.0,
) -> pd.DataFrame:
    """Simulate non-overlapping long-short decile portfolio returns.

    Parameters
    ----------
    predictions : (date, ticker) -> predicted rank/score on each date.
    panel : the full clean panel (used to compute realized returns).
    bps_round_trip : transaction cost per round-trip in basis points.
        We assume 100% turnover per rebalance (worst case): both legs
        churn fully when deciles are re-picked. Cost per period:
        2 * bps / 1e4 (long leg + short leg).

    Returns
    -------
    DataFrame indexed by rebalance date, columns:
      long_return, short_return, gross_return, net_return.
    """
    # Build per-date forward returns at the same horizon as labels.
    fr = forward_return(panel, horizon=rebalance_every, execution_lag=1)

    # Align predictions and forward returns on (date, ticker).
    aligned = pd.concat({"pred": predictions, "fwd": fr}, axis=1).dropna()

    # Pick rebalance dates: every `rebalance_every`-th date starting
    # from the earliest valid one.
    all_dates = aligned.index.get_level_values("date").unique().sort_values()
    rebalance_dates = all_dates[::rebalance_every]

    rows = []
    for d in rebalance_dates:
        try:
            snap = aligned.xs(d, level="date").copy()
        except KeyError:
            continue
        if len(snap) < n_deciles * 2:
            continue

        # qcut handles ties via the duplicates parameter. Decile 9
        # = top, decile 0 = bottom (in 0-indexed convention).
        snap["decile"] = pd.qcut(
            snap["pred"], q=n_deciles, labels=False, duplicates="drop"
        )

        long_ret = snap.loc[snap["decile"] == n_deciles - 1, "fwd"].mean()
        short_ret = snap.loc[snap["decile"] == 0, "fwd"].mean()
        gross = long_ret - short_ret
        # Round-trip cost on both legs.
        cost = 2 * bps_round_trip / 1e4
        net = gross - cost

        rows.append({
            "date": d,
            "long_return": long_ret,
            "short_return": short_ret,
            "gross_return": gross,
            "net_return": net,
        })

    return pd.DataFrame(rows).set_index("date")


def portfolio_diagnostics(
    returns: pd.Series,
    periods_per_year: float = 252 / 20,
) -> dict:
    """Annualized Sharpe, vol, drawdown, hit rate.

    With non-overlapping 20-day holds, periods_per_year ≈ 12.6.
    """
    r = returns.dropna()
    if len(r) == 0:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "max_dd": np.nan, "hit_rate": np.nan, "n_periods": 0}
    mean = r.mean() * periods_per_year
    std = r.std(ddof=1) * np.sqrt(periods_per_year)
    sharpe = mean / std if std > 0 else np.nan
    cum = (1 + r).cumprod()
    drawdown = (cum / cum.cummax()) - 1.0
    return {
        "ann_return": mean,
        "ann_vol": std,
        "sharpe": sharpe,
        "max_dd": drawdown.min(),
        "hit_rate": (r > 0).mean(),
        "n_periods": len(r),
    }


def simulate_all_models_all_costs() -> pd.DataFrame:
    """For each model, run the simulation at 0/5/10 bps. Return a table."""
    panel = load_clean_panel()
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    folds = list(walk_forward_folds(all_dates))

    rows = []
    for model_name in all_model_names():
        pred_dir = PROCESSED_DATA_DIR / "predictions" / model_name

        # Concatenate predictions across all 5 test folds.
        all_preds = []
        for f in folds:
            p = pd.read_parquet(pred_dir / f"fold{f.fold_id}.parquet")["pred"]
            all_preds.append(p)
        full_preds = pd.concat(all_preds)

        for bps in [0.0, 5.0, 10.0]:
            returns_df = simulate_decile_portfolio(
                predictions=full_preds, panel=panel,
                bps_round_trip=bps,
            )
            diag = portfolio_diagnostics(returns_df["net_return"])
            rows.append({
                "model": model_name,
                "bps_round_trip": bps,
                **diag,
            })

    return pd.DataFrame(rows)
"""
Forward-walking backtest of the sp100-rank top-K strategy.

Run with defaults (18-month backtest from 2024-11-01):
    uv run python -m sp100rank.eval.backtest

Customize via CLI flags:
    uv run python -m sp100rank.eval.backtest \
        --start 2024-11-01 \
        --top-k 5 \
        --capital 10000 \
        --tx-cost-bps 5

Strategy:
  - Start with $CAPITAL cash on START date.
  - Each month, retrain the RF model on data through (rebalance - 21d)
    to avoid using labels that require future data.
  - Predict the rebalance date. Pick top-K by predicted rank.
  - Compare with current holdings. Sell drops at close. Use proceeds
    to buy adds, equal-weighted across new purchases.
  - Hold ~21 trading days, then rebalance.
  - Same-day execution at close.
  - Compare against SPY buy-and-hold.
  - Two cost scenarios: gross (0 bps) and net (TX_COST_BPS round-trip).

Outputs (under data/processed/ and figures/):
  - backtest_<run-tag>_daily.csv       — daily portfolio values
  - backtest_<run-tag>_trades.csv      — every buy/sell with prices
  - backtest_<run-tag>_holdings.csv    — what was held each month
  - backtest_<run-tag>_equity.png      — equity curve vs SPY
  - backtest_<run-tag>_monthly_pnl.png — bar chart of period returns
  - backtest_<run-tag>_holdings.png    — holdings grid over time
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from sp100rank.config import (
    CHECKPOINT_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)
from sp100rank.data.clean import build_clean_panel
from sp100rank.data.ingest import download_universe
from sp100rank.features.build import build_features_and_labels
from sp100rank.models.registry import make_model


# ============================================================
# Configuration
# ============================================================

@dataclass
class BacktestConfig:
    """All knobs in one place. Override via CLI flags or by editing here."""

    # --- Time ---
    start_date: pd.Timestamp = pd.Timestamp("2024-11-01")
    end_date: pd.Timestamp | None = None      # None → use today's data

    # --- Strategy ---
    strategy_mode: str = "long_only"          # "long_only" or "long_short"
    top_k: int = 5                            # how many stocks long (and short)
    initial_capital: float = 10_000.0
    holding_days: int = 21                    # ~1 month between rebalances

    # --- Long-short specific ---
    short_borrow_bps_annual: float = 50.0     # annual borrow cost for shorts

    # --- Costs ---
    tx_cost_bps: float = 5.0                  # round-trip basis points

    # --- Model ---
    model_name: str = "rf"                    # which model to retrain
    label_horizon: int = 21                   # forward-return window for labels
    retrain_buffer_days: int = 7              # extra buffer past label_horizon

    # --- Benchmark ---
    benchmark_ticker: str = "SPY"

    # --- Output ---
    run_tag: str = "default"                  # appended to output filenames
    fig_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "figures")
    data_dir: Path = field(default_factory=lambda: PROCESSED_DATA_DIR)


# ============================================================
# Helpers
# ============================================================

@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    side: str
    shares: float
    price: float
    notional: float


def get_rebalance_dates(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Find monthly rebalance dates as actual trading days within [start, end].

    For each calendar month start, snap forward to the next trading day.
    """
    all_dates = panel.index.get_level_values("date").unique().sort_values()
    available = all_dates[(all_dates >= start) & (all_dates <= end)]

    target_months = []
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1)
    last_target = pd.Timestamp(year=end.year, month=end.month, day=1)
    while cursor <= last_target:
        target_months.append(cursor)
        if cursor.month == 12:
            cursor = pd.Timestamp(year=cursor.year + 1, month=1, day=1)
        else:
            cursor = pd.Timestamp(year=cursor.year, month=cursor.month + 1, day=1)

    rebal = []
    for target in target_months:
        candidates = available[available >= target]
        if len(candidates) > 0:
            rebal.append(candidates[0])
    return rebal


def train_model_through_date(
    selected_features: list[str],
    hp: dict,
    train_through: pd.Timestamp,
    model_name: str,
):
    """Train a fresh model on data ≤ train_through with valid labels."""
    X, y = build_features_and_labels()
    X = X[selected_features]

    mask = X.index.get_level_values("date") <= train_through
    X_train = X[mask]
    y_train = y.reindex(X_train.index)

    aligned = pd.concat({"y": y_train, **{c: X_train[c] for c in X_train.columns}}, axis=1).dropna()
    y_clean = aligned["y"]
    X_clean = aligned[X_train.columns]

    print(f"    Training {model_name} on {len(X_clean):,} rows through {train_through.date()}")

    hp_clean = {}
    for k, v in hp.items():
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        hp_clean[k] = v

    model = make_model(model_name, **hp_clean)
    model.fit(X_clean, y_clean)
    return model


def predict_top_k(
    model,
    selected_features: list[str],
    score_date: pd.Timestamp,
    k: int,
) -> list[str]:
    X, _ = build_features_and_labels()
    X = X[selected_features]
    try:
        X_score = X.xs(score_date, level="date").dropna()
    except KeyError:
        raise RuntimeError(f"No features available for {score_date.date()}")
    preds = model.predict(X_score)
    df = pd.DataFrame({"ticker": X_score.index, "pred": preds})
    df = df.sort_values("pred", ascending=False)
    return df.head(k)["ticker"].tolist()

def predict_top_and_bottom_k(
    model,
    selected_features: list[str],
    score_date: pd.Timestamp,
    k: int,
) -> tuple[list[str], list[str]]:
    """Predict ranks on score_date; return (top-k, bottom-k) tickers."""
    X, _ = build_features_and_labels()
    X = X[selected_features]
    try:
        X_score = X.xs(score_date, level="date").dropna()
    except KeyError:
        raise RuntimeError(f"No features available for {score_date.date()}")
    preds = model.predict(X_score)
    df = pd.DataFrame({"ticker": X_score.index, "pred": preds})
    df = df.sort_values("pred", ascending=False)
    longs = df.head(k)["ticker"].tolist()
    shorts = df.tail(k)["ticker"].tolist()
    return longs, shorts


def get_close_price(panel: pd.DataFrame, ticker: str, date: pd.Timestamp) -> float:
    return panel.loc[(date, ticker), "adj_close"]


def get_benchmark_series(
    ticker: str, start: pd.Timestamp, end: pd.Timestamp,
) -> pd.Series:
    bench = yf.download(
        ticker,
        start=start - pd.Timedelta(days=5),
        end=end + pd.Timedelta(days=2),
        progress=False,
        auto_adjust=True,
    )
    if isinstance(bench.columns, pd.MultiIndex):
        bench = bench.xs(ticker, level=1, axis=1)
    series = bench["Close"]
    series.index = pd.to_datetime(series.index)
    return series


# ============================================================
# Backtest engine
# ============================================================

def run_backtest(cfg: BacktestConfig) -> pd.DataFrame:
    print("=" * 60)
    print(f"BACKTEST: {cfg.start_date.date()} → "
          f"{cfg.end_date.date() if cfg.end_date else 'today'}")
    print(f"  top_k={cfg.top_k}, capital=${cfg.initial_capital:,.0f}, "
          f"tx_cost={cfg.tx_cost_bps}bps, model={cfg.model_name}, "
          f"tag='{cfg.run_tag}'")
    print("=" * 60)

    # Refresh data
    print("\n[1/6] Refreshing data...")
    download_universe()
    panel = build_clean_panel()

    # Setup
    selected = json.loads(
        (cfg.data_dir / "selected_features.json").read_text()
    )["selected_features"]
    hp = json.loads(
        (cfg.data_dir / "tuned_hyperparameters.json").read_text()
    )[cfg.model_name]

    all_dates = panel.index.get_level_values("date").unique().sort_values()
    end = cfg.end_date if cfg.end_date is not None else all_dates.max()

    rebal_dates = get_rebalance_dates(panel, cfg.start_date, end)
    print(f"\n[2/6] {len(rebal_dates)} rebalance dates: "
          f"{rebal_dates[0].date()} ... {rebal_dates[-1].date()}")

    bt_dates = all_dates[(all_dates >= rebal_dates[0]) & (all_dates <= end)]

    
    # Init books — books are dict[ticker -> shares], shares can be NEGATIVE for shorts.
    holdings = {}
    holdings_net = {}
    cash_gross = cfg.initial_capital
    cash_net = cfg.initial_capital

    trades_log: list[Trade] = []
    rebalance_log = []

    daily_value = pd.DataFrame(
        index=bt_dates,
        columns=["portfolio_gross", "portfolio_net"],
        dtype=float,
    )

    print(f"\n[3/6] Running rebalance loop...\n")

    # Daily borrow cost rate (bps/year → daily fraction)
    daily_borrow_rate = cfg.short_borrow_bps_annual / 1e4 / 252

    for i, rebal_date in enumerate(rebal_dates, 1):
        print(f"--- Rebalance {i}/{len(rebal_dates)}: {rebal_date.date()} ---")

        # Train model
        train_through_target = rebal_date - pd.Timedelta(
            days=cfg.label_horizon + cfg.retrain_buffer_days
        )
        train_through = all_dates[all_dates <= train_through_target].max()

        model = train_model_through_date(selected, hp, train_through, cfg.model_name)

        # Pick targets based on strategy mode
        if cfg.strategy_mode == "long_only":
            longs = predict_top_k(model, selected, rebal_date, cfg.top_k)
            target_longs = set(longs)
            target_shorts = set()
            print(f"    Long-only top-{cfg.top_k}: {longs}")
        elif cfg.strategy_mode == "long_short":
            longs, shorts = predict_top_and_bottom_k(
                model, selected, rebal_date, cfg.top_k
            )
            target_longs = set(longs)
            target_shorts = set(shorts)
            print(f"    Long top-{cfg.top_k}:    {longs}")
            print(f"    Short bottom-{cfg.top_k}: {shorts}")
        else:
            raise ValueError(f"unknown strategy_mode: {cfg.strategy_mode}")

        # Current state — split by side
        current_longs = {tk for tk, sh in holdings.items() if sh > 0}
        current_shorts = {tk for tk, sh in holdings.items() if sh < 0}

        # Determine actions
        longs_to_close = current_longs - target_longs
        longs_to_open = target_longs - current_longs
        shorts_to_close = current_shorts - target_shorts
        shorts_to_open = target_shorts - current_shorts
        kept_longs = current_longs & target_longs
        kept_shorts = current_shorts & target_shorts

        print(f"    Long  | keep={sorted(kept_longs)} | close={sorted(longs_to_close)} | open={sorted(longs_to_open)}")
        if cfg.strategy_mode == "long_short":
            print(f"    Short | keep={sorted(kept_shorts)} | close={sorted(shorts_to_close)} | open={sorted(shorts_to_open)}")

        # ============ GROSS BOOK ============
        # Close longs (sell)
        for ticker in sorted(longs_to_close):
            shares = holdings.pop(ticker)
            price = get_close_price(panel, ticker, rebal_date)
            cash_gross += shares * price
            trades_log.append(Trade(rebal_date, ticker, "SELL", shares, price, shares * price))

        # Close shorts (buy back)
        for ticker in sorted(shorts_to_close):
            shares = holdings.pop(ticker)  # negative number
            price = get_close_price(panel, ticker, rebal_date)
            # buying back: cash decreases by |shares| * price
            cash_gross += shares * price  # shares is negative, so this is a cost
            trades_log.append(Trade(rebal_date, ticker, "COVER", -shares, price, -shares * price))

        # Open new longs
        if longs_to_open:
            # Allocate half of capital to longs (or all if long-only)
            if cfg.strategy_mode == "long_short":
                long_capital = cfg.initial_capital / 2
                # Already-held longs use up some; new ones get the rest
                cash_for_new_longs = cash_gross - (long_capital * len(kept_longs) / cfg.top_k)
                # Simpler: just split available cash equally — works if we kept the
                # invariant that total long exposure = initial_capital / 2
                # Actually the cleanest: each new long gets long_capital / top_k
                cash_per = long_capital / cfg.top_k
            else:
                # long-only: split available cash across new longs
                cash_per = cash_gross / len(longs_to_open)

            for ticker in sorted(longs_to_open):
                price = get_close_price(panel, ticker, rebal_date)
                shares = cash_per / price
                holdings[ticker] = shares
                cash_gross -= shares * price
                trades_log.append(
                    Trade(rebal_date, ticker, "BUY", shares, price, shares * price)
                )

        # Open new shorts
        if shorts_to_open:
            short_capital = cfg.initial_capital / 2
            cash_per = short_capital / cfg.top_k
            for ticker in sorted(shorts_to_open):
                price = get_close_price(panel, ticker, rebal_date)
                shares = -(cash_per / price)  # negative shares
                holdings[ticker] = shares
                # Shorting receives cash (we record it but conservatively don't use it)
                cash_gross += -shares * price  # shares is negative → -shares is positive
                trades_log.append(
                    Trade(rebal_date, ticker, "SHORT", shares, price, -shares * price)
                )

        # ============ NET BOOK (with tx costs) ============
        # Mirror the gross logic, applying tx_cost_bps on each trade leg
        cost_factor = cfg.tx_cost_bps / 1e4 / 2  # half on each leg (buy + sell = round-trip)

        # Close longs
        for ticker in sorted(longs_to_close):
            if ticker in holdings_net:
                shares = holdings_net.pop(ticker)
                price = get_close_price(panel, ticker, rebal_date)
                cash_net += shares * price * (1 - cost_factor)

        # Close shorts (buy back)
        for ticker in sorted(shorts_to_close):
            if ticker in holdings_net:
                shares = holdings_net.pop(ticker)
                price = get_close_price(panel, ticker, rebal_date)
                # Buying back costs slightly more
                cash_net += shares * price * (1 + cost_factor)

        # Open new longs
        if longs_to_open:
            if cfg.strategy_mode == "long_short":
                cash_per = (cfg.initial_capital / 2) / cfg.top_k
            else:
                cash_per = cash_net / len(longs_to_open)
            for ticker in sorted(longs_to_open):
                price = get_close_price(panel, ticker, rebal_date)
                shares_net = (cash_per / price) * (1 - cost_factor)
                holdings_net[ticker] = shares_net
                cash_net -= cash_per

        # Open new shorts
        if shorts_to_open:
            cash_per = (cfg.initial_capital / 2) / cfg.top_k
            for ticker in sorted(shorts_to_open):
                price = get_close_price(panel, ticker, rebal_date)
                shares_net = -(cash_per / price) * (1 + cost_factor)
                holdings_net[ticker] = shares_net
                cash_net += -shares_net * price * (1 - cost_factor)

        rebalance_log.append({
            "rebalance_date": rebal_date,
            "holdings": dict(holdings),
            "cash_gross": cash_gross,
        })

        # ============ MARK-TO-MARKET ============
        # Daily borrow cost on shorts (gross + net both pay it)
        next_rebal = rebal_dates[i] if i < len(rebal_dates) else end + pd.Timedelta(days=1)
        period_dates = bt_dates[(bt_dates >= rebal_date) & (bt_dates < next_rebal)]
        prev_d = None
        for d in period_dates:
            # Apply daily borrow cost on shorts (charged on the absolute value of short positions)
            if prev_d is not None:
                short_value_gross = sum(
                    abs(sh) * panel.loc[(d, tk), "adj_close"]
                    for tk, sh in holdings.items() if sh < 0
                )
                short_value_net = sum(
                    abs(sh) * panel.loc[(d, tk), "adj_close"]
                    for tk, sh in holdings_net.items() if sh < 0
                )
                cash_gross -= short_value_gross * daily_borrow_rate
                cash_net -= short_value_net * daily_borrow_rate

            v_g = cash_gross + sum(
                sh * panel.loc[(d, tk), "adj_close"] for tk, sh in holdings.items()
            )
            v_n = cash_net + sum(
                sh * panel.loc[(d, tk), "adj_close"] for tk, sh in holdings_net.items()
            )
            daily_value.loc[d, "portfolio_gross"] = v_g
            daily_value.loc[d, "portfolio_net"] = v_n
            prev_d = d

    # Benchmark
    print(f"\n[4/6] Computing {cfg.benchmark_ticker} benchmark...")
    bench = get_benchmark_series(cfg.benchmark_ticker, rebal_dates[0], end)
    bench_aligned = bench.reindex(daily_value.index, method="ffill")
    bench_shares = cfg.initial_capital / bench_aligned.iloc[0]
    daily_value[f"benchmark_{cfg.benchmark_ticker.lower()}"] = bench_shares * bench_aligned

    # Save data
    print(f"\n[5/6] Saving outputs...")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)

    base = f"backtest_{cfg.run_tag}"
    daily_path = cfg.data_dir / f"{base}_daily.csv"
    daily_value.to_csv(daily_path)
    print(f"    Wrote {daily_path}")

    trades_df = pd.DataFrame([
        {"date": t.date, "ticker": t.ticker, "side": t.side,
         "shares": t.shares, "price": t.price, "notional": t.notional}
        for t in trades_log
    ])
    trades_path = cfg.data_dir / f"{base}_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"    Wrote {trades_path}")

    holdings_df = pd.DataFrame([
        {"date": r["rebalance_date"],
         "tickers": ",".join(sorted(r["holdings"].keys()))}
        for r in rebalance_log
    ])
    holdings_path = cfg.data_dir / f"{base}_holdings.csv"
    holdings_df.to_csv(holdings_path, index=False)
    print(f"    Wrote {holdings_path}")

    # Charts
    print(f"\n[6/6] Generating charts...")
    plot_equity_curve(daily_value, rebal_dates, cfg)
    plot_monthly_pnl(daily_value, rebal_dates, cfg)
    plot_holdings_grid(rebalance_log, rebal_dates, cfg)

    # Summary
    final_g = daily_value["portfolio_gross"].iloc[-1]
    final_n = daily_value["portfolio_net"].iloc[-1]
    final_b = daily_value[f"benchmark_{cfg.benchmark_ticker.lower()}"].iloc[-1]

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Window:               {rebal_dates[0].date()} → {daily_value.index[-1].date()}")
    print(f"Rebalances:           {len(rebal_dates)}")
    print(f"Initial:              ${cfg.initial_capital:>12,.2f}")
    print(f"Final (gross):        ${final_g:>12,.2f}  "
          f"({(final_g/cfg.initial_capital - 1) * 100:+.2f}%)")
    print(f"Final (net {cfg.tx_cost_bps:>2.0f}bps):    ${final_n:>12,.2f}  "
          f"({(final_n/cfg.initial_capital - 1) * 100:+.2f}%)")
    print(f"{cfg.benchmark_ticker} benchmark:        ${final_b:>12,.2f}  "
          f"({(final_b/cfg.initial_capital - 1) * 100:+.2f}%)")
    print(f"Excess vs {cfg.benchmark_ticker} (gross): "
          f"{((final_g - final_b) / cfg.initial_capital) * 100:+.2f}%")
    print(f"Excess vs {cfg.benchmark_ticker} (net):   "
          f"{((final_n - final_b) / cfg.initial_capital) * 100:+.2f}%")

    return daily_value


# ============================================================
# Plotting
# ============================================================

def plot_equity_curve(daily: pd.DataFrame, rebal: list, cfg: BacktestConfig):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    bench_col = f"benchmark_{cfg.benchmark_ticker.lower()}"

    ax.plot(daily.index, daily["portfolio_gross"],
            color="#2563eb", linewidth=1.8, label="Strategy (gross)")
    ax.plot(daily.index, daily["portfolio_net"],
            color="#2563eb", linewidth=1.0, linestyle="--",
            alpha=0.7, label=f"Strategy ({cfg.tx_cost_bps:.0f}bps cost)")
    ax.plot(daily.index, daily[bench_col],
            color="#888", linewidth=1.5, label=f"{cfg.benchmark_ticker} buy & hold")

    for d in rebal:
        ax.axvline(d, color="#ddd", linewidth=0.6, zorder=0)
    ax.axhline(cfg.initial_capital, color="#000", linewidth=0.5, linestyle=":")

    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(
        f"${cfg.initial_capital:,.0f} invested using sp100-rank Top-{cfg.top_k} strategy\n"
        f"vs {cfg.benchmark_ticker} buy-and-hold, "
        f"{rebal[0].date()} → {daily.index[-1].date()}"
    )
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    out = cfg.fig_dir / f"backtest_{cfg.run_tag}_equity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Wrote {out}")
    plt.close()


def plot_monthly_pnl(daily: pd.DataFrame, rebal: list, cfg: BacktestConfig):
    bench_col = f"benchmark_{cfg.benchmark_ticker.lower()}"
    end = daily.index[-1]
    period_ends = list(rebal[1:]) + [end]

    pnl_strat, pnl_bench, labels = [], [], []
    for i, (start, period_end) in enumerate(zip(rebal, period_ends)):
        if i == len(rebal) - 1:
            actual_end = end
        else:
            actual_end = daily.index[daily.index < period_end].max()

        v0 = daily.loc[start, "portfolio_gross"]
        v1 = daily.loc[actual_end, "portfolio_gross"]
        b0 = daily.loc[start, bench_col]
        b1 = daily.loc[actual_end, bench_col]

        pnl_strat.append((v1 - v0) / v0 * 100)
        pnl_bench.append((b1 - b0) / b0 * 100)
        labels.append(start.strftime("%b\n%Y"))

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.7), 5.5))
    bars1 = ax.bar(x - width/2, pnl_strat, width,
                   label="Strategy (gross)", color="#2563eb")
    bars2 = ax.bar(x + width/2, pnl_bench, width,
                   label=cfg.benchmark_ticker, color="#aaa")
    for bar, p in zip(bars1, pnl_strat):
        if p < 0:
            bar.set_color("#dc2626")

    ax.axhline(0, color="#000", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Period Return (%)")
    ax.set_title(f"Monthly P&L: Strategy vs {cfg.benchmark_ticker}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = cfg.fig_dir / f"backtest_{cfg.run_tag}_monthly_pnl.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Wrote {out}")
    plt.close()


def plot_holdings_grid(rebalance_log, rebal: list, cfg: BacktestConfig):
    all_tickers = sorted(set(
        tk for r in rebalance_log for tk in r["holdings"].keys()
    ))
    labels = [d.strftime("%b %Y") for d in rebal]

    grid = np.zeros((len(all_tickers), len(rebal)))
    for col, r in enumerate(rebalance_log):
        for ticker in r["holdings"]:
            row = all_tickers.index(ticker)
            grid[row, col] = 1

    fig, ax = plt.subplots(figsize=(
        max(8, len(rebal) * 0.5),
        max(4, len(all_tickers) * 0.32),
    ))
    ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_tickers)))
    ax.set_yticklabels(all_tickers, fontsize=9)

    for i in range(len(all_tickers)):
        for j in range(len(rebal)):
            if grid[i, j] == 1:
                ax.text(j, i, "●", ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")

    ax.set_title(f"Holdings over time (Top-{cfg.top_k} each month)")
    plt.tight_layout()
    out = cfg.fig_dir / f"backtest_{cfg.run_tag}_holdings.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"    Wrote {out}")
    plt.close()


# ============================================================
# CLI
# ============================================================

def parse_args() -> BacktestConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=str, default="2024-11-01",
                   help="backtest start date (YYYY-MM-DD)")
    p.add_argument("--end", type=str, default=None,
                   help="backtest end date (YYYY-MM-DD); default = today")
    p.add_argument("--top-k", type=int, default=5,
                   help="number of stocks to hold")
    p.add_argument("--capital", type=float, default=10_000.0,
                   help="initial capital")
    p.add_argument("--tx-cost-bps", type=float, default=5.0,
                   help="round-trip transaction cost in bps")
    p.add_argument("--model", type=str, default="rf",
                   choices=["rf", "xgb", "lgb", "linear"],
                   help="which model to retrain monthly")
    p.add_argument("--benchmark", type=str, default="SPY",
                   help="benchmark ticker")
    p.add_argument("--tag", type=str, default="default",
                   help="output filename suffix (e.g. 'top10' or '6mo')")
    p.add_argument("--mode", type=str, default="long_only",
                   choices=["long_only", "long_short"],
                   help="strategy mode")
    args = p.parse_args()

    return BacktestConfig(
        start_date=pd.Timestamp(args.start),
        end_date=pd.Timestamp(args.end) if args.end else None,
        strategy_mode=args.mode,
        top_k=args.top_k,
        initial_capital=args.capital,
        tx_cost_bps=args.tx_cost_bps,
        model_name=args.model,
        benchmark_ticker=args.benchmark,
        run_tag=args.tag,
    )


def main() -> int:
    cfg = parse_args()
    try:
        run_backtest(cfg)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
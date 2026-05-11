# src/sp100rank/features/technical.py

import pandas as pd
import numpy as np


def momentum(close: pd.Series, window: int = 60) -> pd.Series:
    return close.pct_change(periods=window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_value = 100.0 - 100.0 / (1.0 + rs)

    no_losses = (avg_loss == 0) & avg_gain.notna()
    rsi_value = rsi_value.where(~no_losses, 100.0)

    return rsi_value


def macd_signal_line(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=True, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=True, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line.ewm(span=signal, adjust=True, min_periods=signal).mean()

def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    avg = volume.rolling(window=window, min_periods=window).mean()
    return volume / avg.replace(0, np.nan)


def pct_52w_high(close: pd.Series, window: int = 252) -> pd.Series:
    rolling_high = close.rolling(window=window, min_periods=window).max()
    return close / rolling_high


def drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    rolling_high = close.rolling(window=window, min_periods=window).max()
    return (close / rolling_high) - 1.0

def short_term_reversal(close: pd.Series, window: int = 5) -> pd.Series:
    return -close.pct_change(periods=window)


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    # ddof=0 = population std (consistent with the original Bollinger
    # 1980s definition). ddof=1 (sample std) would inflate slightly
    # at the start of the window. Difference is negligible at N=20
    # but matches the textbook formula.
    return (close - sma) / (2.0 * std.replace(0, np.nan))


def mom_12_1(close: pd.Series) -> pd.Series:
    """12-month momentum, skipping the most recent 1 month.

    Formula: close_{t-21} / close_{t-252} - 1

    The "12-1" name: 12 months total lookback, but we skip the
    most-recent 1 month. Why skip? Because of the Jegadeesh (1990)
    short-term reversal effect — the most recent month's return is
    a NEGATIVE predictor at short horizons. Including it dilutes
    the medium-horizon momentum signal. Skipping it isolates the
    "winners keep winning" effect from the "winners temporarily
    pause" effect.

    This is the GOLD STANDARD momentum specification in academic
    asset pricing — used in Fama-French momentum factor (UMD),
    Asness-Moskowitz-Pedersen (2013) "Value and Momentum
    Everywhere," and most empirical asset-pricing studies since.

    Causal: .shift(N) brings PAST values into the current row.
    .shift(21) at row t = close[t-21]. .shift(252) at t = close[t-252].
    Both are pure backward-looking.
    """
    # close shifted by 21 days = "close from 21 days ago"
    close_t_minus_21  = close.shift(21)
    # close shifted by 252 days = "close from 252 days ago"
    close_t_minus_252 = close.shift(252)

    # The momentum is the return from t-252 to t-21.
    return (close_t_minus_21 / close_t_minus_252) - 1.0

def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    daily_ret = close.pct_change()
    return daily_ret.rolling(window=window, min_periods=window).std(ddof=0)


def log_dollar_volume(
    close: pd.Series,
    volume: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Log of N-day average dollar trading volume.

    Formula: log(mean(close * volume, last N days))

    Why dollar volume not share volume: a $5 stock trading 10M
    shares ($50M) is much less liquid than a $500 stock trading 1M
    shares ($500M), even though share volume is 10x higher. Dollar
    volume normalizes across price levels and is the standard
    liquidity proxy in cross-sectional studies.

    Why log: dollar volumes span 5+ orders of magnitude across the
    universe (small caps ~$10M/day, mega caps ~$10B/day). Trees
    handle this fine without log, but log compresses the range,
    making the feature numerically well-behaved for the linear
    baseline model too.

    Causal: rolling mean over t-N+1 through t.

    Reference: Amihud (2002) "Illiquidity and stock returns" —
    illiquid stocks earn higher returns as compensation for
    illiquidity risk. Dollar volume is the inverse-illiquidity
    proxy.
    """
    dollar_vol = close * volume
    avg = dollar_vol.rolling(window=window, min_periods=window).mean()
    return np.log1p(avg)


def beta_to_market(
    close: pd.Series,
    market_close: pd.Series,
    window: int = 60,
) -> pd.Series:
    stock_ret  = close.pct_change()
    market_ret = market_close.pct_change()

    cov = stock_ret.rolling(window=window, min_periods=window).cov(market_ret)
    var = market_ret.rolling(window=window, min_periods=window).var(ddof=0)

    return cov / var.replace(0, np.nan)





def compute_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()

    if "^GSPC" in prices.index.get_level_values("ticker"):
        market_close = (
            prices.xs("^GSPC", level="ticker")["adj_close"]
            .sort_index()
        )
    else:
        market_close = None

    out = (
        prices.groupby(level="ticker", group_keys=False)
              .apply(_features_for_one_ticker,
                     market_close=market_close,
                     include_groups=False)
    )
    return out

def cross_sectional_rank_normalize(features: pd.DataFrame) -> pd.DataFrame:
    # groupby on the 'date' level applies the rank computation to each
    # date's cross-section independently. method='average' handles ties
    # symmetrically; pct=True normalizes to [0, 1].
    return (
        features.groupby(level="date")
                .rank(method="average", pct=True)
    )


def _features_for_one_ticker(
    df: pd.DataFrame,
    market_close: pd.Series | None = None,
) -> pd.DataFrame:
    close  = df["adj_close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    out = pd.DataFrame(index=df.index)

    # === Batch 1 — Original momentum/oscillator features ===
    out["mom_60"]      = momentum(close, window=60)
    out["rsi_14"]      = rsi(close, window=14)
    out["macd_signal"] = macd_signal_line(close)

    # === Batch 2 — Volume + position features ===
    out["vol_ratio_20"]   = volume_ratio(volume, window=20)
    out["pct_52w_high"]   = pct_52w_high(close, window=252)
    out["drawdown_60"]    = drawdown(close, window=60)

    # === Batch 3 — New momentum/reversal features ===
    out["rev_5"]          = short_term_reversal(close, window=5)
    out["bb_position_20"] = bollinger_position(close, window=20)
    out["mom_12_1"]       = mom_12_1(close)

    # === Batch 4 — Risk + liquidity features ===
    out["realized_vol_20"] = realized_vol(close, window=20)
    out["log_dollar_vol_60"] = log_dollar_volume(close, volume, window=60)

    # Beta requires market data. Align the market series to this
    # ticker's date index — tickers may have different trading
    # calendars in edge cases (very old data, foreign halts).
    # .reindex() with the ticker's date index drops market dates
    # the ticker doesn't have, fills missing market dates with NaN.
    if market_close is not None:
        ticker_dates = df.index.get_level_values("date")
        market_aligned = market_close.reindex(ticker_dates)
        out["beta_to_spx_60"] = beta_to_market(
            close.reset_index(level="ticker", drop=True),
            market_aligned,
            window=60,
        ).values
    else:
        out["beta_to_spx_60"] = np.nan

    return out
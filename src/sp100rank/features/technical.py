# src/sp100rank/features/technical.py
"""
Technical features for cross-sectional return prediction.

Each feature is implemented as a function that takes a single ticker's
time series and returns a Series. The top-level `compute_all_features`
applies them all per-ticker via groupby.

CAUSAL CONTRACT — read this before adding features:
  Every feature value at date t depends ONLY on data through close
  of t. No `.shift(-N)`, no centered rolling windows, no `.expanding()`
  reaching past t. The no-lookahead test in tests/ enforces this.

  If you need future data, you're writing a LABEL, not a feature.
  Labels go in labels.py.
"""

import pandas as pd
import numpy as np

# === Feature functions ===
#
# Each function takes a single ticker's price (or volume) Series and
# returns a Series of the same length with feature values. NaNs at
# the start are expected — they're warmup periods where there isn't
# enough history yet. Downstream code drops rows with any NaN.


def momentum(close: pd.Series, window: int = 60) -> pd.Series:
    """N-day momentum = pct change over a trailing window.

    Formula: (close_t / close_{t-N}) - 1

    Why this measures momentum: a stock that rose 20% over the last
    60 days has positive momentum; one that fell 20% has negative.
    Captures the "trend continuation" effect documented since
    Jegadeesh & Titman (1993): past winners tend to keep winning
    over horizons of 1-12 months.

    Causal: pct_change(N) at row t uses only close[t] and close[t-N].
    """
    return close.pct_change(periods=window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index, Wilder's smoothing variant.

    Formula:
        gain_t = max(close_t - close_{t-1}, 0)
        loss_t = max(close_{t-1} - close_t, 0)
        avg_gain_t = EMA(gain, alpha=1/window)   # Wilder
        avg_loss_t = EMA(loss, alpha=1/window)   # Wilder
        RS = avg_gain / avg_loss
        RSI = 100 - 100 / (1 + RS)

    Range: 0-100. Conventionally >70 = "overbought," <30 = "oversold."

    Causal: .diff() and .ewm() both use only past data.

    Why ewm with adjust=False, alpha=1/window:
      - Wilder's original 1978 RSI uses a recursive smoother with
        alpha=1/N. pandas's ewm with adjust=False matches that
        formulation exactly. adjust=True gives a different (unbiased
        exponential mean) which is NOT the standard RSI definition.
      - .rolling(window).mean() would give the SMA-RSI variant, also
        different. Wilder is more standard in academic and TA-Lib
        contexts; we use it for compatibility.
    """
    delta = close.diff()
    # .clip(lower=0) returns max(x, 0) elementwise, vectorized.
    # -delta.clip(upper=0) gives the loss leg as positive numbers.
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    # Avoid div-by-zero. When avg_loss == 0 (a strong uptrend with
    # no losses in window), RSI is conventionally 100.
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.fillna(100.0)
    return out


def macd_signal_line(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """MACD signal line — the EMA of MACD line.

    Formula:
        ema_fast = EMA(close, span=fast)
        ema_slow = EMA(close, span=slow)
        macd_line = ema_fast - ema_slow
        signal = EMA(macd_line, span=signal)

    Returns the SIGNAL LINE (the smoothed MACD), not the raw MACD
    line. We chose this because the signal line is less noisy and
    more directly relates to trade signals in the textbook MACD
    interpretation (crossovers).

    Causal: all .ewm() calls use only past data.

    Why adjust=True here (different from RSI above):
      Standard MACD uses pandas's "regular" EMA, which is unbiased
      with adjust=True. There's no Wilder analog for MACD because
      MACD wasn't defined recursively.
    """
    ema_fast = close.ewm(span=fast, adjust=True, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=True, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    return macd_line.ewm(span=signal, adjust=True, min_periods=signal).mean()

def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume relative to its trailing N-day average.

    Formula: volume_t / mean(volume_{t-N+1 ... t})

    Note: window=20 means the rolling mean INCLUDES today's volume.
    That's deliberate. We're asking "is today's volume notable
    relative to recent history including today?" not "is today's
    volume notable relative to history before today?" Both are
    causal — they only use data up to t. Conventional version uses
    inclusive window.

    Why this matters: a stock making a price move on 3x normal
    volume is conveying conviction; the same move on half normal
    volume might be noise. Volume ratio captures this.

    Causal: .rolling(N) at row t looks at rows t-N+1 through t,
    never beyond t.
    """
    avg = volume.rolling(window=window, min_periods=window).mean()
    # Avoid division by zero on the rare zero-volume days that survive
    # cleaning (e.g., index rows where volume is meaningless).
    return volume / avg.replace(0, np.nan)


def pct_52w_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Current price as a fraction of its 52-week trailing high.

    Formula: close_t / max(close_{t-251 ... t})

    Range: (0, 1]. A value of 1.0 means we're AT the 52-week high.
    A value of 0.7 means we're 30% below it.

    Why this works as a feature: the "52-week high" is a salient
    psychological anchor for traders. Stocks within a few percent of
    their 52-week high tend to attract buying; stocks far below
    their high are often in down-trends (George & Hwang 2004,
    Journal of Finance — "The 52-Week High and Momentum Investing").

    252 trading days ≈ 1 calendar year. Inclusive window is the
    conventional definition.

    Causal: rolling max at t uses only past + current data.
    """
    rolling_high = close.rolling(window=window, min_periods=window).max()
    return close / rolling_high


def drawdown(close: pd.Series, window: int = 60) -> pd.Series:
    """Current drawdown from N-day trailing high.

    Formula: (close_t / max(close_{t-N+1 ... t})) - 1

    Range: [-1, 0]. Always non-positive. -0.10 means we're 10% below
    the trailing high; 0 means we're at the high.

    Different from pct_52w_high because:
      - Shorter window (60 days vs 252) captures shorter-term pain
      - Subtracts 1 so 0 == "at the high" (sign-natural for the
        downside-only interpretation)
      - Two complementary features rather than redundant ones —
        a stock can be near its 52-week high while in a 60-day
        drawdown (or vice versa), and that combination is
        informative.

    Causal: same as pct_52w_high.
    """
    rolling_high = close.rolling(window=window, min_periods=window).max()
    return (close / rolling_high) - 1.0

def short_term_reversal(close: pd.Series, window: int = 5) -> pd.Series:
    """N-day reversal signal: short-window momentum, sign-flipped.

    Formula: -(close_t / close_{t-N} - 1)
                 ^
                 leading minus: a stock that just rose 3% has reversal
                 = -3% (i.e., we expect it to give some back).

    Why sign-flipped: at SHORT horizons (1-5 days), returns
    mean-revert. Stocks that ran up tend to dip; stocks that fell
    tend to bounce. This is the OPPOSITE of medium-horizon momentum
    (60 days, where winners keep winning). The sign flip makes the
    feature directly comparable to the momentum features — high
    values predict outperformance for both.

    Reference: Jegadeesh (1990, J. of Finance — "Evidence of
    Predictable Behavior of Security Returns") established 1-month
    reversal. The 5-day version captures even shorter mean-reversion.

    Causal: pct_change at t uses only close[t] and close[t-N].
    """
    return -close.pct_change(periods=window)


def bollinger_position(close: pd.Series, window: int = 20) -> pd.Series:
    """Position within Bollinger Bands.

    Formula: (close_t - SMA_N(close)) / (2 * std_N(close))

    Range: typically ~[-1, +1] when inside the bands; can exceed in
    strong moves. Value of 0 = at the center (mean). +1 = at the
    upper band (2 std above mean). -1 = at the lower band.

    Why this matters: classical Bollinger Band trading strategy
    interprets prices outside the bands as "stretched" and likely
    to revert. As a feature, this gives the model a normalized
    measure of "how unusual is the current price relative to recent
    typical prices for THIS stock?"

    Subtle point on standardization: each ticker is standardized
    against ITS OWN history, not the cross-section. A high-volatility
    stock and a low-volatility stock both produce values in roughly
    the same range, making the feature cross-sectionally comparable.

    Causal: rolling mean and rolling std at t use rows ≤ t only.
    """
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
    """N-day realized volatility — std of daily returns over a window.

    Formula:
        ret_t = close_t / close_{t-1} - 1
        rv_t  = std(ret_{t-N+1 ... t})

    Why this is a feature: low-volatility stocks have historically
    delivered higher RISK-ADJUSTED returns than high-vol stocks (the
    "low-volatility anomaly," Frazzini & Pedersen 2014, "Betting
    Against Beta"). Including realized vol lets the model exploit
    this — it can rank vol differently across stocks.

    We don't annualize. The model doesn't care about units; raw
    daily-return std works fine and is one less multiplication to
    track. If you wanted annualized vol, multiply by sqrt(252).

    Causal: pct_change at t uses close[t] and close[t-1]; rolling
    std at t uses returns from t-N+1 through t.
    """
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
    # log1p instead of log to be safe when avg is exactly zero (rare,
    # but possible on halt days). log1p(x) = log(1+x), behaves
    # smoothly near zero. Effect on values: negligible (we add 1 to
    # numbers in the millions, the +1 is rounding error). Effect on
    # robustness: avoids -inf.
    return np.log1p(avg)


def beta_to_market(
    close: pd.Series,
    market_close: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling N-day market beta via OLS slope.

    Formula:
        ret_t        = close.pct_change()
        market_ret_t = market_close.pct_change()
        beta_t       = cov(ret, market_ret) / var(market_ret)
                       (computed over t-N+1 ... t)

    Beta interpretation: how much the stock moves when the market
    moves, on average. Beta=1 means moves with the market 1-for-1.
    Beta=0.5 means defensive (utilities, staples). Beta=1.5 means
    aggressive (tech, financials). Beta is a structural feature
    that varies slowly within a stock but differs greatly across
    the cross-section.

    Why a feature: high-beta stocks earn higher gross returns (CAPM)
    but lower risk-adjusted returns (the "betting against beta"
    anomaly). Knowing each stock's beta lets the model rank stocks
    of different risk profiles appropriately.

    Causal: rolling cov and var at t use only returns through t.

    Important: this function REQUIRES market_close to be provided
    aligned to the same dates as close (same DatetimeIndex). The
    caller (`_features_for_one_ticker`) handles that alignment.
    """
    stock_ret  = close.pct_change()
    market_ret = market_close.pct_change()

    # Rolling covariance: cov(stock_ret, market_ret) over the window.
    # pandas's .rolling().cov() computes pairwise cov when given
    # another Series.
    cov = stock_ret.rolling(window=window, min_periods=window).cov(market_ret)
    var = market_ret.rolling(window=window, min_periods=window).var(ddof=0)

    return cov / var.replace(0, np.nan)





def compute_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Apply all features per ticker. Returns one row per (date, ticker).

    The panel must contain ^GSPC (the market proxy) — beta_to_spx_60
    requires it. If ^GSPC isn't present, beta will be NaN everywhere
    but other features still compute correctly.
    """
    prices = prices.sort_index()

    # Extract market close series, indexed by date only. We pass this
    # into each ticker's feature computation so beta has access to
    # the market alongside the stock's own data.
    #
    # If ^GSPC isn't in the panel (e.g., synthetic test data), we use
    # a None placeholder; the per-ticker function handles this case.
    if "^GSPC" in prices.index.get_level_values("ticker"):
        market_close = (
            prices.xs("^GSPC", level="ticker")["adj_close"]
            .sort_index()
        )
    else:
        market_close = None

    # group_keys=False keeps the output index aligned with input.
    # We pass market_close as a kwarg to apply, which forwards it to
    # _features_for_one_ticker. include_groups=False is the pandas
    # 3.x way to say "don't pass the grouping column inside each
    # group's frame" (it's already in the index).
    out = (
        prices.groupby(level="ticker", group_keys=False)
              .apply(_features_for_one_ticker,
                     market_close=market_close,
                     include_groups=False)
    )
    return out


def _features_for_one_ticker(
    df: pd.DataFrame,
    market_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute features for a single ticker's time series.

    Adds one column per feature. The output's index matches the input
    so groupby().apply() reassembles correctly into a panel.

    `market_close` is the broad-market index series (^GSPC) used by
    beta_to_spx_60. When None (e.g., in tests with no market data),
    beta is filled with NaN. Other features are unaffected.

    All feature functions defined in this module are CAUSAL by
    contract — they use only data through the row's own date.
    Enforced by tests/test_features_no_lookahead.py.
    """
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
        # market_aligned now has the same length as df, indexed by
        # date only (not MultiIndex). We pass it positionally; the
        # beta function uses .pct_change() and .rolling() which work
        # on plain Series.
        out["beta_to_spx_60"] = beta_to_market(
            close.reset_index(level="ticker", drop=True),
            market_aligned,
            window=60,
        ).values  # .values to align by position, not index.
    else:
        # No market data — fill with NaN. Tests use this path.
        out["beta_to_spx_60"] = np.nan

    return out
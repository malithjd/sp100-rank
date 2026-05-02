# Architecture Decision Records

This file logs significant decisions made during the project, in
chronological order. Each entry has the same structure:

  ## ADR-NNN: Title (YYYY-MM-DD)
  **Decision**: What we decided.
  **Reasoning**: Why.
  **Trade-offs**: What we gave up.
  **Alternatives considered**: What else we looked at.

Add to this file every time you make a choice that future-you would
reasonably want to know about. The writeup will draw heavily from here.

---

## ADR-001: Real Python package vs. Colab notebook (2026-04-30)
**Decision**: Build as a `src/`-layout Python package from day one.
Notebooks reserved for EDA only.

**Reasoning**: Task 5 (GitHub Actions automation) requires `.py`
modules invokable from a YAML workflow. Notebooks don't schedule or
diff cleanly. Starting in Colab and migrating later is a known source
of bugs at the worst possible time (week-of-deadline).

**Trade-offs**: Slightly slower iteration speed for one-off exploration
(though VS Code's native Jupyter kernel mitigates this).

**Alternatives considered**: Colab-first then port; flat-layout package
without `src/`. Rejected the latter because flat layouts allow
accidental imports from CWD that pass locally and break in CI.

---

## ADR-002: Walk-forward CV over CPCV (2026-04-30)
**Decision**: Walk-forward cross-validation with 20-day embargo as the
primary evaluation scheme. Combinatorial Purged CV (CPCV) reserved as
optional robustness check if buffer time permits.

**Reasoning**: (1) Walk-forward mirrors the deployment pipeline (Task
5), where the model is scored against a strictly forward-evolving data
feed. (2) Proposal scope commits to walk-forward.

**Trade-offs**: Arian, Norouzi & Seco (2024, Knowledge-Based Systems)
demonstrate CPCV produces lower Probability of Backtest Overfitting
than walk-forward on synthetic SPX data; we accept the higher PBO risk
in exchange for pipeline alignment, and acknowledge it explicitly.

**Alternatives considered**: CPCV (kept as stretch goal); standard
k-fold (rejected — leaks future into past); single train/test split
(rejected — no fold variance, no ICIR estimate).

---

## ADR-003: Realistic execution alignment (2026-04-30)
**Decision**: Features at date `t` use OHLCV through close of `t`.
Label at date `t` is the return from close of `t+1` to close of `t+21`
(20 trading days held). Implements a 1-day execution lag.

**Reasoning**: The naive same-bar setup uses `close_{t+h}/close_t`,
which implicitly trades AT today's close using TODAY's close as a
feature — same-bar leakage. Inflates IC by ~0.01–0.02 silently.

**Trade-offs**: Removes ~1 day of theoretical edge per holding period
versus the same-bar baseline. Honest IC numbers in exchange.

**Alternatives considered**: Same-bar trading (rejected — leakage);
open-to-open trading (rejected — yfinance opens are unreliable for
some illiquid names; close-to-close is cleaner).

---

## ADR-004: Feature pre-selection via gain importance on Fold 1 train (2026-04-30)
**Decision**: Compute 12 candidate features. Fit a single LightGBM on
Fold 1 training data only, rank by gain importance, retain the top 8.
Use this fixed feature set across all 4 models and all 5 folds.

**Reasoning**: 12 → 8 is a meaningful selection step (~33% drop) that
prevents the curse of dimensionality without being so aggressive as to
discard signal. Selection on Fold 1 train data only — no information
leak from later folds. Gain importance is faster and less overfit-prone
than SHAP for selection (we use SHAP for INTERPRETATION later).

**Trade-offs**: Fold 1 IC is slightly optimistic (the features were
chosen using its training data). Fold 2–5 IC is unbiased. We report
Fold 1 in the table and note this caveat.

**Alternatives considered**: All 12 features (rejected — adds noise,
hurts SHAP interpretability); Lasso for feature selection (rejected
— linear method, biased against the nonlinear interactions GBMs
exploit); held-out 2018 sample for selection (rejected — eats too
much early data given our small training history).

---

## ADR-005: Universe construction (2026-04-30)
**Decision**: Fixed hand-curated universe of 100 large-cap U.S.
equities, plus the ^GSPC index as a market proxy for the
beta_to_spx_60 feature (excluded from the prediction cross-section).

The 100 names are large-cap, liquid stocks broadly drawn from the
S&P 100 / S&P 500 top-tier, but the list is NOT identical to the
index-defined S&P 100 at any specific date. Throughout the report
we refer to it as "100 large-cap U.S. equities," not S&P 100.
For data infrastructure purposes the panel contains 101 tickers (100 equities + ^GSPC), 
but the prediction universe — the set of names the model ranks against each other — is 
the 100 equities only.

**Reasoning**: Index-membership reconstruction at point-in-time is
out of scope for the time budget. A hand-curated stable universe
spanning 2018-2026 (a) avoids the survivorship bias inherent to
"current S&P 100 backtested historically," (b) ensures all names
have full 2018-2026 history available from yfinance, (c) sidesteps
ticker-rename quirks (e.g., FISV → FI in 2023).

The universe was built by starting from a candidate list of 104
large-caps and dropping four:
  - FISV (renamed to FI in 2023; complicates split history)
  - GOOG (kept GOOGL — same entity, redundant cross-sectional weight)
  - EL   (extreme idiosyncratic drawdown 2022-2024 distorts tails)
  - PSA  (REIT with atypical total-return profile)

**Trade-offs**: The universe is hand-picked and not reproducible
from a public index definition. We document the construction
explicitly and list every ticker in src/sp100rank/data/universe.py
to compensate. Hand-picking introduces selection effects (we know
these names had complete history) but does not introduce forward-
looking bias, since membership was decided based on company
identity, not on past returns.

**Alternatives considered**: 
  - Live S&P 100 membership at start date — rejected (survivorship).
  - Point-in-time membership reconstruction — rejected (scope).
  - Top-100-by-current-market-cap — rejected (circular: market cap
    is a function of past returns).
  - S&P 500 with market-cap filter applied each rebalance —
    rejected (too much engineering for a 50-hour budget).

---

## ADR-006: Data cleaning thresholds (2026-04-30)
**Decision**: Apply the following cleaning steps to the raw yfinance panel before feature computation:
1. Drop equity rows with `volume == 0` (preserves indices, where volume is meaningless).
2. Forward-fill `adj_close` gaps up to 2 trading days; drop rows still NaN after.
3. Flag and drop rows with absolute one-day return > 50% (likely data errors or unhandled corporate actions).

**Reasoning**: Cleaning is intentionally conservative — we'd rather lose a few rows than carry bad data into features. Forward-fill cap of 2 days prevents fabricating multi-day signal.

**Trade-offs**: Real ~50% one-day moves do occasionally happen (M&A announcements, earnings). The 50% threshold is high enough that we expect to drop ~0–5 rows per ticker over 8 years — acceptable loss given the alternative.

**Audit results**: 209,272 input rows, 209,272 output rows, 0 dropped (zero-volume / spike / unfilled-NaN). The cleaning logic is structurally correct but had nothing to do because the universe (per ADR-005) consists of continuously-listed large-caps with no IPO gaps, splits mishandled by yfinance, or volume halts in the project window. Verified via post-clean diagnostics — see ADR-008.

---

## ADR-007: MMC → MRSH ticker rename (2026-04-30)
**Decision**: Use ticker MRSH (Marsh & McLennan) in the universe.
yfinance returns full 2018-2026 history under MRSH; MMC fails.

**Reasoning**: MMC was renamed to MRSH in January 2026. yfinance has
back-stitched the historical data under the new symbol, so a single
download under MRSH gives us the complete continuous price series
for the same economic entity. No manual stitching required.

**Trade-offs**: None — yfinance handled the rename transparently.

**Note for future**: Other ticker-rename events to watch for in this
universe over the project window: FB → META (2022, handled), FISV → FI
(2023, dropped from universe in ADR-005). If yfinance flakes on
another ticker mid-project, the diagnostic pattern (try old + new
symbol, compare row counts and date ranges) is in chat history.

---

## ADR-008: Data quality verified (2026-04-30)
**Decision**: Treat the cleaned panel as production-ready for feature
engineering. No further cleaning passes needed.

**Reasoning**: Post-clean diagnostics show:
  - 101 tickers × 2,072 trading days = 209,272 rows, all complete.
  - Zero NaNs in any OHLCV column.
  - Per-ticker row counts identical (2,072 each) — no IPO/delisting
    history gaps in the universe.
  - Top-15 largest one-day moves all attributable to documented real
    events (NFLX 2022 sub loss, META 2022 DAU shock, NVDA 2023 AI
    rally start, ORCL 2025 cloud guidance, etc.). Largest move
    35.95% (ORCL 2025-09-10), well under our 50% spike threshold.

**Trade-offs**: None — the data is what we'd hope for.

**Note**: This level of cleanliness reflects the universe construction
(ADR-005), not luck. Hand-curating to known continuously-listed
large caps avoids the data quirks of less-liquid or recently-IPO'd
names.

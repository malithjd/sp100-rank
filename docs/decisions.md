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

---

## ADR-009: Walk-forward fold configuration (2026-04-30)
**Decision**: 5 folds with 756-day initial train, 126-day test
periods, 20-day embargo, expanding train window.

Effective coverage:
- Fold 1: train 2018-01 → 2020-12, test 2021-02 → 2021-08
- Fold 2: train 2018-01 → 2021-08, test 2021-08 → 2022-03
- Fold 3: train 2018-01 → 2022-03, test 2022-03 → 2022-09
- Fold 4: train 2018-01 → 2022-09, test 2022-10 → 2023-04
- Fold 5: train 2018-01 → 2023-04, test 2023-05 → 2023-11

Hold-out (unused by walk-forward CV): 2024-01 → 2026-03 (~28 months).

**Reasoning**: 5 folds provide enough samples to compute mean IC and
ICIR with reasonable confidence; 6-month test periods are long
enough that per-fold IC isn't overly noisy; 3-year initial train
gives the model enough history to learn cross-sectional patterns.

The 2024-2026 portion of the data is deliberately unused by walk-
forward CV. It serves as a BLIND HOLD-OUT for final evaluation —
the chosen model (selected by walk-forward IC) is evaluated once
on this held-out period to produce a final unbiased number. This
is methodologically stronger than reporting only walk-forward IC
because the hold-out has not been touched at any point during
model selection or hyperparameter tuning.

**Trade-offs**: We use 60% of available data for walk-forward CV
and 30% as hold-out, with 10% lost to embargoes between folds and
between fold 5 and the hold-out start. A more aggressive setup
(more folds, shorter tests) would use more of the data but at the
cost of less stable per-fold IC.

**Alternatives considered**:
  - 4 folds with longer test windows (rejected: fewer samples for ICIR).
  - Rolling (not expanding) train windows (rejected per ADR-002).
  - Using all data in walk-forward, no hold-out (rejected: the hold-
    out is the only unbiased final evaluation we get).

**Regime coverage caveat**: The 5 test folds span Feb 2021 – Nov
2023, a period dominated by the post-COVID monetary-tightening
cycle. Each fold's TRAINING set includes regime-diverse data
(2018–2020 in particular), but the EVALUATION sets are concentrated
in one macro regime. We mitigate via:
  (1) the 2024–2026 hold-out as a genuinely out-of-regime final
      test, with planned sub-stratification (2024 H1 / 2024 H2 /
      2025 / 2026 Q1) for regime-transfer analysis;
  (2) ICIR as primary model-selection criterion, which penalizes
      models whose IC varies widely across folds.

A more aggressive fold structure (3-month tests, more folds) was
considered but rejected: shorter test windows produce unstable per-
fold IC estimates, and the stretch-goal hold-out sub-stratification
provides regime coverage at lower variance cost.

A historical extension (download data back to 2008–2015) was also
considered for broader regime coverage. Rejected on two grounds:
(a) point-in-time index membership data is unavailable, so the
universe would inherit deeper survivorship bias when reaching back
into periods like the 2008 crisis where the index-vs-current-tickers
map drifts substantially; (b) several names in the current universe
(META, ABBV, KHC, ZTS, HCA, NOW, PYPL) IPO'd after 2010, requiring
either ragged-history handling or universe substitution that erodes
the "S&P 100" framing further. Documented as future work for a
follow-on study with proper point-in-time membership.


---

## ADR-010: Phase 4 walk-forward training results (2026-04-30)
**Decision**: Random Forest is the lead candidate for downstream
analysis (SHAP, regime breakdown, portfolio simulation). All four
models are retained for the comparison table; final model selection
will be re-confirmed after Phase 5 evaluation.

**Findings**:
  | Model  | Mean IC | ICIR  |
  |--------|---------|-------|
  | Linear | +0.006  | 0.106 |
  | RF     | +0.022  | 0.579 |
  | XGB    | +0.013  | 0.457 |
  | LGB    | +0.015  | 0.432 |

  Computed across 5 walk-forward folds (Feb 2021 - Nov 2023).
  Tree models substantially outperform the linear baseline,
  validating the proposal hypothesis that non-linear feature
  interactions are needed for cross-sectional ranking.

**Reasoning**: 
- All three tree models produce IC > 2 standard errors above zero;
  linear baseline is statistically indistinguishable from zero.
- ICIR comparison ranks RF > XGB > LGB > Linear. RF's superior
  ICIR (lower IC volatility across folds) is the headline result.
- Fold-by-fold pattern shows clear regime dependence: every model
  produces strongest IC in Fold 3 (2022 bear market, when factor
  models historically work best) and weakest in Fold 5 (mid-2023
  AI rally, when concentrated mega-cap returns break factor models).
  This supports the regime-aware analysis planned for Phase 5.

**Trade-offs**: Fold 1 results are slightly optimistic per ADR-004
(features and hyperparameters were selected on Fold 1 train data).
The Fold 2-5 average — RF mean IC +0.025, ICIR 0.69 — is the
unbiased estimate.

**Hyperparameters chosen** (from Fold 1 inner train/val split):
  - Linear (Ridge): alpha = 10.0
  - Random Forest:  max_depth = 12, min_samples_leaf = 50
  - XGBoost:        max_depth = 4, learning_rate = 0.10
  - LightGBM:       num_leaves = 15, learning_rate = 0.05

  ---

## ADR-011: Phase 5 evaluation results (2026-04-30)
**Decision**: Random Forest is the lead model based on combined IC,
ICIR, regime stability, and held-out evaluation. Linear baseline
shows surprising portfolio Sharpe leadership but regime instability;
documented as a finding rather than a recommendation.

**Walk-forward IC by year**:
  | Model  | 2021    | 2022    | 2023    |
  |--------|---------|---------|---------|
  | Linear | +0.017  | +0.061  | -0.063  |
  | RF     | +0.040  | +0.045  | -0.022  |
  | XGB    | +0.031  | +0.032  | -0.026  |
  | LGB    | +0.051  | +0.021  | -0.029  |
  
  All four models lose skill in 2023. The 2023 AI rally (NVDA-led
  mega-cap concentration) breaks factor-style cross-sectional ranking.

**Regime breakdown — IC by Trend × Vol cell** (RF only):
  - bear/high_vol: +0.033 (best)
  - bull/high_vol: +0.016
  - bear/low_vol:  +0.012
  - bull/low_vol:  +0.016
  
  RF is the only model with positive IC in all 4 regime cells.

**SHAP stability**:
  - log_dollar_vol_60: ranks 1-2 across all 5 folds (stable; robust)
  - drawdown_60:       rank 8 across all 5 folds (consistently weakest)
  - mom_12_1, pct_52w_high: rank swings widely (regime-dependent)
  - macd_signal, beta_to_spx_60: moderately stable middle

**Decile long-short portfolio (32 non-overlapping 20-day periods)**:
  - Linear: Sharpe 0.92 / 0.86 / 0.79 at 0/5/10 bps
  - LGB:    Sharpe 0.72 / 0.63 / 0.54
  - RF:     Sharpe 0.52 / 0.43 / 0.35
  - XGB:    Sharpe 0.48 / 0.39 / 0.30
  
  IC and Sharpe rankings disagree: Linear has the best tails
  (largest top-decile minus bottom-decile spread) despite weakest IC.
  This reflects different aspects of model behavior — IC measures
  full-cross-section ranking, Sharpe measures tail discrimination.

**Hold-out evaluation (RF trained on 2018-2023, evaluated 2024-2026)**:
  - 2024 mean IC: -0.021 (negative — confirms regime-transfer concern)
  - 2025 mean IC: +0.017
  - 2026 Q1 IC:   +0.136 (small-sample, only 40 days)
  - Full hold-out IC: +0.008, ICIR 0.05, t-stat 1.24
  
  The hold-out cannot reject H0: alpha = 0. This is the honest
  unbiased estimate. The walk-forward IC was inflated by selection
  effects (Fold 1 hyperparameter tuning, fold 1-5 within a single
  macro regime). The 2024 negative IC is the project's most
  important caveat.

**Reasoning for RF lead despite Linear's higher Sharpe**:
  - RF has positive IC in all 4 regime cells; Linear has 2 negative cells.
  - RF's hold-out 2024 IC is also negative, but less severely than
    Linear would be (we did not compute Linear hold-out, but its
    2023 walk-forward IC of -0.063 is much worse than RF's -0.022).
  - Sharpe estimate on 32 periods has SE ~0.3; the 0.40 spread
    between Linear and RF Sharpe is barely outside one SE.
  - Regime robustness matters more for production deployment than
    a one-off Sharpe number.

**What we'd do differently in v2**: tighter regime coverage in
training (2008-2015 history with point-in-time membership), explicit
regime-conditional models, more frequent retraining cadence than
quarterly.


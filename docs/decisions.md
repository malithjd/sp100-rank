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
**Decision**: Fixed universe of 100 tickers that are S&P 100 members
as of project start. SPX index added as a 101st ticker for the
beta feature, but excluded from prediction universe.

**Reasoning**: Point-in-time S&P 100 membership is not available from
yfinance and would require Wikipedia-scraping with revision history.
Out of scope for time budget.

**Trade-offs**: Survivorship bias toward names that remained in the
index through 2018–2026. Quantitatively, IC is likely 0.005–0.015
higher than would be measured on a point-in-time panel. Directional
findings (which model wins, which features matter) should not be
materially affected.

**Alternatives considered**: Point-in-time membership reconstruction
(rejected — out of scope); current S&P 500 (rejected — too many
illiquid names dilute the cross-section); top-100-by-current-market-cap
(rejected — circular: market cap is a function of past returns).

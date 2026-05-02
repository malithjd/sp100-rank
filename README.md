# sp100-rank

Cross-sectional rank prediction for S&P 100 equities via gradient-boosted models.
ML course final project — Malith Jayasinghelage Don, 2026.

## Project status
Phase 1: scaffolding complete. See `docs/decisions.md` for design log.

## Setup

```bash
uv sync                    # creates .venv from pyproject.toml + uv.lock
uv run pytest              # runs tests
```

## Structure
src/sp100rank/   # importable package
data/          # OHLCV ingest + cleaning
features/      # technical features + cross-sectional labels
models/        # Linear / RF / XGBoost / LightGBM
eval/          # walk-forward CV + IC metrics + regime tagging
interpret/     # SHAP analysis
pipeline/      # scoring + retraining entry points
tests/           # pytest tests (no-lookahead, label sanity, fold boundaries)
notebooks/       # EDA only — never imported by .py code
docs/            # decisions.md ADR log
## License
MIT

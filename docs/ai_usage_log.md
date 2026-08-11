# AI usage log

Disclosure log of AI-assisted work on this TFM, per UNED's academic
integrity guidance on tool use. One entry per work session; append, don't
rewrite history.

## 2026-08-10 — Data pipeline (Stage 1: data gathering)

- **Tool**: Claude Code (Anthropic).
- **Scope**: Wrote `src/data_pipeline.py` (download S&P 500 `^GSPC` OHLCV
  via `yfinance`, clean/flag, compute log returns), `requirements.txt`,
  `README.md`, and the auto-generated `docs/data_provenance.md`.
- **Environment troubleshooting done by the assistant**: the dev machine's
  `yfinance` install (0.1.74, on Python 3.7.4) could not reach Yahoo's
  current API. Diagnosed and resolved by pinning `yfinance==0.2.55` and
  `multitasking==0.0.11` (both compatible with Python 3.7) — see comments
  in `requirements.txt`.
- **Human review**: outputs (raw/processed CSVs, provenance note, flagged
  rows — a 2020-03-16 COVID-crash outlier return and a 2023-05-24
  zero-volume day) were inspected and spot-checked against known market
  events before accepting.
- **Not AI-generated**: research questions, methodology decisions (walk-
  forward scheme, model list, VaR/ES approach, feature-set separation),
  and all judgment calls in `CLAUDE.md` — these are the author's, set
  before this session and given to the assistant as a brief.

## 2026-08-10 — Froze data pipeline end date; built target/split (`features.py`)

- **Tool**: Claude Code (Anthropic).
- **Scope**: Hardcoded the yfinance pull's end date to 2026-07-07 in
  `src/data_pipeline.py` for reproducibility; re-ran it (3,899 rows). Wrote
  `src/features.py`: 5-day-forward realised-vol target, 5/10/20-day trailing
  historical vol, and the `TrainTestSplit` train/test scaffold (cutoff
  2022-12-31), all built on the frozen dataset.

## 2026-08-10 — Trimmed train boundary; built `walkforward.py`

- **Tool**: Claude Code (Anthropic).
- **Scope**: `TrainTestSplit.split()` now trims the last 5 training rows
  (not 4 — the assistant checked the math against a 5-day target window and
  flagged the discrepancy) so no training target reaches past the cutoff.
  Wrote `src/walkforward.py`: expanding-window engine with a duck-typed
  `fit`/`predict` model interface and daily/weekly refit cadence, smoke-
  tested with a throwaway naive baseline (not a thesis model).

## 2026-08-10 — Econometric models (`src/models/econometric.py`)

- **Tool**: Claude Code (Anthropic).
- **Scope**: Implemented EWMA (RiskMetrics-style, hand-rolled), GARCH(1,1)
  and GJR-GARCH (both via `arch`), sharing one `annualize_5day_vol()` helper
  so all three forecasts land on the same scale as `target_rv_5d`. Ran each
  through the full daily-refit walk-forward loop over the test period
  (2023-01-03 to 2026-07-07, 879 rows) as a sanity check: no NaN forecasts,
  no `arch` convergence warnings, forecast ranges in the same ballpark as
  the target (EWMA 0.073–0.482, GARCH 0.091–0.682, GJR 0.088–0.640 vs.
  target 0.032–0.863), EWMA-vs-actual correlation 0.37 as a rough plausibility
  check. MAE/RMSE/QLIKE deferred to `evaluation.py` (not built yet).

## 2026-08-10 — First comparison table (`src/evaluation.py`)

- **Tool**: Claude Code (Anthropic).
- **Scope**: Implemented MAE/RMSE (vol scale) and QLIKE (variance scale,
  squares both series first, commented against silent "fixes"), aligning
  forecasts/actuals by dropping NaN-actual rows before any metric. Confirmed
  in code: `arch`'s variance forecast is divided by `RETURN_SCALE ** 2`
  (10,000), not 100, before annualizing. Ran EWMA/GARCH(1,1)/GJR-GARCH over
  the full daily-refit test period (874 valid rows) and saved
  `results/tables/econometric_comparison.csv`. Result: GJR-GARCH ranks best
  on all three metrics, GARCH(1,1) second, EWMA third but not far behind --
  human should read the full ranking discussion in the assistant's reply,
  not just this log line, before treating it as a conclusion.

## 2026-08-10 — Audit: nothing can implicitly re-trigger data_pipeline.py

- **Tool**: Claude Code (Anthropic).
- **Scope**: Per `docs/risks_and_roadmap.md`'s yfinance-rate-limiting risk
  item, searched the repo for anything that could re-run
  `src/data_pipeline.py` as a side effect (grepped for `data_pipeline`
  repo-wide, checked for a Makefile/notebook/shell/YAML automation script,
  a `.github/` CI config, and any `__init__.py` package-level imports).
  Found none of those, and confirmed every downstream script
  (`features.py`, `walkforward.py`, `evaluation.py`,
  `models/econometric.py`) reads `data/processed/gspc_processed.csv`
  directly rather than importing `data_pipeline`. Also confirmed
  `data_pipeline.py`'s own `main()` only runs under its
  `if __name__ == "__main__"` guard, so even a hypothetical future import
  of the module wouldn't trigger a download at import time. No code
  changes made — audit only, nothing to fix.

## 2026-08-11 — Python 3.11 migration; ML models (RF, XGBoost)

- **Tool**: Claude Code (Anthropic).
- **Scope, Part 0 (environment)**: Was on Python 3.7.4 (legacy-compatible
  pins, scikit-learn stuck at 0.21.3, xgboost not installed). A newer
  Python was available with zero setup friction (the `quant` conda env
  already had 3.11.14), so per instructions that beat pinning to legacy
  versions. Created a fresh, dedicated `.venv/` off that interpreter,
  reinstalled everything at current versions, and explicitly confirmed
  xgboost 3.2.0 actually works (a real `XGBRegressor.fit()`/`.predict()`
  call, not just an import). `requirements.txt` repinned exactly.
  Regression check: re-ran `evaluation.py` end-to-end in the new venv
  (without touching `data_pipeline.py`) and compared
  `results/tables/econometric_comparison.csv` before/after at full float
  precision -- EWMA differs at ~1e-16 (float rounding, deterministic code),
  GARCH/GJR-GARCH differ at ~1e-8 relative (optimizer landing at a
  very slightly different point under newer scipy/numpy) -- both far
  below the table's reported 5-decimal precision, not a regression.
- **Scope, Part 1**: Added `RQ1_ML_FEATURES` to `features.py` -- lagged/
  absolute/squared returns (lags 1-5) + 5/10/20-day historical vol, named
  and explicit so it can be imported rather than re-derived. Verified by
  hand: 18 columns, `lag_return_1[i] == log_return[i-1]` for every row,
  and confirmed no GARCH/EWMA-named column exists anywhere in the
  underlying dataframe (not just "unused" -- structurally absent).
- **Scope, Part 2**: `src/models/ml_models.py` -- `RandomForestModel` and
  `XGBoostModel`, both `fit(history)`/`predict()`, both with explicit
  hyperparameters (n_estimators, max_depth, etc. -- not library defaults,
  directly motivated by the sklearn default-drift found in Part 0) and a
  fixed `random_state=42`.
- **Scope, Part 3**: Ran both through the existing weekly-refit
  walk-forward harness (879 rows each, no NaN forecasts, no non-positive
  forecasts). First run produced a working `RandomForestRegressor`, `n_jobs=-1`
  triggered a benign sklearn/joblib `UserWarning` ~29,000 times (58,000
  log lines) -- not a correctness bug, but not left alone either: switched
  both models to `n_jobs=1` (removes the warning, removes repeated
  process-pool spin-up overhead, and incidentally makes XGBoost's output
  exactly reproducible -- its `n_jobs=-1` run had differed from the
  `n_jobs=1` rerun by up to ~1.4% relative on RMSE, a real
  multithreading-order effect, not noise). Comparison table renamed
  `econometric_comparison.csv` -> `results/tables/model_comparison.csv`
  (old file removed, git history keeps it) and extended to all 5 models.
  See chat for the full ranking discussion.
- **Not done here**: significance testing (Diebold-Mariano, still queued
  per `docs/risks_and_roadmap.md` until the hybrid model exists too) and
  the regime-split overfitting check that risk doc also flags as worth
  watching for RF/XGBoost -- neither was in scope for this session.

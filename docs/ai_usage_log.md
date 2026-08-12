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

## 2026-08-11 — Correction: RQ1 feature set was over-scoped (18 cols, not 6)

- **Tool**: Claude Code (Anthropic).
- **What went wrong**: the RQ1 feature set built in the entry above
  (`RQ1_ML_FEATURES`) used lags 1-5 of each of lagged/absolute/squared
  returns -- 15 lag columns, 18 total with the three `hist_vol_*`. CLAUDE.md's
  prose ("lagged/absolute/squared returns ... **only**") reads as ambiguous
  between "one lag each" and "several lags each," and the assistant picked
  the wrong reading without checking it against the finalised thesis
  methodology text, which specifies exactly six features (one lagged
  return, one absolute return, one squared return, plus the three
  historical-vol windows). The user caught the mismatch, not the assistant.
- **Fix**: `RQ1_ML_FEATURES` in `features.py` now has exactly those six
  columns; the unused multi-lag columns (lag 2-5 of each type) were removed
  from `build_features()` entirely, not just excluded from the list.
  Verified: 6 columns, `lag_return_1[i] == log_return[i-1]` still holds,
  no stale multi-lag columns remain in the dataframe.
- **Re-run**: RandomForest and XGBoost re-run through the same weekly-refit
  harness with the corrected feature set (see chat for exact hyperparameters,
  unchanged from the entry above -- only the feature set changed). Both
  sanity-checked again (no NaN/non-positive forecasts). `results/tables/
  model_comparison.csv` regenerated; econometric rows unchanged (they don't
  touch `RQ1_ML_FEATURES`), RF/XGBoost numbers changed, and the ranking
  changed too -- GJR-GARCH reclaims the RMSE lead from XGBoost, and
  RandomForest drops below GJR-GARCH on MAE and below GARCH(1,1) on QLIKE
  (it had beaten both under the over-scoped feature set). Full numbers and
  discussion in chat, not restated here.
- **Why this is worth having on record**: the over-scoped run's numbers were
  reported to the user as a real result in the previous session before the
  mismatch was caught -- worth being explicit about for the thesis's
  AI-use integrity disclosure, not just quietly fixing it and moving on.

## 2026-08-12 — VaR/ES, backtesting (Kupiec, Christoffersen, simple ES)

- **Tool**: Claude Code (Anthropic).
- **Scope, Part 0**: Confirmed (grep + directory listing) that only
  aggregated MAE/RMSE/QLIKE were persisted, never per-date forecasts.
  `evaluation.py` gained `combine_forecasts()`, reusing the single
  walk-forward run already needed for the comparison table (no second,
  slower run) to save `results/tables/forecasts_all_models.csv` -- 879
  rows, one actual + 5 model-forecast columns, with a hard assertion that
  every model shares the same (Date, actual) pairs before merging.
- **Scope, Part 1**: `features.py` gets `target_ret_5d` -- signed sum of
  r_{t+1}..r_{t+5}, computed directly from `log_return` (not derived from
  `target_rv_5d`, which discards the sign). Verified against a hand
  calculation, confirmed genuinely distinct from `target_rv_5d`
  (correlation -0.26, not 1), same 5-row NaN tail as the vol target.
- **Scope, Part 2**: `src/risk.py` -- `deannualize_5day_vol()` (inverts
  `features.py`'s sqrt(252/5) factor), then closed-form Gaussian
  VaR/ES at 95%/99%, one function path for all five models (no
  per-model branching). Verified by hand for one value, and checked
  ES_95 > VaR_95, VaR_99 > VaR_95, ES_99 > ES_95, all positive, across
  every model's full forecast series -- all held.
- **Scope, Part 3+4**: `src/backtesting.py` -- Kupiec unconditional
  coverage, Christoffersen independence + conditional coverage (LR_cc =
  LR_uc + LR_ind), and the simple CLAUDE.md-specified ES backtest (mean
  realised shortfall vs. mean forecast ES on violation days only).
  Stress-tested the likelihood-ratio edge cases directly (zero
  violations, all violations, near-perfect large-sample calibration)
  before trusting it on real data -- no NaN/crash, and the near-perfect
  case correctly failed to reject the null (p≈0.29).
- **Scope, Part 5**: One combined table, `results/tables/
  backtest_summary.csv`, 10 rows (5 models x 2 confidence levels):
  violation rate, Kupiec LR/p-value, Christoffersen independence
  LR/p-value, conditional-coverage LR/p-value, and the ES-ratio backtest.
  Real finding, not assumed: every model passes Kupiec at 95% (right
  overall rate) but fails Christoffersen independence at 95%
  (p≈0.0000 for all five -- violations cluster in time), and realised
  shortfall exceeds forecast ES by ~8-21% on violation days across
  models. This is exactly the fat-tail under-coverage
  `docs/risks_and_roadmap.md`'s Gaussian-VaR watch-item predicted --
  updated that entry with the actual numbers rather than leaving it as
  a prediction.
- **Also updated**: `README.md`/`CLAUDE.md` status, confirmed no other
  file referenced a stale forecasts/backtest filename (none existed
  before this session, so nothing to find).

## 2026-08-12 — Correction: full-sample Christoffersen "clustering" was mostly a mechanical artifact

- **Tool**: Claude Code (Anthropic).
- **What went wrong**: the entry above reported all five models failing
  Christoffersen independence at 95% (p≈0.0000) and framed it as a
  genuine finding about violation clustering, without checking whether
  the overlapping 5-day forecast horizon itself could produce that result
  mechanically. The user asked for this check before it went further.
- **What was checked**: confirmed directly (not just argued) that
  `target_ret_5d` at consecutive test dates shares 4 of its 5 underlying
  daily returns (r[i+1..i+5] vs. r[i+2..i+6]) -- exactly the kind of
  overlap known to induce mechanical serial correlation in derived
  indicator series, independent of any real clustering in the underlying
  process.
- **Robustness check added**: `christoffersen_non_overlapping_check()` in
  `backtesting.py` -- reuses `christoffersen_independence_test()` and
  `compute_risk_measures()` unmodified, subsamples every 5th test date
  (stride = the horizon) so consecutive checks share zero underlying
  returns. `kupiec_test()`, `es_backtest()`, and `risk.py` were not
  touched; `results/tables/backtest_summary.csv` regenerated and
  confirmed byte-identical to the pre-check version.
- **Result**: independence p-values move from ≈0.0000 (full sample,
  LR 60-125) to 0.32-0.51 (non-overlapping, n=175 per model) for all five
  models, with **zero** consecutive violations (n11=0) in every model's
  subsample. Saved to
  `results/tables/christoffersen_nonoverlap_check.csv`.
- **Interpretation, stated plainly**: the original full-sample
  Christoffersen failure was mostly a structural artifact of the
  overlapping forecast horizon, not a genuine per-model clustering
  finding -- it doesn't distinguish between the five models, since the
  mechanism producing it is identical across all of them. Kupiec and the
  ES-ratio backtest are unaffected and still stand. `docs/risks_and_
  roadmap.md`'s Gaussian-VaR entry corrected accordingly rather than left
  as the (overstated) original claim.
- **Why this is worth having on record**: same reason as the RQ1
  feature-set correction above -- a real result was reported to the user
  before a caught issue corrected it, and the thesis's AI-use disclosure
  should reflect that rather than only show the final, clean version.

## 2026-08-12 — Descriptive stats table + two thesis figures

- **Tool**: Claude Code (Anthropic).
- **Scope, Part 1**: `src/descriptive_stats.py` -- mean/std (daily and
  annualised), skewness, excess kurtosis, min, max, and a Jarque-Bera
  normality test on `log_return` (full cleaned sample, n=3898, the one
  NaN first-row dropped). Saved to `results/tables/
  returns_descriptive_stats.csv`; prints a booktabs LaTeX table (caption
  below the tabular). Caught and fixed two LaTeX-correctness issues before
  handing the table over: the Jarque-Bera p-value underflows to a literal
  `0.0` in float64 (statistic is ~33,604, astronomically significant) --
  displayed as "< 0.0001" instead of a misleading bare zero -- and the
  original caption had a raw `^GSPC`, which breaks LaTeX outside math mode
  (`^` needs escaping) -- fixed to `\texttt{\textasciicircum GSPC}`.
- **Scope, Part 2**: `src/figures.py`, `matplotlib` added to the venv and
  pinned in `requirements.txt` (wasn't installed before this session).
  Both figures saved as vector PDF, 13cm x 7cm (confirmed by reading each
  PDF's `/MediaBox` directly, not assumed from the `figsize` argument):
  `results/figures/returns_timeseries.pdf` (full sample, 2020-03-16
  COVID crash annotated) and `results/figures/forecast_vs_actual_test.pdf`
  (test period only, actual vs. GJR-GARCH vs. XGBoost). Rendered PNG
  previews of both (temp files, not committed) before accepting the PDFs
  -- the first pass had the crash annotation overlapping the x-axis tick
  labels, and Figure 2's date ticks were so dense the labels ran into each
  other illegibly. Fixed with an explicit date locator/formatter and
  repositioned annotation, re-rendered, re-checked visually before
  finalising.

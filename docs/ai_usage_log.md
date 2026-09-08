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

## 2026-08-13 — Replaced forecast_vs_actual_test.pdf with a forecast-error plot

- **Tool**: Claude Code (Anthropic).
- **Caught a sign-convention error in the user's own request before
  building anything**: the request said "negative values mean the model
  underestimated realised volatility" for `error = actual - forecast`.
  That's backwards -- with that formula, a *positive* error means
  actual > forecast, i.e. realised volatility came in higher than
  predicted, which is the underprediction case. Checked both extremes for
  XGBoost directly before picking one: most negative error was
  2025-04-10 (actual=0.2133, forecast=0.6617, error=-0.4484 --
  *over*-prediction, forecast far exceeded what happened); most positive
  was 2025-04-02 (actual=0.8629, forecast=0.2797, error=+0.5832 -- genuine
  underprediction, and a dramatic one: realised vol hit 86.3% annualised
  against a 28.0% forecast). Annotated the second one; flagged the
  discrepancy to the user rather than silently building whichever the
  literal instruction implied.
- **Scope**: `src/figures.py` -- `plot_forecast_vs_actual()` replaced with
  `plot_forecast_error()` (GJR-GARCH and XGBoost error series, zero
  reference line, XGBoost's largest underprediction annotated,
  y-axis "Forecast error (actual − forecast)"). Saved as
  `results/figures/forecast_error_test.pdf` (new filename, per
  instructions, since this replaces rather than edits the old plot);
  the old `forecast_vs_actual_test.pdf` removed from the repo
  (`git rm`, still recoverable from history). `returns_timeseries.pdf`
  was locked (open in a viewer) during this session and wasn't touched --
  it didn't need to be regenerated anyway, since only the second figure
  changed.
- **Verification**: rendered a PNG preview and checked it visually before
  finalising (same discipline as last session's date-tick catch) --
  annotation placement and legend didn't collide, date ticks stayed
  legible reusing the existing 6-month locator/rotation. Confirmed the
  final PDF is 13cm x 7cm by reading its `/MediaBox` directly.

## 2026-08-13 — Pre-hybrid audit; three correctness fixes; tree figure

- **Tool**: Claude Code (Anthropic).
- **Scope, Part 0 (audit)**: Full pre-hybrid audit at the user's request —
  every figure in the LaTeX cross-checked against its source CSV, a real
  `pdflatex` compile, code-vs-text consistency, reproducibility trace,
  tense/terminology pass, docs hygiene. ~30 findings; the numeric
  cross-check found one transcription error ("within 13% on RMSE" is the
  MAE figure; RMSE is 18%). Two genuine correctness bugs found and, on a
  follow-up instruction, fixed (below). Most findings remain unactioned.
- **Scope, Part 1 (walk-forward contract)**: `predict()` took no arguments
  and reused a feature row cached during `fit()`, so weekly-refit RF and
  XGBoost emitted one frozen forecast per ISO week (184 distinct values
  across 879 test dates). Separately, the training set at date t included
  row t, whose label sums r_{t+1}..r_{t+5} — the exact (X, y) pair being
  predicted. Fixed structurally rather than per-model: the interface is now
  `fit(history)`/`predict(history)`, and `walkforward.available_history()`
  masks not-yet-knowable labels before any model sees the frame, so
  `hybrid.py` inherits both guarantees. arch-backed models now raise instead
  of serving a stale forecast.
- **Scope, Part 2 (feature timing)**: the three return features were built
  with `.shift(1)` (r_{t-1}) while `hist_vol_*d` correctly ended at t —
  contradicting the thesis's own r_t notation. Not leakage (too
  conservative, not too permissive). Renamed to `today_*` and de-lagged.
- **Regression discipline**: the user required econometric output to be
  unchanged. The first check reported FAIL at ~1e-16 — the assistant
  diagnosed this as the *check* being wrong, not the fix: pandas 3.0 writes
  CSVs at 16 significant digits, so the committed file does not round-trip
  float64 and was never a valid baseline (the same delta appeared on a
  column the change cannot touch). Re-run in memory against a pristine
  `git worktree` of the pre-fix commit: EWMA/GARCH/GJR-GARCH bitwise
  identical, and character-identical in the regenerated CSV.
- **Effect on results, stated plainly**: RF/XGBoost numbers moved twice.
  GJR-GARCH holds RMSE and QLIKE; **Random Forest now has the lowest MAE**
  (0.04341 vs 0.04358). XGBoost wins no metric and fails Kupiec at 99%
  (p=0.0129). The original write-up's central claim — XGBoost best on MAE —
  does not survive. Reported to the user rather than presented as a
  refinement.
- **Scope, Part 3 (tree figure)**: `results/figures/xgboost_tree_example.pdf`
  — tree 0 of a one-off 200-tree fit on the 2011–2022 training split, same
  hyperparameters/seed as the evaluated model, explicitly separate from the
  walk-forward refits. Drawn with matplotlib from `trees_to_dataframe()`
  rather than `xgboost.plot_tree`, which needs graphviz plus a `dot` binary
  (absent here, and a system dependency a reader reproducing the repo would
  have to install). First render had the yes/no edge labels colliding with
  node boxes; caught by rendering a PNG preview and looking at it, replaced
  with a stated branch convention.
- **Also**: `thesis/` deleted at the user's request (write-up moved to
  Overleaf; recoverable from git history). LaTeX is to be delivered in chat
  from now on.

## 2026-08-18 — Hybrid model (RQ2): GJR-GARCH + XGBoost-on-residual

- **Tool**: Claude Code (Anthropic).
- **Scope**: `src/models/hybrid.py` built to the design specified by the
  author (GJR-GARCH baseline, XGBoost residual learner, 6 RQ1 features +
  the GJR-GARCH forecast as a 7th, weekly ML refit, in-sample training
  residuals, reuse of the validated walk-forward GJR-GARCH forecasts for
  the test period). Wired into `evaluation.py`; `risk.py`/`backtesting.py`
  picked the model up automatically since both derive their model list from
  the forecast columns.
- **Two bugs caught before they could affect results**, both in the
  in-sample-forecast path added to `econometric.py`:
  (1) `arch`'s `forecast(reindex=True)` returns a forecast only from the END
  of the sample — 3,013 of 3,014 rows came back NaN. `start=0` is required to
  make every in-sample date a forecast origin. (2) The returned index omits
  the row dropped by `fit()`'s `dropna()`, so an initial positional
  alignment would have shifted every training residual by one day. Fixed by
  aligning on Date and asserting no date carries both an in-sample and a
  walk-forward baseline.
- **Unit check, done explicitly** (the author flagged this as the same class
  as the earlier RETURN_SCALE**2 bug): target, baseline forecast and residual
  all verified on the 5-day annualised vol scale, with a runtime assertion in
  `build_hybrid_frame`. Independent confirmation: the final in-sample forecast
  origin reproduces `predict()` on the same fitted model to all digits
  (0.2259568448), so the in-sample path and the validated path agree exactly.
- **Correctness inheritance verified, not assumed**: `gjr_residual` registered
  in `features.FORWARD_LABEL_COLS` so the harness masks it like any label;
  freshness re-tested with the same behavioural test used on RF/XGBoost (fit
  once, predict across six dates -> 6/6 distinct, no cached state).
- **Result reported in both directions**: the hybrid improves MAE (-7.8%) and
  RMSE (-3.5%) over GJR-GARCH but worsens QLIKE (+12.5%) and turns
  GJR-GARCH's comfortable 99% Kupiec pass into a failure (p 0.6754 ->
  0.0129). The assistant identified the mechanism rather than speculating:
  GJR-GARCH runs hot on average, the residual model learned a near-uniform
  downward shading (negative on 91% of days), under-prediction rises from
  32.7% to 46.3% of days. Violation-set overlap shows the ML tail weakness
  transfers (15/17 shared with XGBoost, 16 with RF, only 9 with the hybrid's
  own baseline). Not presented as a win.
- **Verification**: full chain re-run byte-identical; every table
  independently recomputed from its upstream inputs; EWMA/GARCH/GJR-GARCH
  still bitwise identical to the original pre-fix baseline.

## 2026-08-18 — Hybrid variant 2: quantile/pinball-loss residual model

- **Tool**: Claude Code (Anthropic).
- **Scope**: second hybrid variant motivated by the previous session's
  diagnosis (a squared-error residual model shades GJR-GARCH down uniformly
  and breaks its 99% Kupiec calibration). Since VaR at confidence alpha is the
  (1-alpha) quantile of the loss distribution, the residual model was
  retrained with XGBoost's native `reg:quantileerror` pinball objective
  targeting an upper quantile. Confirmed empirically that xgboost 3.2.0
  supports it (`quantile_alpha` is a passthrough kwarg, not a named
  parameter; alpha=0.9 verified to put 90.5% of points below the prediction).
  `hybrid.py` refactored to a shared `_HybridBase` so the two variants differ
  in exactly one thing, the objective; the symmetric variant was re-verified
  unchanged after the refactor (max diff 1.1e-16, the known CSV
  serialization artefact).
- **Quantile level chosen on training data only**: 2011-2019 inner-train,
  2020-2022 inner-validation, evaluated through the same walk-forward harness.
  The frame is truncated at the split date before anything is fitted and the
  truncation is asserted. Selection trace saved to
  `results/tables/hybrid_quantile_selection.csv`.
- **Two selection problems found and reported rather than quietly fixed**:
  (1) the initial grid (0.5-0.9) returned its own upper boundary with the
  criterion still improving monotonically, so it was extended to 0.95/0.99 to
  bracket the optimum; (2) the pre-registered criterion "highest 99% Kupiec
  p-value" turned out to be degenerate -- it is only bounded by the two-sided
  test, so it selected alpha=0.99 at roughly 3x the inner-validation MAE, an
  upper envelope rather than a forecast. Replaced with "lowest
  inner-validation MAE among Kupiec-passing candidates", which selected 0.95.
  Both changes used validation data only; the test set was untouched, and the
  chosen configuration was run through the test walk-forward exactly once.
- **Result reported as it came out, not steered**: the variant did NOT fix
  calibration. 99% Kupiec 2 violations (0.23% vs nominal 1%, p=0.0057) -- still
  a failure and marginally worse by the statistic than the symmetric hybrid's
  p=0.0129, now from over-conservatism; 95% Kupiec collapses to p=0.0000.
  MAE +142%, RMSE +90%, QLIKE +62% against the symmetric variant. Cause
  identified as a regime shift (inner-validation mean realised vol 0.1995 vs
  0.1285 on test), not a coding error. GJR-GARCH alone remains the
  best-calibrated model at 99%.

## 2026-08-18 — Diebold-Mariano test; verified bibliography

- **Tool**: Claude Code (Anthropic).
- **Scope, Part 1**: `src/diebold_mariano.py` -- DM tests for GJR-GARCH vs.
  Random Forest and GJR-GARCH vs. Hybrid (symmetric), on all three reported
  losses. Newey-West HAC (Bartlett) at 4 lags = h-1, per instruction, with lag
  sensitivity q=0..4 reported so the size of the correction is visible. The
  HAC estimator was validated against `statsmodels` OLS-on-a-constant with HAC
  covariance before use (match to 1e-10 at every lag); the shipped module
  itself depends only on numpy/scipy. Harvey-Leybourne-Newbold small-sample
  correction reported alongside.
- **Result worth flagging**: the HAC choice materially changes two of six
  verdicts (p=0.054 -> 0.16-0.21 on RMSE for both pairs), and Random Forest's
  headline MAE win over GJR-GARCH turns out to be **not significant
  (p=0.9098)**. Reported as a constraint on the write-up rather than buried.
- **Scope, Part 2**: built `references.bib` for the eight Literature Review
  sources. Every entry verified against Crossref and/or the publisher record
  rather than written from memory -- which caught an incorrect DOI the
  assistant would otherwise have supplied for Poon and Granger (2003).
- **Integrity issue found and reported, not papered over**: the thesis
  attributes to Misra et al. (2025) a finding that paper does not report (it
  finds ML clearly beating GARCH, not criterion-dependent rankings), and the
  source is an SSRN working paper rather than a peer-reviewed article. The
  assistant declined to reproduce the inaccurate sentence in the converted
  Literature Review and supplied a corrected version plus the option of
  dropping the citation. Logged as `[act now]` in
  `docs/risks_and_roadmap.md`.

## 2026-08-18 (cont.) — Source verification: Gunnarsson et al. attribution

- **Tool**: Claude Code (Anthropic).
- **Scope**: attempted to verify, against the paper body rather than the
  abstract, whether Gunnarsson et al. (2024) genuinely discuss data leakage
  and short-sample overfitting as recurring themes, as the Literature Review
  claims. Full text could NOT be obtained: ScienceDirect returns 403, and
  although Unpaywall reports hybrid open access, neither it, Semantic Scholar,
  nor the NTNU Open repository exposes a retrievable PDF.
- **Reported as unconfirmed rather than resolved either way.** The verbatim
  abstract mentions none of data leakage, look-ahead bias or overfitting, and
  the paper's stated aims are different (ML vs. econometric performance,
  explainable-AI uptake, future research). The "data leakage" material that
  search engines surface next to this paper traces to a different 2025
  Computational Economics review. On that evidence the specific attribution
  was NOT restored; the safer characterization stands, and two better-sourced
  quotes from the verified abstract were offered instead.

## 2026-09-08 — Final pre-submission audit; thesis synced back into the repo

- **Tool**: Claude Code (Anthropic).
- **Scope**: full audit of the complete thesis against its own source data
  before submission, then applying the author-approved corrections. Compile
  check, cross-reference and citation check, every printed number re-verified
  against the committed CSVs, table/page overflow check, and a pass against
  the university guide's final checklist.
- **Verification method, not spot-checking**: `backtest_summary.csv`,
  `christoffersen_nonoverlap_check.csv`, `diebold_mariano.csv` and
  `model_comparison.csv` were each re-derived from source by re-running the
  scripts and compared to the committed files (all matched to 1e-9). Derived
  claims in the prose -- the hybrid's correction statistics, the 2 April 2025
  episode, the violation-set overlaps, the inner-validation regime shift --
  were recomputed from `forecasts_all_models.csv` rather than taken on trust.
- **One real error found, and it changed a stated verdict.** The four Random
  Forest and XGBoost rows of the Kupiec table were stale pre-feature-timing-fix
  values. Random Forest at 99% was reported as p=0.0535, "passing only
  marginally"; it is p=0.0270 and **fails**. XGBoost at 99% was reported as
  p=0.0059; it is p=0.0129. The document also contradicted itself: the prose
  said the hybrid matched XGBoost's statistics exactly, which is true of the
  real numbers and false of the printed row. Corrected in the table, the
  Results text and the Conclusions.
- **The correction strengthened the thesis's argument rather than weakening
  it** -- all three econometric models pass at 99% and all three ML-based
  models fail -- which is stated here because the opposite outcome would have
  been equally reportable.
- **Also found and reported**: the Overleaf project was still building a stale
  `forecast_error_test.pdf` (annotating +0.583 where current data gives
  +0.730); an orphaned Conclusions paragraph left at the end of the Literature
  Review; three cross-references pointing at methodology sections rather than
  the evidence they invoked; four tables overflowing the text block, one badly
  enough that a column header was cut off the page; and `diebold_mariano.py`
  plus its results table being untracked in git, so the section carrying the
  thesis's only statistically confirmed result had no code in the repository.
- **Author-directed fixes applied**: the corrections above, plus a descriptive
  statistics table (Table 1) built from the existing
  `returns_descriptive_stats.csv` so the heavy-tails argument the Conclusions
  rely on is shown rather than asserted; `hyperref[hidelinks]`; a full title
  page; list of tables and figures; and a rewritten README documenting the
  complete reproduction path, which did not previously exist.
- **Nothing was fixed before being reported.** The audit was delivered as a
  findings list first, prioritized, and the author decided what to action.
- Placeholders deliberately left for the author rather than invented: tutor
  name, submission date, repository URL.

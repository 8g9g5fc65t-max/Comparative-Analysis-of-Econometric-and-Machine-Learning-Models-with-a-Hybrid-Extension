# Dynamic Volatility and Market Risk Forecasting

TFM (Master's thesis) — Máster en Finanzas Cuantitativas, UNED.
Author: Alejandro Basabe Oleaga. Deadline: 14 September 2026.

*Dynamic Volatility and Market Risk Forecasting: A Comparative Analysis of
Econometric and Machine Learning Models with a Hybrid Extension.*

Seven models forecast 5-day-ahead realised volatility of the S&P 500 under one
strictly out-of-sample walk-forward framework, and every forecast is carried
through to backtested Value at Risk and Expected Shortfall:

| Family | Models |
|---|---|
| Econometric | EWMA (λ = 0.94), GARCH(1,1), GJR-GARCH |
| Machine learning | Random Forest, XGBoost |
| Hybrid | GJR-GARCH + XGBoost-on-residual, symmetric and quantile-loss variants |

See [CLAUDE.md](CLAUDE.md) for the full project brief (research questions,
locked-in methodology decisions, model list, tutor guidance) and
[docs/risks_and_roadmap.md](docs/risks_and_roadmap.md) for known risks and the
extension backlog. The thesis LaTeX is in [thesis/](thesis/).

## Status

- [x] Data pipeline
- [x] Walk-forward engine
- [x] Econometric models (EWMA, GARCH(1,1), GJR-GARCH)
- [x] ML models (Random Forest, XGBoost)
- [x] Evaluation (MAE / RMSE / QLIKE)
- [x] VaR / Expected Shortfall + backtesting (Kupiec, Christoffersen)
- [x] Hybrid model (GJR-GARCH + XGBoost-on-residual), both variants
- [x] Diebold-Mariano significance tests (Newey-West HAC)
- [x] Descriptive statistics and thesis figures
- [x] Thesis document (`thesis/Thesis.tex`, mirrored in Overleaf)

## Setup

Requires Python 3.9+ (developed and pinned against 3.11.14). Use a dedicated
virtual environment rather than installing into a shared/base Python:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate
```

(`source .venv/bin/activate` on macOS/Linux.)

```bash
pip install -r requirements.txt
```

Every script is run from the repository root and resolves its own paths
relative to the repo, so no environment variables or path configuration are
needed.

## Reproducing every result

Run the steps in this order. Steps 1–2 build the data; step 4 is the expensive
one and produces the forecast table everything downstream reads.

### 1. Data pipeline — `src/data_pipeline.py`

```bash
python src/data_pipeline.py
```

Downloads daily OHLCV for the S&P 500 index (`^GSPC`, 2011 to a frozen end date
of 2026-07-07) from Yahoo Finance via `yfinance`, and writes:

- `data/raw/gspc_raw.csv` — the untouched pull.
- `data/processed/gspc_processed.csv` — cleaned (duplicate/weekend-row checks,
  NYSE trading-calendar gap check) with `log_return` computed on Adj Close and
  three boolean flag columns (`flag_missing_value`, `flag_zero_volume`,
  `flag_outlier_return`). Suspicious rows are flagged, never silently dropped.
- `docs/data_provenance.md` — source, download date, period returned, cleaning
  rules, and the dividend-adjustment caveat (`^GSPC` is a price index, not
  total return). Regenerated on every run; do not hand-edit.

**In normal use, skip this step.** The end date is frozen, so re-running only
re-downloads data that cannot change the results, and Yahoo Finance rate
limiting has bitten this project more than once. `data/raw/gspc_raw.csv` is
committed and should be treated as final. Run it only to re-verify provenance
from scratch.

### 2. Features — `src/features.py`

```bash
python src/features.py
```

Builds the forecasting target and the RQ1 feature set from the processed
returns, prints a train/test split summary, and writes
`data/processed/gspc_features.csv`.

- Target `target_rv_5d` = `sqrt((252/5) · Σ r²_{t+1..t+5})`; `target_ret_5d` is
  the signed 5-day forward return, used for VaR breach checks.
- Six RQ1 features, all dated at *t* and using information through the close of
  day *t*: `today_return`, `today_abs_return`, `today_sq_return`,
  `hist_vol_5d`, `hist_vol_10d`, `hist_vol_20d`.
- Split: train ≤ 2022-12-31 (3,020 rows), test 2023-01-03 onward (879 rows).

Downstream scripts call `build_features()` directly, so the CSV is an
inspection artefact rather than a dependency — but running this first is the
quickest way to confirm the data is intact.

### 3. Descriptive statistics — `src/descriptive_stats.py`

```bash
python src/descriptive_stats.py
```

Writes `results/tables/returns_descriptive_stats.csv` (n, mean, std, skewness,
excess kurtosis, min, max, Jarque-Bera), the source for Table 1 of the thesis.

### 4. Walk-forward and evaluation — `src/evaluation.py`

```bash
python src/evaluation.py
```

**The main run, and the slow one:** an expanding-window walk-forward over all
879 test dates for all seven models. Econometric models refit daily, ML and
hybrid models weekly — but *every* model produces a fresh forecast on *every*
test date from that date's own feature vector. Writes:

- `results/tables/model_comparison.csv` — MAE / RMSE / QLIKE per model
  (Tables 4, 5 and 9 of the thesis).
- `results/tables/forecasts_all_models.csv` — per-date `Date`, `actual`, and
  one column per model. Every step below reads this file rather than re-running
  the walk-forward.

This script pulls the models from `src/models/econometric.py`,
`src/models/ml_models.py` and `src/models/hybrid.py`, and drives them through
`src/walkforward.py`.

### 5. Backtesting — `src/backtesting.py`

```bash
python src/backtesting.py
```

Reads `forecasts_all_models.csv`, converts each forecast to Gaussian VaR and ES
via `src/risk.py`, and writes:

- `results/tables/backtest_summary.csv` — Kupiec unconditional coverage,
  full-sample Christoffersen independence and conditional coverage, and the
  simple ES backtest, per model at 95% and 99% (Tables 6, 7 and 11).
- `results/tables/christoffersen_nonoverlap_check.csv` — the same independence
  test on every 5th test date, so consecutive checks share no underlying
  returns (Table 8). This is the *primary* independence result; the full-sample
  one is a secondary diagnostic, because the overlapping 5-day horizon induces
  serial correlation mechanically.

### 6. Diebold-Mariano tests — `src/diebold_mariano.py`

```bash
python src/diebold_mariano.py
```

Writes `results/tables/diebold_mariano.csv` (Table 12): GJR-GARCH vs. Random
Forest and GJR-GARCH vs. the symmetric hybrid, on all three losses. Headline
numbers use a Newey-West (Bartlett kernel) HAC variance at 4 lags = h−1, never
the naive i.i.d. estimator; the Harvey-Leybourne-Newbold small-sample
correction and a q = 0…4 lag sensitivity (q = 0 being the i.i.d. case) are
reported in the same file.

### 7. Quantile-level selection audit trail — `src/quantile_selection.py`

```bash
python src/quantile_selection.py
```

Writes `results/tables/hybrid_quantile_selection.csv`: how the quantile hybrid's
α was chosen on a 2011–2019 / 2020–2022 inner split of the **training period
only**, with the test period truncated off the frame before anything is fitted
(the truncation is asserted, not assumed).

Note the ordering: this script reads the GJR-GARCH column of
`forecasts_all_models.csv`, so it runs *after* step 4 — but the α it selects
(0.95) is already pinned as `SELECTED_QUANTILE_ALPHA` in
`src/models/hybrid.py`, which step 4 imports. Running it reproduces and
documents that choice; it does not feed back into the models.

### 8. Figures — `src/figures.py`

```bash
python src/figures.py
```

Writes all three thesis figures to `results/figures/`:

- `returns_timeseries.pdf` (Figure 1) — daily log returns, COVID crash annotated.
- `forecast_error_test.pdf` (Figure 3) — GJR-GARCH and XGBoost forecast error
  over the test period, worst day annotated.
- `xgboost_tree_example.pdf` (Figure 2) — one tree from an illustrative XGBoost
  fit, kept separate from the walk-forward evaluation.

**These are the files Overleaf needs.** The LaTeX in `thesis/` references them
by bare filename, so after regenerating them they must be uploaded to the
Overleaf project, replacing the copies there.

### Diagnostics (optional, write nothing)

```bash
python src/walkforward.py
```

Runs a naive last-value baseline through the walk-forward harness at each refit
cadence and prints row counts — a quick check that the engine is intact.

```bash
python src/risk.py
```

Prints per-model min/mean/max VaR and ES at both confidence levels.

## Building the thesis

```bash
cd thesis && pdflatex Thesis.tex && bibtex Thesis && pdflatex Thesis.tex && pdflatex Thesis.tex
```

`thesis/Thesis.tex` includes the three figures by bare filename, so a local
build needs them alongside the `.tex` — either copy `results/figures/*.pdf`
into `thesis/` first, or build with
`pdflatex -include-directory=../results/figures`. The Overleaf project is the
authoritative build environment; the copy here is kept in sync with it.

## Repo structure

```
data/raw/                 - untouched Yahoo Finance pull (frozen; do not re-download)
data/processed/           - cleaned series, returns, and the feature/target frame
src/
  data_pipeline.py        - download + clean + provenance
  features.py             - target and RQ1 feature construction, train/test split
  walkforward.py          - expanding-window engine; fit(history)/predict(history)
  models/econometric.py   - EWMA, GARCH(1,1), GJR-GARCH
  models/ml_models.py     - Random Forest, XGBoost
  models/hybrid.py        - GJR-GARCH + XGBoost-on-residual, both variants
  evaluation.py           - MAE/RMSE/QLIKE; runs the full walk-forward
  risk.py                 - Gaussian VaR and Expected Shortfall
  backtesting.py          - Kupiec, Christoffersen, simple ES backtest
  diebold_mariano.py      - HAC-corrected pairwise significance tests
  quantile_selection.py   - quantile-hybrid alpha selection audit trail
  descriptive_stats.py    - return distribution summary
  figures.py              - all thesis figures
results/tables/           - every table in the thesis, script-generated
results/figures/          - every figure in the thesis, script-generated
thesis/                   - LaTeX source and bibliography
docs/                     - provenance note, AI usage log, risks and roadmap
```

## Reproducibility notes

- Fixed random seed (42) for Random Forest and XGBoost. XGBoost is trained
  single-threaded: parallelised training introduced small floating-point
  differences across otherwise identical runs.
- No table or figure in the thesis is hand-edited after export. Every one is
  produced by a script listed above; if a number in the document disagrees with
  its CSV, the CSV is right.
- The walk-forward interface (`fit(history)` / `predict(history)`) is the
  enforcement point for two correctness rules: no model caches inputs between
  `fit` and `predict`, and no training label built from returns after the
  forecasting date is ever visible. Any new model inherits both by
  construction — see CLAUDE.md before adding one.
- Close CSVs in Excel before running anything that writes to
  `results/tables/`; file locks have interrupted runs before.

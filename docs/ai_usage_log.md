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

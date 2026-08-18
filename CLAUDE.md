# TFM — Dynamic Volatility and Market Risk Forecasting

## Project summary
Master's thesis (TFM) for the Máster en Finanzas Cuantitativas, UNED. Not an
official/regulated master's — the thesis should stay clear, well-scoped, and
clean rather than maximally complex. Deadline: **14 September 2026**.

Title: *Dynamic Volatility and Market Risk Forecasting: A Comparative Analysis
of Econometric and Machine Learning Models with a Hybrid Extension*.

Author: Alejandro Basabe Oleaga. Tutor approved the proposal (see "Tutor
guidance" below).

## Research questions
1. Can machine-learning methods improve out-of-sample forecasting of
   financial market volatility compared with traditional econometric models,
   and does any improvement translate into more accurate market-risk measures
   (VaR / Expected Shortfall)?
2. Does combining GARCH and machine learning (hybrid model) provide
   additional forecasting value relative to the individual models?

## Build priority (in case of time crunch)
Core deliverable (RQ1) must be finished first and must stand alone. The
hybrid model (RQ2) is a stretch extension built last, so that if time runs
out it can be dropped or left partial without breaking anything else.

Order: data pipeline → walk-forward engine → econometric models → ML models
→ evaluation → VaR/backtesting → hybrid model (last).

## Data
- Asset: S&P 500 index, ticker `^GSPC`, via `yfinance`.
- Period: ~2011–2026 (daily OHLCV).
- Note: `^GSPC` is a price index, not total return — no dividend adjustment.
  State this as a caveat in the data section.
- Save raw pull untouched to `data/raw/`; cleaned series + computed returns
  to `data/processed/`.
- Document: source, download date, ticker, period, cleaning rules (gaps,
  duplicates, zero-volume days, outliers flagged not silently dropped).
- Log returns: r_t = ln(P_t / P_{t-1}), using adjusted close.
- Forecasting target: 5-day-ahead realised volatility,
  sqrt((252/5) * sum(r_{t+i}^2, i=1..5)).

## Methodology decisions (locked in — do not silently change)
- **Out-of-sample scheme**: walk-forward expanding window, not a random
  train/test split. Train ~2011–2022, test ~2023–2026 as the initial split.
- **ML retrain cadence**: refit Random Forest / XGBoost **weekly**, not
  daily, on the expanding window. GARCH-family models can refit daily
  (cheap) — document this asymmetry explicitly in the methodology section.
- **Refit cadence and prediction freshness are SEPARATE concerns — never
  let one imply the other.** Refit cadence governs how often a model's
  *parameters* are re-estimated. It must never govern how often a
  *forecast* is produced. Every model forecasts on every test date, from
  that date's own feature vector. A weekly-refit model still predicts
  daily: same parameters all week, fresh inputs each day.
  *Why this is spelled out:* the original wording ("refit weekly") was read
  as "forecast weekly". `predict()` took no arguments and reused a feature
  row cached during the last `fit()`, so Random Forest and XGBoost emitted
  one frozen value per ISO week — 184 distinct forecasts across 879 test
  dates instead of 879. It flattered both ML models and inverted the RQ1
  ranking. Fixed 2026-08-13 by making `predict(history)` take the current
  history explicitly, so there is no cached state to go stale.
- **No training label may reach past the forecasting date.** At date t, a
  label built from r_{t+1}..r_{t+5} is knowable only from t+5 onward, so the
  last 5 rows of any expanding window must not be trained on. This is the
  rule `TrainTestSplit.split()` applies at the train/test cutoff, and it has
  to hold at *every* step of the walk-forward loop, not just that one
  boundary. `walkforward.available_history()` enforces it by masking those
  labels to NaN before any model sees the frame.
  *Why this is spelled out:* it was violated until 2026-08-13 — the training
  set at date t included row t itself, so the exact (X, y) pair being
  predicted was a training example.
- **The walk-forward interface is the enforcement point for both rules
  above.** `fit(history)` / `predict(history)`, one contract for every model
  family. Any new model (the hybrid included) inherits both guarantees by
  construction. Do not add a model that caches inputs during `fit()`, and do
  not reintroduce a no-argument `predict()`.
- **VaR**: parametric Gaussian VaR (σ̂ · z_α) as the baseline for all models,
  same distributional assumption across models so risk-measure differences
  trace back to volatility-forecast differences, not methodology
  differences. A Student-t robustness check is optional stretch, not core.
- **ES**: computed under the same Gaussian assumption; backtest can be
  simple (average shortfall beyond VaR vs. ES forecast), Acerbi-Szekely only
  if time allows.
- **Feature sets — kept separate to avoid circularity**:
  - RQ1 comparison (pure ML vs. econometric horse race): ML features are
    today's return, absolute return and squared return, plus 5/10/20-day
    historical vol — six columns, **only**. Do NOT feed GARCH/EWMA forecasts
    into this feature set — it would make the ML-vs-econometric comparison
    circular.
- **Every RQ1 feature is dated at t and uses information through the close of
  day t — no extra lag.** The target sums r_{t+1}..r_{t+5}, strictly after t,
  so r_t is known when the forecast is made; using it is not leakage. The
  columns are named `today_return` / `today_abs_return` / `today_sq_return`
  precisely so the names cannot imply a shift that isn't there.
  *Why this is spelled out:* until 2026-08-13 the three return features were
  built with `.shift(1)` (r_{t-1}) while `hist_vol_*d` beside them correctly
  ended at t — so the feature set silently mixed two different information
  cutoffs, and the models were handed *less* than the thesis claimed. Not
  leakage (it was too conservative, not too permissive), but a real
  code-vs-spec mismatch. Fixing it improved every RF/XGBoost metric.
  - Hybrid model (RQ2): ML component is trained to predict the **residual**
    left unexplained by GARCH, i.e. σ̂_hybrid = σ̂_GARCH + ê_ML. This is
    where GARCH/EWMA-derived features belong.
- **QLIKE**: defined on variance, not volatility. Watch unit consistency —
  squaring vol forecasts before computing QLIKE is an easy silent bug.
- **Backtesting**: Kupiec test (unconditional coverage) + Christoffersen
  test (independence / conditional coverage) for VaR. ES backtest per above.

## Models
- Econometric: EWMA (benchmark), GARCH(1,1), GJR-GARCH (asymmetry).
- ML: Random Forest, XGBoost.
- Hybrid: GARCH baseline + ML-on-residual (last to build).
- Evaluation metrics: MAE, RMSE, QLIKE — report all three, no single metric
  is authoritative (per Poon & Granger, 2003).

## Tutor guidance (email, already incorporated)
- Bound the number of models/assets/period early — done (single asset,
  fixed model list above).
- Comparison must be out-of-sample with homogeneous evaluation criteria —
  done via walk-forward + shared metrics.
- Complement VaR with a backtesting procedure — done (Kupiec/Christoffersen).
- For the hybrid model: clearly define each component's contribution and
  compare against standalone models to assess whether combining actually
  helps, not just assume it does.

## Repo structure
```
tfm-volatility-forecasting/
├── README.md
├── requirements.txt
├── data/{raw,processed}/
├── src/
│   ├── data_pipeline.py
│   ├── features.py
│   ├── models/{econometric.py, ml_models.py, hybrid.py}
│   ├── walkforward.py
│   ├── evaluation.py
│   ├── risk.py
│   ├── backtesting.py
│   ├── descriptive_stats.py
│   └── figures.py
├── results/{tables,figures}/
└── docs/{ai_usage_log.md, risks_and_roadmap.md, data_provenance.md}
```
No `thesis/` directory — the LaTeX lives in Overleaf (see "Current status").

## Conventions
- Python, fixed random seeds wherever relevant.
- No data leakage: every feature must be constructible from information
  available strictly at the forecasting date. Double-check this whenever
  adding a feature.
- Keep code organized by pipeline stage (matches TFM guide's reproducibility
  requirements); avoid notebook-style throwaway exploration in the final
  repo — clean scripts only.
- Don't hardcode local paths, keys, or credentials.
- Every table/figure generated for the thesis should be reproducible from a
  script, not manually edited after export.

## AI usage disclosure
The university requires a transparent declaration of AI tool use in the
final thesis (template already in `thesis/` guide docs). Keep a running,
honest log of what Claude Code was used for in `docs/ai_usage_log.md` as we
go, rather than reconstructing it at the end.

## Current status
Data pipeline, walk-forward engine, econometric models (EWMA, GARCH,
GJR-GARCH), and ML models (Random Forest, XGBoost) are built and evaluated
together in `results/tables/model_comparison.csv`. Per-date forecasts for
all five models are persisted in `results/tables/forecasts_all_models.csv`
(for the Diebold-Mariano test later, without re-running walk-forward).
Gaussian VaR/ES (`src/risk.py`) and Kupiec/Christoffersen/simple-ES
backtesting (`src/backtesting.py`) are built and run in
`results/tables/backtest_summary.csv` plus
`results/tables/christoffersen_nonoverlap_check.csv`. Descriptive stats
(`src/descriptive_stats.py`) and thesis figures (`src/figures.py`) are built.
Dev environment moved to Python 3.11 (see `requirements.txt`). See README.md
for how to run things.

**Thesis document lives in Overleaf, NOT in this repo.** `thesis/` was deleted
on 2026-08-13 (recoverable from git history — it was committed in `e376036`).
Do not recreate it. When the write-up needs changing, **output the LaTeX in
chat** for the author to paste into Overleaf. The draft there covers
Introduction, Literature Review, Data, Methodology, Results and Conclusions,
with the hybrid-model subsection in future tense and no bibliography yet.

**2026-08-13 — three correctness fixes, and what they invalidated.** Prediction
freshness, future labels, and feature timing (all three now in "Methodology
decisions") were being violated. Econometric results are **bitwise unchanged**
throughout; every RF/XGBoost number moved twice. Net effect on the headline
RQ1 result:
- GJR-GARCH wins RMSE and QLIKE outright.
- **Random Forest now has the lowest MAE** (0.04341 vs GJR-GARCH's 0.04358,
  a 0.4% gap), so "no single model dominates all three criteria" is true
  again — but with RF, not XGBoost, as the ML model that wins a metric.
- XGBoost wins nothing: 3rd on MAE, 4th on RMSE and QLIKE, and it is the
  model whose 99% VaR coverage fails hardest.
`results/tables/*` and `results/figures/*` are regenerated and current. The
Overleaf Results and Conclusions still describe the *original* (pre-fix)
numbers and need rewriting against the committed tables.

A full audit on 2026-08-13 also found ~30 further issues (missing
`references.bib`, an undefined `\ref{sec:dm}`, two overfull tables,
README/reproducibility gaps). Only the three correctness bugs and one
figure-path typo were fixed; the rest is unactioned.

## Related docs
`docs/risks_and_roadmap.md` tracks known risks/watch-items and a prioritized
backlog of possible extensions. Not loaded automatically — check it when
debugging something that feels like a recurring issue, or when deciding
whether a stretch feature is worth building.

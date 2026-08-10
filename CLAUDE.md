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
- **VaR**: parametric Gaussian VaR (σ̂ · z_α) as the baseline for all models,
  same distributional assumption across models so risk-measure differences
  trace back to volatility-forecast differences, not methodology
  differences. A Student-t robustness check is optional stretch, not core.
- **ES**: computed under the same Gaussian assumption; backtest can be
  simple (average shortfall beyond VaR vs. ES forecast), Acerbi-Szekely only
  if time allows.
- **Feature sets — kept separate to avoid circularity**:
  - RQ1 comparison (pure ML vs. econometric horse race): ML features are
    lagged/absolute/squared returns and 5/10/20-day historical vol **only**.
    Do NOT feed GARCH/EWMA forecasts into this feature set — it would make
    the ML-vs-econometric comparison circular.
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
│   └── backtesting.py
├── results/{tables,figures}/
├── thesis/          # LaTeX source
└── docs/ai_usage_log.md
```

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
Data pipeline, walk-forward engine, and econometric models (EWMA, GARCH,
GJR-GARCH) are built and evaluated. See README.md for how to run the
pipeline.

## Related docs
`docs/risks_and_roadmap.md` tracks known risks/watch-items and a prioritized
backlog of possible extensions. Not loaded automatically — check it when
debugging something that feels like a recurring issue, or when deciding
whether a stretch feature is worth building.

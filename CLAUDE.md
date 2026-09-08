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
- **Hybrid specification (BUILT 2026-08-18, `src/models/hybrid.py`)** — chosen
  deliberately over alternatives, do not substitute:
  - Baseline **GJR-GARCH**, not GARCH(1,1): it is the strongest econometric
    model here, so ML is asked to add value on top of the best benchmark
    rather than one that is easier to beat.
  - Residual learner **XGBoost**, not Random Forest: boosting already works by
    fitting each stage to the previous stage's residuals.
  - Features: the six RQ1 features **plus** GJR-GARCH's own forecast for that
    date (7 total). The RQ1 anti-circularity rule does **not** apply here —
    feeding the econometric forecast to the ML component is the mechanism of a
    hybrid, not a violation. `RQ1_ML_FEATURES` is imported, never redefined,
    so the two feature sets cannot drift into each other.
  - Baseline forecasts have two provenances on purpose: **test period reuses
    the validated walk-forward GJR-GARCH run** (never recomputed — a
    reimplementation could drift); **training period uses one in-sample fit**
    over 2011–2022, not a walk-forward re-run. Those training residuals are
    in-sample and therefore optimistically small; it is not leakage (every
    training date precedes the whole test window) but it does bias the
    residual model toward under-correcting.
  - The residual column is registered in `features.FORWARD_LABEL_COLS`, so the
    harness masks it exactly like any other forward-looking label.
  - ML component refits **weekly** through the standard walk-forward
    interface, inheriting both correctness guarantees; verified behaviourally
    (fit once, predict across six dates → 6/6 distinct, no cached state).
- **Two hybrid variants exist, differing ONLY in the residual model's training
  objective** (`src/models/hybrid.py`, shared `_HybridBase`):
  - **Hybrid (symmetric)** — `reg:squarederror`, targets the mean residual.
  - **Hybrid (quantile)** — `reg:quantileerror` (XGBoost's native pinball
    loss) at `SELECTED_QUANTILE_ALPHA = 0.95`, targeting an upper quantile of
    the residual distribution. Motivated by VaR at confidence α being the
    (1−α) quantile of the loss distribution, so the training objective matches
    the downstream use.
  - Same baseline, same 7 features, same capacity, same seed, same weekly
    cadence — a one-variable comparison by construction.
  - The quantile level is chosen by `src/quantile_selection.py` on a
    2011–2019 / 2020–2022 inner split of the **training period only**; the
    test set is truncated off the frame before anything is fitted, and the
    truncation is asserted. **Never hand-tune it against a test result.**
  - **The quantile variant did not work** (2026-08-18) — see Current status.
    It is kept because the negative result is informative, not because it is a
    recommended configuration.
- **QLIKE**: defined on variance, not volatility. Watch unit consistency —
  squaring vol forecasts before computing QLIKE is an easy silent bug.
- **Backtesting**: Kupiec test (unconditional coverage) + Christoffersen
  test (independence / conditional coverage) for VaR. ES backtest per above.
- **Diebold-Mariano must use a HAC variance estimator, never the naive i.i.d.
  one** (`src/diebold_mariano.py`). Forecasts are daily over a 5-day horizon,
  so consecutive loss differentials share four of five underlying returns and
  are mechanically autocorrelated — the same artifact that already invalidated
  a full-sample Christoffersen reading. Headline numbers use **Newey-West
  (Bartlett kernel) at 4 lags = h−1**; lag sensitivity q=0..4 is reported
  alongside, q=0 being exactly the i.i.d. case.
  *Why this is spelled out:* it is not cosmetic. Two of the six comparisons
  sit at p≈0.054 under i.i.d. and move to p=0.16–0.21 under HAC — i.i.d. would
  have manufactured two borderline-significant results that are not there.

## Models
- Econometric: EWMA (benchmark), GARCH(1,1), GJR-GARCH (asymmetry).
- ML: Random Forest, XGBoost.
- Hybrid: GJR-GARCH baseline + XGBoost-on-residual. **Built** — see the
  hybrid specification bullet above.
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
│   ├── quantile_selection.py
│   ├── diebold_mariano.py
│   └── figures.py
├── results/{tables,figures}/
├── thesis/{Thesis.tex, references.bib}
└── docs/{ai_usage_log.md, risks_and_roadmap.md, data_provenance.md}
```
`thesis/` holds the LaTeX source, mirrored in Overleaf (see "Current status").

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
**All seven models are built and evaluated.** Data pipeline, walk-forward
engine, econometric models (EWMA, GARCH, GJR-GARCH), ML models (Random Forest,
XGBoost) and both hybrid variants (symmetric and quantile) are compared in
`results/tables/model_comparison.csv`; the quantile level's selection trace is
in `results/tables/hybrid_quantile_selection.csv`. Per-date forecasts for all
seven models are persisted in `results/tables/forecasts_all_models.csv`
(for the Diebold-Mariano test later, without re-running walk-forward).
Gaussian VaR/ES (`src/risk.py`) and Kupiec/Christoffersen/simple-ES
backtesting (`src/backtesting.py`) are built and run in
`results/tables/backtest_summary.csv` plus
`results/tables/christoffersen_nonoverlap_check.csv`. Descriptive stats
(`src/descriptive_stats.py`) and thesis figures (`src/figures.py`) are built.
Dev environment moved to Python 3.11 (see `requirements.txt`). See README.md
for how to run things.

**The thesis LaTeX is BACK in the repo (2026-09-08) and is the working copy.**
`thesis/Thesis.tex` + `thesis/references.bib`, restored from the Overleaf
export and then corrected in place. Overleaf is still where the author builds
and submits, so the two are **mirrors that must be kept in sync**: edit
`thesis/Thesis.tex` here, verify it compiles, and tell the author to paste it
into Overleaf — do not silently let the two diverge. (Earlier guidance said the
repo had no `thesis/` and that LaTeX should only be output in chat. That is
obsolete.)

The document is complete: title page, abstract, keywords, ToC/LoT/LoF,
Introduction, Literature Review, Data (incl. descriptive statistics),
Methodology, Results (incl. hybrid, quantile variant and Diebold-Mariano),
Conclusions, Directions for Future Research, Declaration of AI Use, and a
working bibliography. It compiles clean: 31 pages, no undefined references or
citations, **no overfull boxes**.

Three placeholders remain for the author, marked in the source: tutor name,
submission date, repository URL.

The figures are referenced by bare filename, so `results/figures/*.pdf` must be
uploaded to Overleaf whenever they are regenerated. A local build needs them
alongside the `.tex` — see README.md.

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
`results/tables/*` and `results/figures/*` are regenerated and current.
(The write-up was rewritten against these numbers over 2026-08-18/09-08; the
last stale values were cleared on 2026-09-08 — see below.)

The rest of the 2026-08-13 audit list (missing `references.bib`, an undefined
`\ref{sec:dm}`, overfull tables, README/reproducibility gaps) is **now
actioned** as of 2026-09-08.

**2026-08-18 — hybrid model built; the RQ2 answer is genuinely split.** The
hybrid **improves MAE (−7.8%) and RMSE (−3.5%) over GJR-GARCH but worsens
QLIKE (+12.5%)**, and it **destroys GJR-GARCH's tail calibration**: 99% Kupiec
goes from p=0.6754 (comfortable pass, 10 violations) to p=0.0129 (fail, 17
violations). Mechanism, not speculation: GJR-GARCH systematically runs hot
(mean forecast 0.1425 vs mean actual 0.1285), so the residual model learned a
near-uniform downward shading — the correction is negative on 91% of days.
That shrinks average error but pushes under-prediction from 32.7% to 46.3% of
days, which is exactly what QLIKE and VaR coverage punish. The ML tail
weakness **does** carry over: the hybrid's 99% violation set overlaps XGBoost
on 15 of 17 days and Random Forest on 16, but its own baseline on only 9.
So RQ2's answer is "adds value on symmetric loss, costs value on
risk-relevant loss" — not a clean win.

**2026-08-18 — quantile-loss hybrid variant tried; it did NOT fix calibration.**
Motivated directly by the diagnosis above: train the residual model with
pinball loss at an upper quantile so the objective matches VaR's own
definition. Quantile level 0.95 selected on training-period inner-validation
(2020–2022), test set untouched during selection. **On the test set it
overshot severely and failed in the opposite direction:**
- 99% Kupiec: 2 violations (0.23% vs nominal 1%), p=0.0057 — still a FAIL, and
  a marginally worse one than the symmetric hybrid's p=0.0129. 95% Kupiec
  collapses outright: 0.80% vs nominal 5%, p=0.0000.
- Accuracy cost is enormous: MAE +142% and RMSE +90% vs the symmetric hybrid;
  QLIKE +62%. Mean forecast 0.2177 against mean actual 0.1285.
- Mechanism: the correction is positive on **100%** of days (mean +0.0752),
  where the symmetric one was negative on 91%. It swapped systematic
  under-forecasting for much larger systematic over-forecasting.
- Cause of the overshoot is a regime shift, not a coding error: inner
  -validation (2020–2022) has mean realised vol 0.1995, the test period only
  0.1285 — 36% calmer. A 95th-percentile residual learned under COVID-era
  stress is far too wide for 2023–2026.
**GJR-GARCH alone remains the best-calibrated model at 99% (p=0.6754); neither
hybrid variant beats it on tail calibration.** Keep the negative result — it is
a real finding about objective-alignment not being sufficient when the
residual distribution is regime-dependent.

**2026-08-18 — Diebold-Mariano done (`results/tables/diebold_mariano.csv`),
and it changes what can be claimed.** Two pairs × three losses, Newey-West
HAC at 4 lags:
- **Random Forest's MAE win over GJR-GARCH is not significant — p=0.9098.**
  The 0.4% gap is noise. "RF has the lowest MAE" is a ranking, not a finding,
  and the write-up must not lean on it.
- GJR-GARCH's QLIKE advantage over Random Forest **is** significant (p=0.0156).
- The symmetric hybrid's MAE improvement over GJR-GARCH **is** significant
  (p=0.0050); its RMSE improvement is **not** (p=0.1627).
- GJR-GARCH's QLIKE advantage over the hybrid **is** significant (p=0.0290).
So the defensible summary is: GJR-GARCH is significantly better on QLIKE than
both challengers; the hybrid is significantly better on MAE; everything else
is a tie. Harvey-Leybourne-Newbold small-sample correction agrees with every
verdict (n=874, so it barely moves).

**Bibliography verified 2026-08-18, both problems fixed.** All eight Literature
Review sources checked against Crossref/publisher records. Two attribution
problems were found (Gunnarsson et al. 2024, Misra et al. 2025) and **both are
now corrected in the text**: Gunnarsson is cited for claims verbatim from its
abstract, and Misra is described as reporting what it actually reports (ML
beating GARCH substantially) and labelled a working paper. `references.bib`
carries `note = {Working paper; not peer reviewed}` on the Misra entry. Nothing
outstanding here.

**2026-09-08 — final pre-submission audit. One real error found, in the Kupiec
table.** Everything else in the document was re-verified against the committed
CSVs and matched, and all three CSVs were re-confirmed to reproduce from source
(`backtest_summary.csv`, `diebold_mariano.csv`, `model_comparison.csv`).
The error: the four **Random Forest and XGBoost rows of the Kupiec table were
stale pre-feature-timing-fix values**, and one changed a verdict.
- **Random Forest at 99% FAILS: 16 violations, 1.83%, LR 4.891, p=0.0270.** The
  document had claimed p=0.0535, "passing only marginally". It does not pass.
- XGBoost at 99% is 17 violations, LR 6.179, **p=0.0129** (the document had
  2.06% / 7.588 / 0.0059). This also removed a self-contradiction: the text
  already stated the hybrid matched XGBoost's count and statistics exactly,
  while the printed XGBoost row disagreed with it.
- The 95% rows for both models were also wrong, though the "all pass at 95%"
  verdict survives.
**Net effect on the argument: it got stronger.** All three econometric models
pass at 99%; all three ML-based models (RF, XGBoost, hybrid) fail. The
"systematic tail-calibration weakness of the tree-based approach" claim no
longer rests on one failure plus one near-miss.
Also fixed in the same pass: an orphaned Conclusions paragraph sitting at the
end of the Literature Review, three cross-references pointing at methodology
sections rather than the evidence, all four overfull tables (the
Diebold-Mariano table's last column header was being cut off the page edge),
`hyperref` link boxes, a missing descriptive-statistics table, and the title
page. `src/diebold_mariano.py` and `results/tables/diebold_mariano.csv` were
untracked until this date — Table 12 had no code in the repo — and are now
committed.

## Related docs
`docs/risks_and_roadmap.md` tracks known risks/watch-items and a prioritized
backlog of possible extensions. Not loaded automatically — check it when
debugging something that feels like a recurring issue, or when deciding
whether a stretch feature is worth building.

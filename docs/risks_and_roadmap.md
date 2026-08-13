# Risks & roadmap

Working notes, not auto-loaded by Claude Code every session (see pointer in
`CLAUDE.md`). Consult when relevant: debugging, reviewing a component before
moving on, or deciding what to build next if time allows.

Tags:
- Risks: `[watch]` aware, no action needed yet · `[act now]` do something this week
- Backlog: `[do now]` high value / low cost, worth doing before finishing the
  base thesis · `[if time permits]` genuine stretch · `[cut unless way ahead]`
  only if the base thesis is fully done with real time to spare

---

## Known risks & things to watch

**RESOLVED (2026-08-13) — two walk-forward correctness bugs that inverted the
RQ1 result.** Found by a full pre-hybrid audit; both are now fixed
structurally in `walkforward.py`'s model interface, and both rules are written
into `CLAUDE.md`'s locked-in methodology decisions.
1. *Prediction freshness.* `predict()` took no arguments and re-used a feature
   vector cached during the last `fit()`. Under the weekly ML refit cadence
   that froze one forecast per ISO week: RF and XGBoost emitted **184 distinct
   forecasts across 879 test dates**, 79% of values carried over, constant in
   all 184 weeks (econometric models, refitting daily, were unaffected).
2. *Label leakage.* At date t the training set included row t itself, whose
   label is built from r_{t+1}..r_{t+5} — so the exact (X, y) pair being
   predicted was a training example. `TrainTestSplit.split()` already trimmed
   these 5 rows at the train/test cutoff, but `walkforward.py` rebuilt its own
   untrimmed history and bypassed that guard.

**Fix**: the contract is now `fit(history)` / `predict(history)` for every
model, and `available_history()` masks not-yet-knowable labels before any model
sees the frame. There is no cached state to go stale and no reachable future
label, so neither bug can be reintroduced by `hybrid.py`. Arch-backed models
raise rather than serve a stale forecast if asked to predict for a date they
weren't fitted through. Verified after the fix: 879/879 distinct forecasts for
all five models, and EWMA/GARCH/GJR-GARCH forecasts **bitwise identical** to
pre-fix output (checked in memory against a pristine `git worktree` of HEAD —
note the committed CSVs are written at 16 significant digits and do *not*
round-trip float64 exactly, so a CSV diff is the wrong instrument for this).

**Consequence, not yet actioned**: the corrected numbers **invert the thesis's
central RQ1 finding**. GJR-GARCH now beats every model on MAE, RMSE and QLIKE;
XGBoost falls to 3rd on MAE and *last* on RMSE; "no single model dominates
across all three criteria" is false. XGBoost still fails Kupiec at 99% and
fails harder (p 0.027 → 0.0059), but the Conclusions' central irony — that the
best-MAE model has the worst tail coverage — no longer holds, because XGBoost
no longer has the best MAE. Results, Conclusions and §5.2 of `thesis/Thesis.tex`
describe the old numbers and must be rewritten against the regenerated tables.

**[act now] `results/figures/forecast_error_test.pdf` is stale.** It still
plots the pre-fix XGBoost forecasts. Regeneration is blocked only by a file
lock (both figure PDFs were open in a viewer); rerun `python src/figures.py`
with the viewer closed. The annotated point stays 2025-04-02 but its numbers
change: forecast 28.0% → **13.8%**, underprediction 58.3pp → **72.5pp**. The
thesis paragraph quoting those figures needs updating with it.

**[act now] No significance testing on model comparisons yet.** The
econometric table (GJR-GARCH < GARCH < EWMA on MAE/RMSE/QLIKE) shows ranked
numbers but not whether the differences are statistically meaningful. A
Diebold-Mariano test is the standard tool for this. Cheap to add, and it's
exactly the kind of "diagnostics and robustness" the TFM guide's evaluation
checklist asks for. See backlog — do this once all models (econometric + ML
+ hybrid) are in, so it's one pairwise comparison pass instead of three.
Per-date forecasts for all five current models are now persisted in
`results/tables/forecasts_all_models.csv`, so this is unblocked whenever
it's picked up -- no need to re-run walk-forward to get there.

**[watch] Gaussian VaR underestimates tail risk.** Parametric Gaussian VaR
is the agreed baseline (fast, consistent across models, easy to defend) but
financial returns have fatter tails than Gaussian assumes — VaR will likely
under-cover during genuine crisis periods (e.g. anything COVID-like in the
test window). This isn't a bug, it's a known limitation of the chosen
method. Say so explicitly in the conclusions/limitations section rather
than let an evaluator find it unstated — a named limitation reads as rigor,
an unnamed one reads as an oversight. The simple ES backtest
(`results/tables/backtest_summary.csv`) shows realised shortfall exceeding
forecasted ES by **~8-24%** across models on violation days (was ~8-21%
before the 2026-08-13 walk-forward fix; the econometric rows are unchanged,
the widening is RF/XGBoost) — that part of the under-coverage story holds and
is unaffected by the item below.

**Correction (2026-08-12) — the full-sample Christoffersen "clustering"
result was mostly a mechanical artifact, not a genuine per-model finding.**
The previous entry here reported all five models failing Christoffersen
independence at 95% (p≈0.0000) and framed it as evidence of real
violation clustering. That framing doesn't hold up: `target_ret_5d` at
consecutive test dates shares 4 of its 5 underlying daily returns
(confirmed directly: r[i+1..i+5] vs. r[i+2..i+6] overlap in 4 places) --
with a horizon that overlapping, the violation indicator is mechanically
serially correlated regardless of whether the underlying volatility
process clusters at all. Re-running Christoffersen on a non-overlapping
subsample (every 5th test date, so consecutive checks share zero
underlying returns -- `results/tables/christoffersen_nonoverlap_check.csv`)
flips the result completely: independence p-values move to **0.27-0.51** for
every model, and every model has **zero** consecutive violations in the
subsample (n11=0). The original full-sample LR statistics (59-116,
against a chi2(1) critical value of ~3.84) were almost entirely the
overlapping-horizon artifact, not a per-model signal.
(p-value range and LR range both restated 2026-08-13 after the walk-forward
fix changed the RF/XGBoost forecasts; the conclusion is unchanged, and the
econometric numbers in it never moved.)
**Practical effect on the thesis**: don't present the original
Christoffersen failure as a finding that discriminates between models --
it doesn't (it's ~identical in cause across all five). If Christoffersen
independence is reported at all, report both the full-sample number (with
this caveat attached) and the non-overlapping robustness check, and lead
with the latter as the more trustworthy read. The Kupiec results and the
ES-ratio backtest are untouched by this and still stand as reported.

**[watch] ML overfitting risk on a relatively short sample.** ~3,000
training rows is workable but not large for tree-based models with many
potential splits. Gunnarsson et al. (2024) flag this as a common failure
mode in this literature. The walk-forward + weekly refit design already
mitigates this structurally — but if RF/XGBoost dramatically outperform
GARCH-family models on training-adjacent test dates and then degrade later
in the test window, that's a symptom worth checking for, not ignoring.

**[watch] ML vs. econometric result may be a tie or a loss for ML.** Not a
risk to the thesis — a tie is a valid, useful answer to RQ1 — but a risk to
expectations. Don't let "we hoped ML would win" quietly bias how the
Results section is written if it doesn't.

**[act now] yfinance rate limiting hit twice already during setup.** Not
currently blocking, but recurring. Since `END_DATE` is frozen
(2026-07-07), there's no reason to ever re-pull from Yahoo again this
project — the raw data in `data/raw/` should be treated as final. Confirm
`data_pipeline.py` isn't being re-run unnecessarily (e.g. as part of a
larger "run everything" script) close to the deadline, when a rate-limit
delay would be most costly.

**[watch] Recurring Excel file-lock friction (CSV open while Claude Code
writes it).** Minor, but has happened twice and will keep happening if the
processed CSVs stay open in Excel during sessions. Low cost to just
develop the habit of closing data files before starting a Claude Code
session.

**[watch] Zero-volume anomaly (2023-05-24) is explained speculatively**
("likely a data-provider quirk") in the LaTeX. That's an honest
characterization of genuine uncertainty, not a weakness — just don't let a
future edit accidentally upgrade it to a confident claim we can't support.

**[watch] Feature-set separation (RQ1 vs. hybrid) — re-verify again once
the hybrid model is built.** `CLAUDE.md` specifies pure ML features for RQ1
(no GARCH/EWMA inputs) and GARCH-informed features only for the hybrid
model. Checked at RF/XGBoost implementation time: `RQ1_ML_FEATURES` in
`features.py` is a named, explicit constant (lagged/absolute/squared
returns + 5/10/20-day historical vol only), `ml_models.py` imports it
rather than re-listing columns, and the underlying dataframe never even
contains a GARCH/EWMA-named column to leak in accidentally. Still worth a
second look once `hybrid.py` exists and GARCH-derived features enter the
codebase for real — that's when the two sets will actually sit side by
side.

**[watch] Timeline — hybrid model is last for a reason.** If the ML stage
runs long (tuning rabbit hole, debugging `arch`-style scale issues in a new
library, etc.), the hybrid model is the one that should shrink or drop, not
the core RQ1 comparison or the write-up window (Sept 8–13 per the original
plan). Revisit this list before making that call, not after.

---

## Future extensions backlog

**[do now] Diebold-Mariano test for pairwise model comparison.** Directly
strengthens the core Results section rather than being a bolt-on. Do once
all models (econometric, ML, hybrid) have gone through the same walk-forward
harness — one round of pairwise tests instead of redoing it per stage.

**[if time permits] Student-t (or another fat-tailed) VaR as a robustness
check alongside the Gaussian baseline.** Already flagged as optional in
`CLAUDE.md`. Worth doing if the core VaR/backtesting section is done early
and there's real slack — directly answers the Gaussian-limitation risk
above with evidence instead of just a caveat sentence.

**[if time permits] Acerbi-Szekely backtest for Expected Shortfall**,
instead of the simpler "average shortfall beyond VaR" approach. More
citable, moderate additional effort.

**[if time permits] Parkinson or Garman-Klass realized volatility as a
robustness check.** The raw data already has High/Low, so this uses data
already collected rather than requiring a new pull — check whether the
5-day close-to-close target's conclusions hold under a range-based
volatility estimator too.

**[if time permits] Basic feature-importance / SHAP summary for the ML and
hybrid models.** Evaluators tend to respond well to a model not being a
black box — even a simple built-in feature-importance plot from
RF/XGBoost adds interpretability to the Results section for relatively low
cost.

**[cut unless way ahead] Regime-split analysis** (e.g. does GJR-GARCH's
advantage hold pre- vs. post-2023, or around specific volatility regimes).
Interesting, adds real depth, but is genuinely extra scope beyond what the
thesis needs to answer RQ1/RQ2 — only worth it if the base thesis, hybrid
model, and write-up are all done with meaningful time still on the clock.

**[cut unless way ahead] Second asset / multi-index robustness check.**
Would meaningfully strengthen external validity but is a significant time
cost (new data pipeline, re-run everything) for a thesis that's explicitly
scoped as single-asset per the tutor's own recommendation to bound scope
early. Do not start this unless everything else, including the write-up,
is finished.

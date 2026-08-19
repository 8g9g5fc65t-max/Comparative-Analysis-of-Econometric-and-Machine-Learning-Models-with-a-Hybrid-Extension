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

**RESOLVED (2026-08-13) — RQ1 feature timing: return features were lagged a
day further than the spec.** `lag_return_1/lag_abs_return_1/lag_sq_return_1`
were built with `.shift(1)`, i.e. r_{t-1}, while `hist_vol_5d/10d/20d` beside
them correctly used a window ending at t — so the six-feature set mixed two
information cutoffs, and contradicted the thesis's own r_t / |r_t| / r_t^2
notation. **Not leakage**: it was too conservative, handing the models less
information than advertised, which is why it survived every earlier
no-leakage check. Fixed by dropping the shift and renaming the columns to
`today_return` / `today_abs_return` / `today_sq_return` so the names cannot
imply a shift that isn't there. Verified after the change: all six features
reconstructible from returns ≤ t, target still built from r_{t+1}..r_{t+5}
only, label masking intact, newest training row still ≥5 rows before t.
*Effect*: every RF/XGBoost metric improved (RMSE ~5%, MAE ~2%) and **the
ranking changed again — Random Forest took the best MAE from GJR-GARCH**
(0.04341 vs 0.04358). Econometric numbers bitwise unchanged.

**[act now] The Overleaf write-up describes pre-fix numbers throughout.**
Results, Conclusions and §5.2 still quote the original ranking (XGBoost best
MAE, "no single model dominates" justified by XGBoost). Also stale and needing
the regenerated values: the forecast-error figure paragraph (now 2025-04-02,
XGBoost forecast **13.3%** vs realised 86.3%, under by **73.0pp**) and the ES
excess range (8–21% → **8–24%**). The ML feature table should now read r_t /
|r_t| / r_t^2 — which, after the feature-timing fix, matches the code exactly
for the first time. The Christoffersen sentence ("all five pass, p > 0.32")
needs **no** change: the non-overlap minimum is 0.3216 again.

**[act now] The hybrid's tail-calibration cost is the headline RQ2 finding —
don't let it get written up as a win.** The hybrid beats GJR-GARCH on MAE
(−7.8%) and RMSE (−3.5%) and takes first place on both, which is an easy
result to over-claim. It simultaneously **worsens QLIKE by 12.5%** and turns
GJR-GARCH's comfortable 99% Kupiec pass (p=0.6754, 10 violations) into a
failure (p=0.0129, 17 violations). The mechanism is identified, not guessed:
GJR-GARCH runs hot on average (mean forecast 0.1425 vs mean actual 0.1285), so
the residual model learned an almost uniform downward shading — negative on
91% of days, mean −0.0171 — which shrinks symmetric error but raises
under-prediction from 32.7% to 46.3% of days. QLIKE and VaR coverage both
punish exactly that. The ML tail weakness demonstrably transfers: the hybrid's
99% violation set shares 15 of 17 days with XGBoost and 16 with Random Forest,
but only 9 with its own GJR-GARCH baseline. This is the tutor's "assess
whether combining actually helps, not just assume it does" question answered
with evidence in both directions — write it that way.

**TRIED AND FAILED (2026-08-18) — quantile-loss residual model does not fix
the hybrid's tail calibration.** Direct follow-up to the item above: since VaR
at confidence α is the (1−α) quantile of the loss distribution, the residual
model was retrained with XGBoost's `reg:quantileerror` (pinball loss) at an
upper quantile, so the training objective matches the downstream use instead
of targeting the mean. Level 0.95 selected on a 2011–2019 / 2020–2022 inner
split of the training period; test set truncated off the frame before fitting,
truncation asserted.
**Result: it overshot and failed in the opposite direction.** 99% Kupiec goes
from 17 violations (1.95%, p=0.0129) to 2 (0.23%, p=0.0057) — still failing,
marginally worse by the statistic, now from over-conservatism. 95% Kupiec
collapses from a comfortable pass to p=0.0000 (0.80% vs nominal 5%). Accuracy
cost is severe: MAE +142%, RMSE +90%, QLIKE +62% versus the symmetric hybrid.
The correction is positive on 100% of days (mean +0.0752) where the symmetric
one was negative on 91%.
**Why**, and it is not a bug: the residual quantile is regime-dependent.
Inner-validation (2020–2022) has mean realised vol 0.1995 against 0.1285 in
2023–2026 — a 36% calmer test period — so a 95th-percentile residual learned
under COVID-era stress is far too wide later. Two selection artefacts worth
recording: the first candidate grid (0.5–0.9) returned its own upper boundary,
so it was extended to 0.95/0.99 to bracket the optimum; and the original
criterion ("highest 99% Kupiec p") proved degenerate — it is only bounded by
the two-sided test, so it drove toward alpha=0.99 at 3× the MAE. It was
replaced with "lowest inner-validation MAE among Kupiec-passing candidates".
Both changes used validation data only, but adapting a selection rule after
seeing validation results is itself mild selection-on-validation and should be
one sentence in the write-up.
**For the thesis**: this is a genuine negative result and worth keeping.
Objective alignment alone is not sufficient when the residual distribution
shifts regime between training and deployment. Note also that **GJR-GARCH
alone still has the best 99% calibration of any model (p=0.6754)** — neither
hybrid variant beats it there.

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
flips the result completely: independence p-values move to **0.32-0.51** for
every model, and every model has **zero** consecutive violations in the
subsample (n11=0). The original full-sample LR statistics (59-116,
against a chi2(1) critical value of ~3.84) were almost entirely the
overlapping-horizon artifact, not a per-model signal.
(Both ranges restated 2026-08-13 against the final post-fix tables. They moved
transiently mid-session — after the walk-forward fix the p-range was 0.27-0.51
— and landed back at 0.32-0.51 once the feature-timing fix went in. The
conclusion never changed, and the econometric numbers in it never moved.)
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

**RESOLVED (2026-08-18) — feature-set separation, now that the two sets
actually sit side by side.** This was the open re-verification item for when
`hybrid.py` exists. Checked at build time: `RQ1_ML_FEATURES` is still exactly
six columns and still contains no GARCH/EWMA/GJR-named input;
`HYBRID_ML_FEATURES` is constructed as `list(RQ1_ML_FEATURES) + ["gjr_forecast"]`,
so it *extends* the RQ1 constant rather than restating it and the two cannot
drift apart. Both `ml_models.py` and `hybrid.py` import the constant. The
hybrid's GARCH-derived input is deliberate and confined to the hybrid, which
is exactly the separation the rule asked for. Verified assertions live in the
build-time checks, not just in prose.

**RESOLVED (2026-08-18) — hybrid residual label is masked like any other.**
`gjr_residual` inherits `target_rv_5d`'s forward-looking window, so it is
registered in `features.FORWARD_LABEL_COLS` and the harness masks it
automatically; `mask_unknown_labels()` skips absent columns, so the
registration works even though `build_features()` never creates it. Verified:
last 5 residuals masked at an arbitrary t, `gjr_forecast` correctly left
unmasked (made at t, not forward-looking), newest hybrid training row still
≥5 rows before t. Freshness verified with the same behavioural test used on
RF/XGBoost: fit once, predict across six dates → 6/6 distinct, no
DataFrame/Series/ndarray cached on the model.

**[watch] Hybrid training residuals are in-sample, and it shows.** The
training-period baseline comes from ONE in-sample GJR-GARCH fit over
2011–2022 (a deliberate cost saving — the component is not being
re-validated), so those residuals are smaller and easier than the
out-of-sample errors the model actually meets. Not leakage: every training
date precedes the entire 2023–2026 test window, so nothing a test forecast is
built from postdates it. But it biases the residual model toward
under-correcting, and is a fair thing for an evaluator to ask about — state it
in the methodology rather than let it be found. If the hybrid result is ever
challenged, generating the training residuals by walk-forward instead is the
obvious robustness check (at real compute cost).

**[watch] Timeline — hybrid model is last for a reason.** If the ML stage
runs long (tuning rabbit hole, debugging `arch`-style scale issues in a new
library, etc.), the hybrid model is the one that should shrink or drop, not
the core RQ1 comparison or the write-up window (Sept 8–13 per the original
plan). Revisit this list before making that call, not after.

---

## Future extensions backlog

**[do now — now UNBLOCKED] Diebold-Mariano test for pairwise model
comparison.** Directly strengthens the core Results section rather than being
a bolt-on. The precondition (all models through the same walk-forward
harness) is met as of 2026-08-18: all six, hybrid included, are in
`results/tables/forecasts_all_models.csv`, so this is one pairwise pass with
no walk-forward re-run needed. Two gaps it would now settle: Random Forest
beats GJR-GARCH on MAE by only 0.4%, and the hybrid beats GJR-GARCH on RMSE
by 3.5% while losing 12.5% on QLIKE — neither is obviously significant.

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

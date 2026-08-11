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

**[act now] No significance testing on model comparisons yet.** The
econometric table (GJR-GARCH < GARCH < EWMA on MAE/RMSE/QLIKE) shows ranked
numbers but not whether the differences are statistically meaningful. A
Diebold-Mariano test is the standard tool for this. Cheap to add, and it's
exactly the kind of "diagnostics and robustness" the TFM guide's evaluation
checklist asks for. See backlog — do this once all models (econometric + ML
+ hybrid) are in, so it's one pairwise comparison pass instead of three.

**[watch] Gaussian VaR underestimates tail risk.** Parametric Gaussian VaR
is the agreed baseline (fast, consistent across models, easy to defend) but
financial returns have fatter tails than Gaussian assumes — VaR will likely
under-cover during genuine crisis periods (e.g. anything COVID-like in the
test window). This isn't a bug, it's a known limitation of the chosen
method. Say so explicitly in the conclusions/limitations section rather
than let an evaluator find it unstated — a named limitation reads as rigor,
an unnamed one reads as an oversight.

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

"""Hybrid model (RQ2): GJR-GARCH baseline + XGBoost residual correction.

    sigma_hybrid(t) = sigma_GJR(t) + e_ML(t)

where e_ML is an XGBoost estimate of what GJR-GARCH is about to get wrong.
Conforms to the walk-forward interface (src/walkforward.py):
    fit(history)      -> None   # re-estimate the residual model's parameters
    predict(history)  -> float  # forecast for history's LAST date

DESIGN (chosen deliberately, see CLAUDE.md -- do not substitute):
  - Baseline is GJR-GARCH, not GARCH(1,1): it is the strongest econometric
    model in this thesis, so "does ML add value on top?" is asked against the
    best available benchmark rather than a weaker one that is easier to beat.
  - Residual learner is XGBoost, not Random Forest: boosting already works by
    sequentially fitting each tree to the previous stage's residuals, so
    "predict what GJR-GARCH got wrong" is the same shape of problem.
  - Features are the six RQ1 features PLUS the GJR-GARCH forecast for the same
    date (HYBRID_ML_FEATURES). The RQ1 anti-circularity rule does not apply
    here and is not being violated: feeding the econometric forecast to the ML
    component is the entire mechanism of a hybrid model. That rule exists to
    keep the RQ1 horse race clean, and RQ1's feature set is untouched --
    RQ1_ML_FEATURES is imported, not redefined, so the two sets cannot drift.

UNITS -- checked explicitly, because a silent scale mismatch here is the same
class of bug as the arch RETURN_SCALE**2 rescaling caught earlier. All three
of target_rv_5d, the GJR-GARCH forecast and hence the residual live on the
5-day annualised volatility scale sqrt((252/5) * sum of 5 squared returns).
econometric.annualize_5day_vol() and features._annualized_rv() are the same
formula applied forward vs. backward, so the subtraction is dimensionally
sound. build_hybrid_frame() asserts the ranges agree rather than trusting it.

WHERE THE GJR-GARCH FORECASTS COME FROM (two different provenances, on
purpose):
  - Test period (2023-2026): the already-validated walk-forward forecasts are
    reused as-is, passed in by the caller. Not recomputed -- that component
    cost real effort to validate and a reimplementation could drift from it.
  - Training period (2011-2022): ONE in-sample fit over the whole training
    window (econometric.in_sample_forecasts), not a walk-forward re-run. This
    is the deliberate cost saving; the component is not being re-validated.
    The consequence is stated plainly in build_hybrid_frame's docstring: those
    residuals are in-sample and therefore optimistically small.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import (RQ1_ML_FEATURES, TARGET_COL, HYBRID_RESIDUAL_COL,
                      TrainTestSplit)
from models.econometric import GJRGARCHModel

RANDOM_SEED = 42

# Quantile level for HybridGJRQuantileModel, chosen by src/quantile_selection.py
# on a 2011-2019 / 2020-2022 inner split of the TRAINING period only. Criterion:
# lowest inner-validation MAE among the candidates passing the 99% Kupiec test
# (0.95 and 0.99 passed; 0.95 was far more accurate). The 2023-2026 test set was
# not consulted. Re-run that script if the training data or the baseline ever
# change -- do not hand-tune this against a test-set result.
SELECTED_QUANTILE_ALPHA = 0.95

# The GJR-GARCH forecast for date t, used as the 7th input to the residual
# model. Made AT t from data through t, so it is not forward-looking and must
# NOT be added to features.FORWARD_LABEL_COLS (the residual built from it is).
GJR_FORECAST_COL = "gjr_forecast"

# Six RQ1 features + the econometric forecast. RQ1_ML_FEATURES is imported so
# there is exactly one definition of the RQ1 six; this list extends it rather
# than restating it.
HYBRID_ML_FEATURES = list(RQ1_ML_FEATURES) + [GJR_FORECAST_COL]


def build_hybrid_frame(df, test_gjr_forecasts, split=None):
    """Attach GJR_FORECAST_COL and HYBRID_RESIDUAL_COL to `df`.

    df: features.build_features() output.
    test_gjr_forecasts: DataFrame with Date + forecast for the test period --
        the walk-forward GJR-GARCH run, reused rather than recomputed.

    Training rows get their GJR forecast from a single in-sample fit over the
    training window; test rows get the walk-forward forecast passed in.

    The in-sample half is optimistic: those parameters were estimated using the
    whole 2011-2022 window, so a 2012 row's "forecast" is informed by data from
    2013 onwards, and its residual is smaller than a genuine out-of-sample one
    would be. That is a deliberate, documented simplification of the training
    target, NOT leakage into the evaluation: every training date involved lies
    entirely before 2023, so nothing a test-date forecast is built from
    postdates that forecast. It does mean the residual model is trained on
    errors that are easier than the ones it meets out of sample, which biases
    it towards under-correcting -- worth knowing when reading the results.
    """
    split = split or TrainTestSplit()
    train, _ = split.split(df)

    baseline = GJRGARCHModel()
    baseline.fit(train)
    in_sample = baseline.in_sample_forecasts()

    # Align on Date, never on position. in_sample is indexed by the subset of
    # `train`'s rows that survived fit()'s dropna() -- the first row has no
    # log_return and is absent -- so a positional slice would be off by one and
    # would silently shift every training residual by a day.
    train_dates = train.loc[in_sample.index, "Date"]
    gjr_map = dict(zip(train_dates, in_sample.to_numpy()))

    test_map = dict(zip(test_gjr_forecasts["Date"], test_gjr_forecasts["forecast"]))
    overlap = set(gjr_map) & set(test_map)
    if overlap:
        raise ValueError(f"{len(overlap)} date(s) have both an in-sample and a "
                         f"walk-forward GJR forecast, e.g. {sorted(overlap)[:3]}")
    gjr_map.update(test_map)

    out = df.copy()
    out[GJR_FORECAST_COL] = out["Date"].map(gjr_map)
    train_mask = out["Date"].isin(set(train_dates))
    test_mask = out["Date"].isin(set(test_map))

    _assert_same_scale(out, train_mask, test_mask)

    out[HYBRID_RESIDUAL_COL] = out[TARGET_COL] - out[GJR_FORECAST_COL]
    return out


def _assert_same_scale(frame, train_mask, test_mask):
    """Fail loudly if the GJR forecasts and the target are not on the same
    scale. A factor-of-100 (or 10,000) slip would otherwise pass silently and
    only show up as an absurd hybrid forecast much later."""
    target = frame[TARGET_COL].dropna()
    for label, mask in (("in-sample (train)", train_mask), ("walk-forward (test)", test_mask)):
        fc = frame.loc[mask, GJR_FORECAST_COL].dropna()
        if fc.empty:
            raise ValueError(f"no GJR-GARCH forecasts for the {label} rows")
        ratio = fc.median() / target.median()
        if not 0.2 < ratio < 5:
            raise ValueError(
                f"GJR-GARCH {label} forecasts look mis-scaled against {TARGET_COL}: "
                f"median forecast {fc.median():.6g} vs median target {target.median():.6g} "
                f"(ratio {ratio:.4g}). Both must be on the 5-day annualised vol scale.")


# Shared across both variants: identical learner capacity, identical seed, so
# the ONLY thing separating them is the training objective.
_XGB_BASE_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    n_jobs=1,
)


class _HybridBase:
    """Shared plumbing for both hybrid variants.

    Holds no input data between calls -- `fit` stores only fitted parameters,
    and `predict` reads both the feature row and the baseline forecast out of
    the `history` it is handed. Both variants therefore inherit the
    walk-forward contract's two guarantees (fresh daily prediction, no future
    label) by construction rather than by re-implementing them.

    Subclasses supply the estimator via _build_estimator(); everything else --
    features, residual target, additive combination -- is identical, which is
    what makes the symmetric-vs-quantile comparison a clean one-variable test.
    """

    def __init__(self):
        self._model = self._build_estimator()

    def _build_estimator(self):
        raise NotImplementedError

    def fit(self, history):
        train = history.dropna(subset=HYBRID_ML_FEATURES + [HYBRID_RESIDUAL_COL])
        self._model.fit(train[HYBRID_ML_FEATURES], train[HYBRID_RESIDUAL_COL])

    def predict(self, history):
        row = history.iloc[[-1]]
        baseline = float(row[GJR_FORECAST_COL].iloc[0])
        correction = float(self._model.predict(row[HYBRID_ML_FEATURES])[0])
        return baseline + correction


class HybridGJRXGBoostModel(_HybridBase):
    """Hybrid (symmetric): residual model trained under squared-error loss.

    Hyperparameters mirror models/ml_models.XGBoostModel so the hybrid's ML
    component is the same learner evaluated standalone in RQ1; any difference
    in results comes from the target it is trained on (a GJR-GARCH residual
    rather than volatility itself) and the extra feature, not from tuning.

    Known behaviour (2026-08-18): because squared-error loss targets the MEAN
    residual and GJR-GARCH runs hot on average, this learns a near-uniform
    DOWNWARD shading -- better MAE/RMSE, worse QLIKE, and it breaks the
    baseline's 99% Kupiec calibration. That diagnosis is what motivated the
    quantile variant below.
    """

    def _build_estimator(self):
        # objective left at XGBoost's default reg:squarederror -- stated
        # explicitly here only because it is the variable under test.
        return XGBRegressor(objective="reg:squarederror", **_XGB_BASE_PARAMS)


class HybridGJRQuantileModel(_HybridBase):
    """Hybrid (quantile): residual model trained under pinball loss.

    Motivation, not an arbitrary retune: VaR at confidence alpha IS the
    (1-alpha) quantile of the loss distribution, so a residual model whose
    training objective is a quantile of the residual distribution is aligned
    with the downstream use. Targeting an UPPER quantile of GJR-GARCH's
    residual distribution asks "how far might GJR-GARCH under-forecast?"
    rather than "how far does it miss on average", which is the question tail
    risk actually poses.

    Everything else is identical to the symmetric variant -- same baseline,
    same 7 features, same capacity, same seed, same weekly cadence. The single
    changed input is the objective (and its quantile level).

    quantile_alpha is selected on an inner-train/inner-validation split of the
    TRAINING period only (src/quantile_selection.py); the 2023-2026 test set
    plays no part in choosing it.
    """

    def __init__(self, quantile_alpha):
        if not 0 < quantile_alpha < 1:
            raise ValueError(f"quantile_alpha must be in (0, 1), got {quantile_alpha}")
        self.quantile_alpha = quantile_alpha
        super().__init__()

    def _build_estimator(self):
        # reg:quantileerror is XGBoost's native pinball-loss objective
        # (available from XGBoost 2.0; this project pins 3.2.0).
        # quantile_alpha is accepted as a passthrough kwarg rather than a named
        # parameter of XGBRegressor.__init__, which is why it is set here.
        return XGBRegressor(objective="reg:quantileerror",
                            quantile_alpha=self.quantile_alpha,
                            **_XGB_BASE_PARAMS)

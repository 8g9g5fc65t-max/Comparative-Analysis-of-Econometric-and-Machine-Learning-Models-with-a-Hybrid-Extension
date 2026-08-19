"""Econometric volatility models: EWMA, GARCH(1,1), GJR-GARCH.

All three conform to the walk-forward interface (see src/walkforward.py):
    fit(history: pd.DataFrame) -> None      # re-estimate parameters
    predict(history: pd.DataFrame) -> float # 5-day annualised vol for history's LAST date

All three go through the same annualize_5day_vol() helper so their forecasts
land on exactly the scale as features.TARGET_COL (target_rv_5d) -- required
for a fair comparison later.

None of these models reads a label column, so the harness's label masking is a
no-op for them; they were never exposed to the leakage it prevents. They take
`history` in predict() anyway, because a single uniform contract across every
model family is what stops the two concerns (parameter cadence vs. forecast
freshness) from being conflated again -- see walkforward.py's module docstring.
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.utility.exceptions import ConvergenceWarning

TRADING_DAYS_PER_YEAR = 252
TARGET_WINDOW = 5
RETURN_SCALE = 100  # arch's optimizer wants returns on an ~O(1) scale, not ~O(0.01)


def annualize_5day_vol(daily_variances):
    """Shared scale conversion for all three models: given daily variance
    forecast(s) sigma^2_{t+1}..sigma^2_{t+TARGET_WINDOW} (on the raw
    log-return scale, not percent), return the annualised 5-day-ahead vol
        sqrt((252/5) * sum(daily_variances))
    -- the same formula used to build target_rv_5d from realised squared
    returns in features.py, just applied to model-implied variances. A
    scalar is broadcast across all TARGET_WINDOW days (EWMA has no variance
    term structure, so its 1-step forecast is repeated).
    """
    variances = np.broadcast_to(daily_variances, (TARGET_WINDOW,))
    return float(np.sqrt((TRADING_DAYS_PER_YEAR / TARGET_WINDOW) * np.sum(variances)))


class EWMAModel:
    """RiskMetrics-style EWMA: sigma2_t = lam*sigma2_{t-1} + (1-lam)*r_{t-1}^2.
    No external library, no mean reversion -- the 5-day forecast is just the
    latest 1-step variance estimate repeated for all 5 days.

    lam is a fixed convention (RiskMetrics 0.94), not an estimated parameter,
    so fit() has nothing to do: the whole model is the recursion, and the
    recursion is a filter over returns rather than a parameter estimate. Running
    it inside predict() makes this model correct at ANY refit cadence -- its
    forecast tracks returns through the current date whether or not fit() was
    called today.
    """

    def __init__(self, lam=0.94):
        self.lam = lam

    def fit(self, history):
        """No parameters to estimate -- lam is fixed by convention."""

    def predict(self, history):
        r = history["log_return"].dropna().values
        var = r[0] ** 2  # seed value; decays away after ~a few hundred obs
        for x in r[1:]:
            var = self.lam * var + (1 - self.lam) * x ** 2
        return annualize_5day_vol(var)


class _ArchGarchModel:
    """Shared plumbing for the two `arch`-backed models below. `o` selects
    plain GARCH (o=0) vs. GJR-GARCH's asymmetric term (o=1)."""

    def __init__(self, o=0, dist="normal"):
        self.o = o
        self.dist = dist
        self.n_convergence_warnings = 0
        self._result = None
        self._fitted_through = None

    def fit(self, history):
        r = history["log_return"].dropna() * RETURN_SCALE
        am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=self.o, q=1, dist=self.dist)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._result = am.fit(disp="off", show_warning=False)
        self.n_convergence_warnings += sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning))
        self._fitted_through = history["Date"].iloc[-1]

    def predict(self, history):
        # Unlike EWMA and the ML models, an arch forecast is produced from the
        # fitted result object, whose conditional-variance filter is bound to
        # the sample it was estimated on -- it cannot be rolled forward to a
        # later date without refitting. So this model is only correct under a
        # daily refit cadence, and says so loudly instead of quietly returning
        # a forecast that ignores everything since the last fit. That silent
        # staleness is exactly the bug the walkforward.py contract now exists
        # to prevent (rule 1); this guard is how THIS model honours it.
        t = history["Date"].iloc[-1]
        if self._fitted_through != t:
            raise RuntimeError(
                f"{type(self).__name__} was fitted through {self._fitted_through} but "
                f"asked to forecast for {t}. arch-backed models must be refit on every "
                f"forecasting date (refit_frequency='daily'); a stale fit would silently "
                f"ignore all returns since it was estimated.")
        fc = self._result.forecast(horizon=TARGET_WINDOW, reindex=False)
        variances_pct2 = fc.variance.values[-1]  # on the RETURN_SCALE^2 scale
        variances = variances_pct2 / (RETURN_SCALE ** 2)
        return annualize_5day_vol(variances)

    def in_sample_forecasts(self):
        """5-day annualised vol forecast for EVERY date in the fitted sample,
        from the single fit already performed -- one call to arch instead of a
        refit per date.

        Row t is the forecast made at t for t+1..t+TARGET_WINDOW, on exactly
        the scale predict() returns and exactly the scale features.TARGET_COL
        is built on, so `TARGET_COL - this` is a residual in vol units.
        Returns a Series indexed like the fitted sample.

        IN-SAMPLE by construction: parameters were estimated using the whole
        fitted window, so row t's forecast is informed by data after t. Only
        use this to build a training target inside a period that is entirely
        in the past relative to every date being forecast (models/hybrid.py
        uses it on 2011-2022 only, to train a residual model later applied to
        2023-2026). Never use it to score forecast accuracy.
        """
        if self._result is None:
            raise RuntimeError("fit() must be called before in_sample_forecasts()")
        # start=0 is load-bearing: without it arch forecasts only from the END
        # of the sample and returns a single non-NaN row. start=0 makes every
        # in-sample date a forecast origin, which is what a per-date residual
        # needs. The returned index is the fitted sample's index -- i.e. the
        # caller's rows MINUS any dropped by fit()'s dropna() -- so align on it
        # rather than assuming it covers every input row.
        fc = self._result.forecast(horizon=TARGET_WINDOW, start=0, reindex=True)
        variances = fc.variance.to_numpy() / (RETURN_SCALE ** 2)
        annualised = np.sqrt((TRADING_DAYS_PER_YEAR / TARGET_WINDOW)
                              * variances.sum(axis=1))
        return pd.Series(annualised, index=fc.variance.index)


class GARCHModel(_ArchGarchModel):
    """GARCH(1,1) via the `arch` package."""

    def __init__(self):
        super().__init__(o=0)


class GJRGARCHModel(_ArchGarchModel):
    """GJR-GARCH(1,1) via `arch` (o=1 adds the asymmetric/leverage term)."""

    def __init__(self):
        super().__init__(o=1)

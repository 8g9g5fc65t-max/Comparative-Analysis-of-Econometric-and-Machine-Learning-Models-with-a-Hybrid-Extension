"""Econometric volatility models: EWMA, GARCH(1,1), GJR-GARCH.

All three conform to the walk-forward interface (see src/walkforward.py):
    fit(history: pd.DataFrame) -> None   # history has a 'log_return' column
    predict() -> float                    # 5-day-ahead annualised vol forecast

All three go through the same annualize_5day_vol() helper so their forecasts
land on exactly the scale as features.TARGET_COL (target_rv_5d) -- required
for a fair comparison later.
"""
import warnings

import numpy as np
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
    """

    def __init__(self, lam=0.94):
        self.lam = lam
        self._var = None

    def fit(self, history):
        r = history["log_return"].dropna().values
        var = r[0] ** 2  # seed value; decays away after ~a few hundred obs
        for x in r[1:]:
            var = self.lam * var + (1 - self.lam) * x ** 2
        self._var = var

    def predict(self):
        return annualize_5day_vol(self._var)


class _ArchGarchModel:
    """Shared plumbing for the two `arch`-backed models below. `o` selects
    plain GARCH (o=0) vs. GJR-GARCH's asymmetric term (o=1)."""

    def __init__(self, o=0, dist="normal"):
        self.o = o
        self.dist = dist
        self.n_convergence_warnings = 0
        self._result = None

    def fit(self, history):
        r = history["log_return"].dropna() * RETURN_SCALE
        am = arch_model(r, mean="Zero", vol="GARCH", p=1, o=self.o, q=1, dist=self.dist)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self._result = am.fit(disp="off", show_warning=False)
        self.n_convergence_warnings += sum(
            1 for w in caught if issubclass(w.category, ConvergenceWarning))

    def predict(self):
        fc = self._result.forecast(horizon=TARGET_WINDOW, reindex=False)
        variances_pct2 = fc.variance.values[-1]  # on the RETURN_SCALE^2 scale
        variances = variances_pct2 / (RETURN_SCALE ** 2)
        return annualize_5day_vol(variances)


class GARCHModel(_ArchGarchModel):
    """GARCH(1,1) via the `arch` package."""

    def __init__(self):
        super().__init__(o=0)


class GJRGARCHModel(_ArchGarchModel):
    """GJR-GARCH(1,1) via `arch` (o=1 adds the asymmetric/leverage term)."""

    def __init__(self):
        super().__init__(o=1)

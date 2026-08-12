"""Gaussian VaR and Expected Shortfall at the 5-day forecast horizon.

Applied identically to every model's forecast (all on features.TARGET_COL's
annualised-5-day-vol scale) -- same de-annualisation, same VaR/ES formulas,
no per-model special-casing. That uniformity is what lets "risk-measure
differences trace back to volatility-forecast differences" (CLAUDE.md's
Methodology) actually hold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import TARGET_WINDOW, TRADING_DAYS_PER_YEAR

REPO_ROOT = Path(__file__).resolve().parent.parent
FORECASTS_TABLE_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"

CONFIDENCE_LEVELS = (0.95, 0.99)


def deannualize_5day_vol(annualized_vol):
    """Inverse of features.py's annualisation (the sqrt(252/5) factor):
    turns a model's target_rv_5d-scale forecast back into raw, non-
    annualised 5-day volatility -- the scale VaR/ES are defined on."""
    return np.asarray(annualized_vol) * np.sqrt(TARGET_WINDOW / TRADING_DAYS_PER_YEAR)


def gaussian_var(sigma_5day, confidence):
    """Parametric Gaussian VaR, zero-mean assumption: z_alpha * sigma.
    Returned as a positive loss magnitude (CLAUDE.md's sigma_hat * z_alpha)."""
    z = norm.ppf(confidence)
    return z * sigma_5day


def gaussian_es(sigma_5day, confidence):
    """Closed-form Gaussian Expected Shortfall, zero-mean assumption:
    sigma * phi(z_alpha) / (1 - alpha). Positive loss magnitude."""
    z = norm.ppf(confidence)
    return sigma_5day * norm.pdf(z) / (1 - confidence)


def compute_risk_measures(annualized_vol_forecast, confidence_levels=CONFIDENCE_LEVELS):
    """annualized_vol_forecast: array-like of forecasts on features.TARGET_COL's
    scale -- any of the five models, same function, same formula every time.
    Returns a DataFrame with VaR_{pct}/ES_{pct} columns per confidence level
    (pct = 95, 99, ...), preserving the input's index if it's a Series."""
    sigma_5day = deannualize_5day_vol(annualized_vol_forecast)
    index = annualized_vol_forecast.index if isinstance(annualized_vol_forecast, pd.Series) else None

    out = {}
    for c in confidence_levels:
        pct = round(c * 100)
        out[f"VaR_{pct}"] = gaussian_var(sigma_5day, c)
        out[f"ES_{pct}"] = gaussian_es(sigma_5day, c)
    return pd.DataFrame(out, index=index)


def main():
    forecasts = pd.read_csv(FORECASTS_TABLE_PATH, parse_dates=["Date"])
    model_cols = [c for c in forecasts.columns if c not in ("Date", "actual")]

    print(f"Confidence levels: {CONFIDENCE_LEVELS}")
    for name in model_cols:
        risk = compute_risk_measures(forecasts[name])
        stats = {col: (risk[col].min(), risk[col].mean(), risk[col].max())
                 for col in risk.columns}
        print(f"\n{name}:")
        for col, (lo, mean, hi) in stats.items():
            print(f"  {col}: min={lo:.4f} mean={mean:.4f} max={hi:.4f}")


if __name__ == "__main__":
    sys.exit(main())

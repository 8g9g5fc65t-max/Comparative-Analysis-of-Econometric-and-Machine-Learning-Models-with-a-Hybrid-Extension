"""VaR backtesting (Kupiec, Christoffersen) and a simple ES backtest.

A violation on date t: the actual 5-day forward return (features.TARGET_RET_COL,
signed) is worse than -VaR_t, i.e. target_ret_5d < -VaR_t. Runs identically
for all five models at both confidence levels (risk.py's uniformity carries
through here too -- no per-model special-casing).

References: Kupiec (1995) unconditional coverage; Christoffersen (1998)
independence and conditional coverage. Both are likelihood-ratio tests,
chi-squared distributed under the null.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, load_processed, TARGET_RET_COL
from risk import compute_risk_measures, CONFIDENCE_LEVELS

REPO_ROOT = Path(__file__).resolve().parent.parent
FORECASTS_TABLE_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"
BACKTEST_SUMMARY_PATH = REPO_ROOT / "results" / "tables" / "backtest_summary.csv"
ROBUSTNESS_CHECK_PATH = REPO_ROOT / "results" / "tables" / "christoffersen_nonoverlap_check.csv"


def _log_likelihood(prob, successes, total):
    """n*log(p) terms, with the n=0 case short-circuited to 0 instead of
    evaluating log(p) -- avoids 0*log(0)=nan when a probability estimate is
    exactly 0 or 1 (which happens whenever there are zero violations, or
    zero non-violations, in a sample)."""
    failures = total - successes
    term_success = 0.0 if successes == 0 else successes * np.log(prob)
    term_failure = 0.0 if failures == 0 else failures * np.log(1 - prob)
    return term_success + term_failure


def kupiec_test(violations, alpha):
    """Kupiec (1995) unconditional coverage test: is the overall violation
    rate consistent with the VaR confidence level? violations: bool
    array/Series (True = VaR breached). alpha: VaR confidence (e.g. 0.95) --
    expected violation probability is 1-alpha."""
    v = np.asarray(violations, dtype=bool)
    n = len(v)
    x = int(v.sum())
    p = 1 - alpha
    phat = x / n if n else np.nan

    ll_null = _log_likelihood(p, x, n)
    ll_alt = _log_likelihood(phat, x, n)
    LR = -2 * (ll_null - ll_alt)
    p_value = float(1 - chi2.cdf(LR, df=1))
    return {"n": n, "n_violations": x, "violation_rate": phat, "expected_rate": p,
            "kupiec_LR": LR, "kupiec_pvalue": p_value}


def christoffersen_independence_test(violations):
    """Christoffersen (1998) independence test: are violations clustered in
    time, rather than scattered independently? A model can have exactly the
    right overall rate (passes Kupiec) while still failing this -- e.g. all
    its violations bunched in one crisis week."""
    v = np.asarray(violations, dtype=int)
    prev, curr = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) else 0.0

    ll_restricted = _log_likelihood(pi, n01 + n11, n00 + n01 + n10 + n11)
    ll_unrestricted = (_log_likelihood(pi01, n01, n00 + n01)
                        + _log_likelihood(pi11, n11, n10 + n11))
    LR = -2 * (ll_restricted - ll_unrestricted)
    p_value = float(1 - chi2.cdf(LR, df=1))
    return {"christoffersen_indep_LR": LR, "christoffersen_indep_pvalue": p_value,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11}


def conditional_coverage_test(kupiec_result, indep_result):
    """Christoffersen conditional coverage: LR_cc = LR_uc + LR_ind, chi2(2)
    under the null (correct rate AND independent)."""
    LR_cc = kupiec_result["kupiec_LR"] + indep_result["christoffersen_indep_LR"]
    p_value = float(1 - chi2.cdf(LR_cc, df=2))
    return {"christoffersen_cc_LR": LR_cc, "christoffersen_cc_pvalue": p_value}


def es_backtest(actual_returns, es_forecast, violations):
    """Simple ES backtest (CLAUDE.md: 'average shortfall beyond VaR vs. ES
    forecast'), restricted to violation days. realized_shortfall = -actual
    return (a positive loss magnitude) is compared against the model's own
    forecast ES on those same days. es_ratio near 1 = well-calibrated;
    >1 = ES understates realised tail losses; <1 = overstates them."""
    v = np.asarray(violations, dtype=bool)
    if v.sum() == 0:
        return {"es_mean_realized_shortfall": np.nan, "es_mean_forecast": np.nan,
                "es_ratio": np.nan}
    realized_shortfall = -np.asarray(actual_returns)[v]
    forecast_es_on_violations = np.asarray(es_forecast)[v]
    mean_realized = float(np.mean(realized_shortfall))
    mean_forecast = float(np.mean(forecast_es_on_violations))
    return {"es_mean_realized_shortfall": mean_realized,
            "es_mean_forecast": mean_forecast,
            "es_ratio": mean_realized / mean_forecast if mean_forecast else np.nan}


def run_backtests(forecasts, features_df, confidence_levels=CONFIDENCE_LEVELS):
    """forecasts: forecasts_all_models.csv (Date, actual, <model columns...>).
    features_df: build_features() output, needs Date + TARGET_RET_COL.
    Returns one row per (model, confidence level)."""
    model_names = [c for c in forecasts.columns if c not in ("Date", "actual")]
    merged = forecasts.merge(features_df[["Date", TARGET_RET_COL]], on="Date")
    # Same forward-looking tail edge case as target_rv_5d (see features.py):
    # the last TARGET_WINDOW test dates have no realised 5-day return yet.
    merged = merged.dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    actual_ret = merged[TARGET_RET_COL]

    rows = []
    for name in model_names:
        risk = compute_risk_measures(merged[name], confidence_levels)
        for c in confidence_levels:
            pct = round(c * 100)
            violations = (actual_ret < -risk[f"VaR_{pct}"]).values

            row = {"model": name, "confidence": c}
            row.update(kupiec_test(violations, c))
            indep = christoffersen_independence_test(violations)
            row.update(indep)
            row.update(conditional_coverage_test(row, indep))
            row.update(es_backtest(actual_ret, risk[f"ES_{pct}"].values, violations))
            rows.append(row)

    return pd.DataFrame(rows)


def christoffersen_non_overlapping_check(forecasts, features_df, confidence=0.95, stride=5):
    """Robustness check on the full-sample Christoffersen independence
    result. target_ret_5d at consecutive test dates shares 4 of its 5
    underlying daily returns (confirmed directly: r[i+1..i+5] vs.
    r[i+2..i+6]) -- that overlap mechanically induces serial correlation in
    the violation indicator regardless of whether the volatility process
    itself clusters. Subsampling every `stride` dates (default 5, matching
    the horizon) removes the overlap entirely: consecutive checks in the
    subsample come from target_ret_5d windows that share zero underlying
    returns, isolating genuine clustering from the mechanical artifact.

    Reuses christoffersen_independence_test() and compute_risk_measures()
    unmodified -- doesn't touch kupiec_test, es_backtest, or risk.py.
    """
    model_names = [c for c in forecasts.columns if c not in ("Date", "actual")]
    merged = forecasts.merge(features_df[["Date", TARGET_RET_COL]], on="Date")
    merged = merged.dropna(subset=[TARGET_RET_COL]).reset_index(drop=True)
    subsample = merged.iloc[::stride].reset_index(drop=True)
    actual_ret = subsample[TARGET_RET_COL]
    pct = round(confidence * 100)

    rows = []
    for name in model_names:
        risk = compute_risk_measures(subsample[name], (confidence,))
        violations = (actual_ret < -risk[f"VaR_{pct}"]).values
        row = {"model": name, "confidence": confidence, "n": len(violations),
               "n_violations": int(violations.sum()),
               "violation_rate": float(violations.mean()) if len(violations) else float("nan")}
        row.update(christoffersen_independence_test(violations))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    forecasts = pd.read_csv(FORECASTS_TABLE_PATH, parse_dates=["Date"])
    features_df = build_features(load_processed())

    summary = run_backtests(forecasts, features_df)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    BACKTEST_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BACKTEST_SUMMARY_PATH, index=False)
    print(f"\nSaved: {BACKTEST_SUMMARY_PATH}")

    robustness = christoffersen_non_overlapping_check(forecasts, features_df)
    print("\nNon-overlapping robustness check (every 5th test date, 95% confidence):")
    print(robustness.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    robustness.to_csv(ROBUSTNESS_CHECK_PATH, index=False)
    print(f"\nSaved: {ROBUSTNESS_CHECK_PATH}")


if __name__ == "__main__":
    sys.exit(main())

"""Select the quantile level for the Hybrid (quantile) residual model.

The choice is made ENTIRELY inside the 2011-2022 training period, on a
chronological inner-train / inner-validation split. The 2023-2026 test set is
never read: the frame is truncated at features.SPLIT_DATE before anything is
fitted, and that truncation is asserted, not assumed.

Design of the search (kept small and bounded on purpose -- this is a
methodology step, not a tuning exercise):
  - Inner-train      : 2011-01-03 .. INNER_SPLIT_DATE
  - Inner-validation : INNER_SPLIT_DATE+1 .. end of the training period
  - Candidates       : 0.5 (sanity baseline == median/pinball-at-0.5) and
                       0.6 / 0.7 / 0.8 / 0.9
  - Selection criterion (CONSTRAINED, see the revision note below): among
    candidates that PASS the 99% Kupiec test on inner-validation at the 5%
    level, take the one with the lowest inner-validation MAE. Tail calibration
    is the constraint the variant exists to satisfy; forecast accuracy is what
    is then optimised subject to it. Under-prediction rate and the 95% Kupiec
    result are reported as further diagnostics.

REVISION NOTE, stated because it matters for how much weight the choice bears.
The criterion originally registered here was the simpler "highest 99% Kupiec
p-value". Run against the grid it proved degenerate: the p-value rises almost
monotonically with the quantile level, because a higher quantile means a more
conservative forecast, a wider VaR and fewer violations. Nothing in that rule
penalises over-forecasting until the two-sided Kupiec test finally objects, so
it selected alpha=0.99 -- which does calibrate (1.06% violations) but at an
inner-validation MAE of 0.2073 against 0.0690 for alpha=0.5, i.e. a forecast
roughly three times less accurate and larger than the quantity being forecast.
That is an upper envelope, not a volatility forecast. The constrained criterion
above encodes the guard the original rationale already claimed the secondary
diagnostics provided, and it was fixed before the test set was touched. Both
the criterion revision and the grid extension used inner-validation data only;
the 2023-2026 test set played no part in either. The honest caveat is that
adapting a selection rule after seeing validation results is itself a mild form
of selection on validation -- worth a sentence in the write-up, not a reason to
distrust the result.

Each candidate is evaluated through the SAME walk-forward harness used for the
real test run (weekly refit, expanding window), just with the split cutoff
moved back to INNER_SPLIT_DATE. So the selection sees the model behaving the
way it will behave in production, and inherits the harness's label-masking and
prediction-freshness guarantees.

Caveat worth stating in the write-up: the inner-validation slice contains the
COVID crash, an extreme volatility regime with no counterpart in 2023-2026.
Choosing a tail-calibration parameter on it is defensible -- tail behaviour is
best judged under stress -- but it is not a neutral sample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (build_features, load_processed, TrainTestSplit,
                      SPLIT_DATE, TARGET_COL, TARGET_RET_COL)
from walkforward import run_walkforward
from evaluation import mae, rmse
from risk import compute_risk_measures
from backtesting import kupiec_test
from models.hybrid import build_hybrid_frame, HybridGJRQuantileModel

REPO_ROOT = Path(__file__).resolve().parent.parent
SELECTION_TABLE_PATH = REPO_ROOT / "results" / "tables" / "hybrid_quantile_selection.csv"
FORECASTS_TABLE_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"

# Chronological inner split. Chosen so inner-validation (2020-2022, ~750 rows)
# is comparable in length to the real test period (879 rows) and immediately
# precedes it, rather than being carved out of the distant past.
INNER_SPLIT_DATE = "2019-12-31"
# The first pass used (0.5, 0.6, 0.7, 0.8, 0.9) and returned 0.9 -- the top of
# the grid -- with the criterion still improving monotonically across every
# candidate. A boundary solution means the grid failed to BRACKET the optimum,
# so 0.95 and 0.99 were appended to close it. That extension is a grid-design
# fix, not a re-tune: it uses inner-validation only, the criterion was fixed
# beforehand, and the test set is still untouched. Both passes are reported.
CANDIDATE_QUANTILES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
SELECTION_CONFIDENCE = 0.99


def training_only_frame(df, test_gjr_forecasts):
    """Hybrid frame truncated to the training period, with the truncation
    asserted. Nothing downstream of this function can see a test-set row."""
    hybrid = build_hybrid_frame(df, test_gjr_forecasts)
    train_only = hybrid[hybrid["Date"] <= SPLIT_DATE].reset_index(drop=True)
    if train_only["Date"].max() > pd.Timestamp(SPLIT_DATE):
        raise AssertionError("training-only frame leaked a post-split row")
    return train_only


def evaluate_candidate(train_only, quantile_alpha):
    """Walk-forward the candidate over the inner-validation slice and score it."""
    split = TrainTestSplit(cutoff=INNER_SPLIT_DATE)
    result = run_walkforward(train_only, HybridGJRQuantileModel(quantile_alpha),
                              refit_frequency="weekly", split=split)

    merged = result.merge(train_only[["Date", TARGET_RET_COL]], on="Date")
    merged = merged.dropna(subset=["actual", TARGET_RET_COL]).reset_index(drop=True)

    risk = compute_risk_measures(merged["forecast"], (SELECTION_CONFIDENCE,))
    pct = round(SELECTION_CONFIDENCE * 100)
    violations = (merged[TARGET_RET_COL] < -risk[f"VaR_{pct}"]).to_numpy()
    kupiec = kupiec_test(violations, SELECTION_CONFIDENCE)

    risk95 = compute_risk_measures(merged["forecast"], (0.95,))
    viol95 = (merged[TARGET_RET_COL] < -risk95["VaR_95"]).to_numpy()
    kupiec95 = kupiec_test(viol95, 0.95)

    return {
        "quantile_alpha": quantile_alpha,
        "n_inner_val": len(merged),
        "MAE": mae(merged["actual"], merged["forecast"]),
        "RMSE": rmse(merged["actual"], merged["forecast"]),
        "underprediction_rate": float((merged["actual"] > merged["forecast"]).mean()),
        "mean_correction": float((merged["forecast"]
                                  - merged["gjr_forecast"]).mean())
        if "gjr_forecast" in merged else np.nan,
        "n_violations_99": kupiec["n_violations"],
        "violation_rate_99": kupiec["violation_rate"],
        "kupiec_LR_99": kupiec["kupiec_LR"],
        "kupiec_pvalue_99": kupiec["kupiec_pvalue"],
        "n_violations_95": kupiec95["n_violations"],
        "violation_rate_95": kupiec95["violation_rate"],
        "kupiec_pvalue_95": kupiec95["kupiec_pvalue"],
    }


KUPIEC_ALPHA = 0.05  # significance level at which a candidate must pass


def select(table):
    """Constrained selection: of the candidates that pass the 99% Kupiec test
    on inner-validation, return the most accurate (lowest MAE). Falls back to
    the highest Kupiec p-value if none passes, so the function still returns
    something defensible on a validation slice where nothing calibrates."""
    passing = table[table["kupiec_pvalue_99"] > KUPIEC_ALPHA]
    if passing.empty:
        return float(table.sort_values("kupiec_pvalue_99", ascending=False)
                     .iloc[0]["quantile_alpha"]), False
    best = passing.sort_values(["MAE", "quantile_alpha"]).iloc[0]
    return float(best["quantile_alpha"]), True


def main():
    df = build_features(load_processed())
    forecasts = pd.read_csv(FORECASTS_TABLE_PATH, parse_dates=["Date"])
    gjr_col = "GJR-GARCH"
    test_gjr = forecasts[["Date", gjr_col]].rename(columns={gjr_col: "forecast"})

    train_only = training_only_frame(df, test_gjr)
    inner_train = train_only[train_only["Date"] <= INNER_SPLIT_DATE]
    inner_val = train_only[train_only["Date"] > INNER_SPLIT_DATE]
    print(f"Training period only: {len(train_only)} rows, "
          f"{train_only['Date'].min().date()} to {train_only['Date'].max().date()}")
    print(f"  inner-train     : {len(inner_train)} rows, "
          f"{inner_train['Date'].min().date()} to {inner_train['Date'].max().date()}")
    print(f"  inner-validation: {len(inner_val)} rows, "
          f"{inner_val['Date'].min().date()} to {inner_val['Date'].max().date()}")
    print(f"  test set ({SPLIT_DATE}+) not present in this frame: "
          f"{train_only['Date'].max() <= pd.Timestamp(SPLIT_DATE)}\n")

    rows = []
    for q in CANDIDATE_QUANTILES:
        row = evaluate_candidate(train_only, q)
        rows.append(row)
        print(f"  alpha={q:.2f}  MAE={row['MAE']:.5f}  RMSE={row['RMSE']:.5f}  "
              f"under={100*row['underprediction_rate']:.1f}%  "
              f"99% viol={100*row['violation_rate_99']:.2f}% "
              f"(n={row['n_violations_99']}, p={row['kupiec_pvalue_99']:.4f})", flush=True)

    table = pd.DataFrame(rows)
    table["passes_kupiec_99"] = table["kupiec_pvalue_99"] > KUPIEC_ALPHA
    chosen, constrained = select(table)
    table["selected"] = table["quantile_alpha"] == chosen

    SELECTION_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(SELECTION_TABLE_PATH, index=False)
    passing = table.loc[table["passes_kupiec_99"], "quantile_alpha"].tolist()
    print(f"\n  candidates passing 99% Kupiec (p>{KUPIEC_ALPHA}): {passing or 'NONE'}")
    print(f"Selected quantile_alpha = {chosen}")
    print("  criterion: " + ("lowest inner-validation MAE among Kupiec-passing candidates"
                              if constrained else
                              "FALLBACK - no candidate passed; highest Kupiec p-value"))
    print(f"Saved: {SELECTION_TABLE_PATH}")
    return chosen


if __name__ == "__main__":
    main()

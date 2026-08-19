"""Forecast evaluation: MAE, RMSE, QLIKE.

MAE and RMSE are computed directly on the volatility scale (forecast vs.
target_rv_5d). QLIKE is defined on VARIANCE, not volatility -- squaring
both series before computing it is the definition (Poon & Granger, 2003),
not a stylistic choice. Don't "fix" it back to volatility scale.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, load_processed
from walkforward import run_walkforward
from models.econometric import EWMAModel, GARCHModel, GJRGARCHModel
from models.ml_models import RandomForestModel, XGBoostModel
from models.hybrid import (build_hybrid_frame, HybridGJRXGBoostModel,
                           HybridGJRQuantileModel, SELECTED_QUANTILE_ALPHA)

REPO_ROOT = Path(__file__).resolve().parent.parent
# Renamed from econometric_comparison.csv now that ML models (and eventually
# the hybrid model) share this table -- one evolving comparison, not one
# file per model family.
COMPARISON_TABLE_PATH = REPO_ROOT / "results" / "tables" / "model_comparison.csv"
# Per-date (Date, actual, <model forecasts...>) for every model, from the
# same walk-forward run that builds the table above -- so a later
# Diebold-Mariano test (or anything else needing raw forecasts) doesn't
# have to re-run the slow walk-forward loops a second time.
FORECASTS_TABLE_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"


def align(result_df):
    """Drop rows with no actual target (the TARGET_WINDOW tail rows -- see
    features.py's forward-target edge case) before computing any metric, so
    NaNs never silently propagate into a mean."""
    return result_df.dropna(subset=["actual"])


def mae(actual, forecast):
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual, forecast):
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def qlike(actual, forecast):
    """QLIKE on VARIANCE. `actual`/`forecast` arrive on the annualised-vol
    scale (same as target_rv_5d); square them into variance before taking
    the ratio -- that squaring is not optional.
    """
    actual_var = actual ** 2
    forecast_var = forecast ** 2
    ratio = actual_var / forecast_var
    return float(np.mean(ratio - np.log(ratio) - 1))


def evaluate(result_df):
    """result_df: walk-forward output with 'actual'/'forecast' columns.
    Returns {'MAE', 'RMSE', 'QLIKE', 'n_obs'}."""
    clean = align(result_df)
    return {
        "MAE": mae(clean["actual"], clean["forecast"]),
        "RMSE": rmse(clean["actual"], clean["forecast"]),
        "QLIKE": qlike(clean["actual"], clean["forecast"]),
        "n_obs": len(clean),
    }


def compare_models(results):
    """results: {model_name: walk-forward output DataFrame}. Returns a
    DataFrame with one row per model, MAE/RMSE/QLIKE/n_obs as columns."""
    table = pd.DataFrame.from_dict(
        {name: evaluate(df) for name, df in results.items()}, orient="index")
    table["n_obs"] = table["n_obs"].astype(int)
    table.index.name = "model"
    return table[["MAE", "RMSE", "QLIKE", "n_obs"]]


def combine_forecasts(results):
    """results: {model_name: walk-forward output DataFrame (Date, forecast,
    actual)}. Returns one wide DataFrame -- Date, actual, <model_1>,
    <model_2>, ... -- one row per test date. Every model runs the same
    split against the same target_rv_5d, so their (Date, actual) pairs must
    be identical; asserted here so a future alignment bug fails loudly
    instead of silently merging mismatched rows."""
    names = list(results.keys())
    base = results[names[0]][["Date", "actual"]].reset_index(drop=True)
    combined = base.copy()
    for name in names:
        aligned = results[name][["Date", "actual"]].reset_index(drop=True)
        if not aligned.equals(base):
            raise ValueError(f"{name}'s (Date, actual) don't match the other "
                              f"models' -- walk-forward alignment bug")
        combined[name] = results[name]["forecast"].reset_index(drop=True).values
    return combined


def main():
    df = build_features(load_processed())

    # Refit cadence per CLAUDE.md: econometric models refit daily (cheap);
    # ML models refit weekly.
    econometric_models = {
        "EWMA": EWMAModel(),
        "GARCH(1,1)": GARCHModel(),
        "GJR-GARCH": GJRGARCHModel(),
    }
    ml_models = {
        "RandomForest": RandomForestModel(),
        "XGBoost": XGBoostModel(),
    }
    results = {name: run_walkforward(df, model, refit_frequency="daily")
               for name, model in econometric_models.items()}
    results.update({name: run_walkforward(df, model, refit_frequency="weekly")
                     for name, model in ml_models.items()})

    # Hybrid (RQ2): GJR-GARCH baseline + XGBoost on its residuals. The test
    # -period baseline forecasts are REUSED from the GJR-GARCH walk-forward
    # just run above rather than recomputed -- same single validated run that
    # gets written to forecasts_all_models.csv, so the hybrid cannot drift from
    # the standalone GJR-GARCH column it is meant to be built on. The ML
    # component refits weekly, like the other two ML models.
    # Two variants sharing everything except the residual model's objective:
    #   symmetric -> squared-error loss, targets the MEAN residual
    #   quantile  -> pinball loss at SELECTED_QUANTILE_ALPHA, targets an upper
    #                quantile of the residual distribution, which is what VaR
    #                (itself a quantile of the loss distribution) actually needs
    hybrid_df = build_hybrid_frame(df, results["GJR-GARCH"])
    results["Hybrid (symmetric)"] = run_walkforward(
        hybrid_df, HybridGJRXGBoostModel(), refit_frequency="weekly")
    results["Hybrid (quantile)"] = run_walkforward(
        hybrid_df, HybridGJRQuantileModel(SELECTED_QUANTILE_ALPHA),
        refit_frequency="weekly")

    table = compare_models(results)
    print(table.to_string(float_format=lambda x: f"{x:.5f}"))

    COMPARISON_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_TABLE_PATH)
    print(f"\nSaved: {COMPARISON_TABLE_PATH}")

    forecasts = combine_forecasts(results)
    forecasts.to_csv(FORECASTS_TABLE_PATH, index=False)
    print(f"Saved: {FORECASTS_TABLE_PATH} ({len(forecasts)} rows, "
          f"{len(results)} model columns)")


if __name__ == "__main__":
    sys.exit(main())

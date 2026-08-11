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

REPO_ROOT = Path(__file__).resolve().parent.parent
# Renamed from econometric_comparison.csv now that ML models (and eventually
# the hybrid model) share this table -- one evolving comparison, not one
# file per model family.
COMPARISON_TABLE_PATH = REPO_ROOT / "results" / "tables" / "model_comparison.csv"


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

    table = compare_models(results)
    print(table.to_string(float_format=lambda x: f"{x:.5f}"))

    COMPARISON_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_TABLE_PATH)
    print(f"\nSaved: {COMPARISON_TABLE_PATH}")


if __name__ == "__main__":
    sys.exit(main())

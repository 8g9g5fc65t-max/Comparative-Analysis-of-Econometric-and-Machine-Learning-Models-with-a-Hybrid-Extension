"""Volatility target, RQ1 ML feature set, and train/test split.

Reads data/processed/gspc_processed.csv (from data_pipeline.py) and builds:
  - target_rv_5d: 5-day-forward realised volatility (the forecasting target)
  - RQ1_ML_FEATURES: exactly six columns -- one lagged return (r_{t-1}),
    one lagged absolute return (|r_{t-1}|), one lagged squared return
    (r_{t-1}^2), and 5/10/20-day trailing historical vol. This is
    CLAUDE.md's RQ1 feature set (pure ML vs. econometric horse race) and
    matches the finalised thesis methodology text exactly -- no multi-lag
    expansion. Deliberately excludes any GARCH/EWMA-derived input: mixing
    those in would make the RQ1 comparison circular. GARCH-informed
    features belong only to the hybrid model (RQ2), built separately.
The target and hist_vol_* share one annualised realised-vol formula,
sqrt((252/n) * sum r^2), just applied looking forward vs. backward.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "gspc_processed.csv"
FEATURES_PATH = REPO_ROOT / "data" / "processed" / "gspc_features.csv"

TARGET_WINDOW = 5
TARGET_COL = "target_rv_5d"
HIST_VOL_WINDOWS = (5, 10, 20)
TRADING_DAYS_PER_YEAR = 252

# RQ1 ML feature set (CLAUDE.md, "kept separate to avoid circularity", and
# matching the finalised thesis methodology text exactly): one lagged
# return, one lagged absolute return, one lagged squared return, plus
# 5/10/20-day historical vol -- SIX columns, no multi-lag expansion. No
# GARCH/EWMA inputs here -- those are hybrid-model-only. Named and explicit
# so downstream code (ml_models.py, hybrid.py) imports this list instead of
# re-deriving or blurring the RQ1-vs-hybrid feature split.
RQ1_ML_FEATURES = [
    "lag_return_1",
    "lag_abs_return_1",
    "lag_sq_return_1",
] + [f"hist_vol_{w}d" for w in HIST_VOL_WINDOWS]

# Train/test cutoff (locked in per CLAUDE.md): train = Date <= SPLIT_DATE,
# test = Date > SPLIT_DATE. Single source of truth -- walkforward.py and any
# other script should import SPLIT_DATE from here rather than hardcoding it.
SPLIT_DATE = "2022-12-31"


def load_processed(path=PROCESSED_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def _annualized_rv(r2, window):
    """sqrt((252/window) * sum of squared returns over `window` days)."""
    return np.sqrt((TRADING_DAYS_PER_YEAR / window) * r2.rolling(window).sum())


def build_features(df):
    df = df.copy()
    r2 = df["log_return"] ** 2

    # Forward target: needs r_{t+1}..r_{t+5}, i.e. the trailing 5-day sum
    # evaluated 5 rows ahead, shifted back onto row t. The last TARGET_WINDOW
    # rows have no future returns to sum, so they come out NaN -- expected,
    # not a bug: there is no 5-day-ahead target for the last 5 trading days
    # in the sample (walk-forward/evaluation code must drop or ignore them).
    fwd_sum_r2 = r2.rolling(TARGET_WINDOW).sum().shift(-TARGET_WINDOW)
    df[TARGET_COL] = np.sqrt((TRADING_DAYS_PER_YEAR / TARGET_WINDOW) * fwd_sum_r2)

    # Trailing historical vol (uses r_{t-window+1}..r_t only -- no leakage).
    for w in HIST_VOL_WINDOWS:
        df[f"hist_vol_{w}d"] = _annualized_rv(r2, w)

    # Lagged/absolute/squared return, lag 1 only (RQ1 ML feature set).
    # shift(1) uses only r_{t-1} -- strictly past information, no leakage.
    df["lag_return_1"] = df["log_return"].shift(1)
    df["lag_abs_return_1"] = df["log_return"].abs().shift(1)
    df["lag_sq_return_1"] = r2.shift(1)

    return df


@dataclass
class TrainTestSplit:
    """Date-based split: train = Date <= cutoff, test = Date > cutoff."""
    cutoff: str = SPLIT_DATE

    def split(self, df):
        train = df[df["Date"] <= self.cutoff].reset_index(drop=True)
        test = df[df["Date"] > self.cutoff].reset_index(drop=True)
        # The last TARGET_WINDOW train rows have a target that sums returns
        # past the cutoff (row i's target uses r_{i+1}..r_{i+5}), so trim
        # them -- every remaining training target is boundary-clean.
        train = train.iloc[:-TARGET_WINDOW]
        return train, test


def print_summary(df, train, test):
    n_tail_nan = int(df[TARGET_COL].isna().sum())
    print("Full sample:", len(df), "rows,", df["Date"].min().date(), "to",
          df["Date"].max().date())
    print("NaN target rows at tail (no 5-day-ahead data yet):", n_tail_nan)
    print(f"Train: {len(train)} rows (boundary-clean, {TARGET_WINDOW} rows "
          f"trimmed off the end), {train['Date'].min().date()} to "
          f"{train['Date'].max().date()} "
          f"({train[TARGET_COL].notna().sum()} with a valid target)")
    print(f"Test:  {len(test)} rows, {test['Date'].min().date()} to "
          f"{test['Date'].max().date()} "
          f"({test[TARGET_COL].notna().sum()} with a valid target)")


def main():
    df = build_features(load_processed())
    train, test = TrainTestSplit().split(df)
    print_summary(df, train, test)

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False)
    print(f"\nFeatures saved: {FEATURES_PATH}")


if __name__ == "__main__":
    sys.exit(main())

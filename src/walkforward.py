"""Expanding-window walk-forward engine -- the harness every model plugs into.

At each test-period date t: refit (per the model's cadence) on the expanding
window of data up to and including t, then ask for a forecast of target_rv_5d
at t. Output is a (date, forecast, actual) DataFrame for evaluation.py.

Model interface (duck-typed, no base class -- econometric and ML models
share this without inheriting anything):
    model.fit(history: pd.DataFrame) -> None   # history = df[Date <= t]
    model.predict() -> float                    # forecast for date t

`history` carries every column in df, including target_rv_5d for past dates
-- a model must never use that column as an input feature (it's the label).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_features, load_processed, TrainTestSplit, TARGET_COL

REFIT_FREQUENCIES = ("daily", "weekly")


def run_walkforward(df, model, refit_frequency="daily", split=None, target_col=TARGET_COL):
    """Walk forward over the test period, returning a DataFrame of
    Date / forecast / actual."""
    if refit_frequency not in REFIT_FREQUENCIES:
        raise ValueError(
            f"refit_frequency must be one of {REFIT_FREQUENCIES}, got {refit_frequency!r}")

    split = split or TrainTestSplit()
    _, test = split.split(df)

    records = []
    current_week = None
    for _, row in test.iterrows():
        t = row["Date"]
        history = df[df["Date"] <= t]

        if refit_frequency == "daily":
            should_refit = True
        else:  # weekly: refit on the first test date seen in each ISO week
            week_key = t.isocalendar()[:2]  # (ISO year, ISO week)
            should_refit = week_key != current_week
            current_week = week_key

        if should_refit:
            model.fit(history)
        forecast = model.predict()

        records.append({"Date": t, "forecast": forecast, "actual": row[target_col]})

    return pd.DataFrame(records)


class NaiveLastValueModel:
    """Trivial baseline used only to smoke-test the engine: forecasts
    tomorrow's target as today's 5-day trailing historical vol. Not one of
    the thesis models -- those live in src/models/."""

    def __init__(self):
        self._last_hist_vol = None

    def fit(self, history):
        self._last_hist_vol = history["hist_vol_5d"].iloc[-1]

    def predict(self):
        return self._last_hist_vol


def main():
    df = build_features(load_processed())

    for freq in REFIT_FREQUENCIES:
        result = run_walkforward(df, NaiveLastValueModel(), refit_frequency=freq)
        print(f"refit_frequency={freq}: {len(result)} rows, "
              f"{result['Date'].min().date()} to {result['Date'].max().date()}, "
              f"{result['actual'].notna().sum()} rows with a valid actual")


if __name__ == "__main__":
    sys.exit(main())

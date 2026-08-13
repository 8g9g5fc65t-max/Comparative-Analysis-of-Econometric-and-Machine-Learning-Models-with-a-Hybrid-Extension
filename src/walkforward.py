"""Expanding-window walk-forward engine -- the harness every model plugs into.

At each test-period date t the harness builds `available_history(df, t)`: every
row up to and including t, with the forward-looking label columns of the last
TARGET_WINDOW rows masked to NaN because those labels have not happened yet.
That single frame is handed to both fit() and predict(), so a model physically
cannot reach information that did not exist at t.

Model interface (duck-typed, no base class -- econometric and ML models
share this without inheriting anything):
    model.fit(history: pd.DataFrame) -> None      # re-estimate parameters
    model.predict(history: pd.DataFrame) -> float # forecast for history's LAST date

TWO RULES, and the interface exists to enforce both structurally:

1. PREDICTION FRESHNESS. predict() is called on *every* test date and must
   derive its answer from the `history` it is handed on that call. It must
   never read inputs cached during fit(). refit_frequency controls how often
   PARAMETERS are re-estimated -- it must not control how often the forecast
   is refreshed. (predict() taking `history` explicitly, rather than models
   stashing a feature vector during fit(), is precisely what makes the two
   concerns impossible to conflate again: there is no stale state to read.)
   A model whose parameters are structurally tied to their estimation sample
   -- the arch-backed GARCH family -- must assert that its fit is current for
   the passed history rather than silently return a stale forecast; see
   models/econometric.py.

2. NO FUTURE LABELS. Training rows must never carry a label whose window
   reaches past t. The harness guarantees this by masking, so models just
   dropna() on the label column as usual and get the right training set for
   free. This is features.TrainTestSplit.split()'s train/test boundary rule
   applied continuously instead of once.

Both rules were violated before 2026-08-13 (weekly-cadence ML models froze one
forecast per ISO week, and every training set leaked five forward-reaching
labels including the very row being predicted); the interface is shaped the way
it is so neither can be reintroduced by a model added later.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (build_features, load_processed, mask_unknown_labels,
                      TrainTestSplit, TARGET_COL)

REFIT_FREQUENCIES = ("daily", "weekly")


def available_history(df, t):
    """Everything genuinely knowable at forecasting date t: all rows up to and
    including t, with labels that reach past t masked out (features.py's
    mask_unknown_labels). The only frame the harness ever gives a model."""
    return mask_unknown_labels(df[df["Date"] <= t])


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
        history = available_history(df, t)

        if refit_frequency == "daily":
            should_refit = True
        else:  # weekly: refit on the first test date seen in each ISO week
            week_key = t.isocalendar()[:2]  # (ISO year, ISO week)
            should_refit = week_key != current_week
            current_week = week_key

        if should_refit:
            model.fit(history)
        # Unconditional, and outside the refit branch on purpose: the forecast
        # is refreshed every day regardless of cadence (rule 1 above).
        forecast = model.predict(history)

        records.append({"Date": t, "forecast": forecast, "actual": row[target_col]})

    return pd.DataFrame(records)


class NaiveLastValueModel:
    """Trivial baseline used only to smoke-test the engine: forecasts
    tomorrow's target as today's 5-day trailing historical vol. Not one of
    the thesis models -- those live in src/models/.

    Holds no state between calls, which is the point: predict() reads the
    current history it is given."""

    def fit(self, history):
        """No parameters to estimate."""

    def predict(self, history):
        return float(history["hist_vol_5d"].iloc[-1])


def main():
    df = build_features(load_processed())

    for freq in REFIT_FREQUENCIES:
        result = run_walkforward(df, NaiveLastValueModel(), refit_frequency=freq)
        print(f"refit_frequency={freq}: {len(result)} rows, "
              f"{result['Date'].min().date()} to {result['Date'].max().date()}, "
              f"{result['actual'].notna().sum()} rows with a valid actual")


if __name__ == "__main__":
    sys.exit(main())

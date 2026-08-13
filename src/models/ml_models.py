"""ML volatility models: Random Forest, XGBoost.

Both conform to the walk-forward interface (see src/walkforward.py):
    fit(history: pd.DataFrame) -> None      # history has all of features.py's columns
    predict(history: pd.DataFrame) -> float # forecast for history's LAST date

Unlike the econometric models (which forecast forward from a fitted time
series), these are plain supervised regressors: fit(X, y) on every row of
history where both the RQ1 feature set and the target are available, then
predict(history) applies the fitted model to the *last* row's feature vector --
the forecasting date's own features, X(t), used to predict y(t) = target_rv_5d(t).

These models are refit weekly but forecast daily, so predict() reads X(t) out
of the history it is handed on each call. Neither class stores a feature
vector, or any other input, between calls: with nothing cached there is nothing
to go stale, which is what keeps a weekly refit cadence from silently becoming
a weekly forecast (walkforward.py, rule 1 -- this was a real bug here until
2026-08-13, when predict() took no arguments and reused the feature row that
fit() had stashed, freezing each model's output for a whole ISO week).

Training labels come pre-masked by the harness (walkforward.available_history),
so the dropna() below cannot retain a row whose target reaches past the
forecasting date -- rule 2, and the reason _training_frame needs no boundary
logic of its own.

Only features.RQ1_ML_FEATURES go in as X -- explicitly imported rather than
listed here, so there's one place the RQ1-vs-hybrid feature split can drift
from, not two.
"""
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import RQ1_ML_FEATURES, TARGET_COL

RANDOM_SEED = 42


def _training_frame(history):
    """Rows of history with every RQ1 feature and the target present --
    drops the leading rows before all lag/rolling windows are full, and the
    trailing rows whose target reaches past the forecasting date (the harness
    masks those to NaN before handing `history` over, so they fall out here)."""
    return history.dropna(subset=list(RQ1_ML_FEATURES) + [TARGET_COL])


def _latest_features(history):
    """X(t) for history's last row -- a 1-row DataFrame so the estimator sees
    the same feature names it was fitted with. Read fresh on every predict()
    call; deliberately not cached anywhere."""
    return history[RQ1_ML_FEATURES].iloc[[-1]]


class RandomForestModel:
    """sklearn's RandomForestRegressor. Hyperparameters set explicitly, not
    left at library defaults -- sklearn's own default n_estimators changed
    10 -> 100 across versions (see requirements.txt/Part-0 environment
    notes), which is exactly the kind of silent drift this project got
    bitten by once already. max_depth/min_samples_leaf are capped modestly
    given ~3,000 training rows (docs/risks_and_roadmap.md flags tree-model
    overfitting risk on a sample this size).
    """

    def __init__(self):
        self._model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            # n_jobs=1, not -1: at this row count a fresh joblib process pool
            # spun up on every one of ~184 weekly refits is pure overhead, and
            # on this sklearn/joblib combo it also floods stdout with a
            # benign-but-relentless UserWarning (thousands of repeats in
            # testing) -- both go away single-threaded, for no real cost.
            n_jobs=1,
        )

    def fit(self, history):
        train = _training_frame(history)
        self._model.fit(train[RQ1_ML_FEATURES], train[TARGET_COL])

    def predict(self, history):
        return float(self._model.predict(_latest_features(history))[0])


class XGBoostModel:
    """xgboost's XGBRegressor. Hyperparameters set explicitly for the same
    reason as RandomForestModel above -- shallower trees than the RF
    (max_depth=4) since boosting compounds many weak trees rather than
    averaging deep ones, with a modest learning rate and row/column
    subsampling as further overfitting guards on a ~3,000-row sample.
    """

    def __init__(self):
        self._model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=1,  # single-threaded: same overhead/determinism reasoning
                       # as RandomForestModel above, kept consistent across both.
        )

    def fit(self, history):
        train = _training_frame(history)
        self._model.fit(train[RQ1_ML_FEATURES], train[TARGET_COL])

    def predict(self, history):
        return float(self._model.predict(_latest_features(history))[0])

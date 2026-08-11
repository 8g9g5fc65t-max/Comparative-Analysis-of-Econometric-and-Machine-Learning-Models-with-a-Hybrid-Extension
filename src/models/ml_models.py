"""ML volatility models: Random Forest, XGBoost.

Both conform to the walk-forward interface (see src/walkforward.py):
    fit(history: pd.DataFrame) -> None   # history has all of features.py's columns
    predict() -> float                    # forecast for the last row's date in history

Unlike the econometric models (which forecast forward from a fitted time
series), these are plain supervised regressors: fit(X, y) on every row of
history where both the RQ1 feature set and the target are available, then
predict() applies the fitted model to the *last* row's feature vector --
the forecasting date's own features, X(t), used to predict y(t) = target_rv_5d(t).

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
    drops the leading rows before all lag/rolling windows are full and any
    trailing rows whose target hasn't happened yet (see features.py)."""
    return history.dropna(subset=list(RQ1_ML_FEATURES) + [TARGET_COL])


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
        self._latest_X = None

    def fit(self, history):
        train = _training_frame(history)
        self._model.fit(train[RQ1_ML_FEATURES], train[TARGET_COL])
        self._latest_X = history[RQ1_ML_FEATURES].iloc[[-1]]

    def predict(self):
        return float(self._model.predict(self._latest_X)[0])


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
        self._latest_X = None

    def fit(self, history):
        train = _training_frame(history)
        self._model.fit(train[RQ1_ML_FEATURES], train[TARGET_COL])
        self._latest_X = history[RQ1_ML_FEATURES].iloc[[-1]]

    def predict(self):
        return float(self._model.predict(self._latest_X)[0])

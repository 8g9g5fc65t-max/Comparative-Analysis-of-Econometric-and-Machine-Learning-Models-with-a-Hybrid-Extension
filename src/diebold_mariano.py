"""Diebold-Mariano tests for pairwise forecast-accuracy comparison.

Scoped to the two comparisons where the ranking is close enough that the raw
numbers do not settle it:
  - GJR-GARCH vs. Random Forest        (RF wins MAE by 0.4%)
  - GJR-GARCH vs. Hybrid (symmetric)   (hybrid wins MAE/RMSE, loses QLIKE)

Run for all three loss functions the thesis reports, so each DM result lines up
with a column of results/tables/model_comparison.csv.

VARIANCE ESTIMATOR -- the point of this module. The forecast horizon is five
days while forecasts are produced daily, so consecutive loss differentials
share four of their five underlying returns and are mechanically
autocorrelated. Assuming i.i.d. differentials would understate the standard
error and overstate significance -- the same overlapping-horizon artifact that
already invalidated a full-sample Christoffersen reading earlier in this
project (docs/risks_and_roadmap.md). Every headline number here therefore uses
a **Newey-West HAC estimator with a Bartlett kernel at 4 lags** (h-1 for h=5,
the standard choice for an h-step overlapping horizon). The Bartlett kernel
guarantees a non-negative variance estimate.

Lag sensitivity q = 0..4 is reported alongside, with q=0 being exactly the
naive i.i.d. case, so the size of the correction is visible rather than
asserted.

Also reported: the Harvey-Leybourne-Newbold (1997) small-sample correction,
compared against t(n-1) rather than N(0,1). At n=874 it is a negligible
adjustment, included for completeness rather than because it changes anything.

Sign convention: d_t = L(model_a) - L(model_b), so a NEGATIVE mean differential
means model_a has the lower loss, i.e. model_a forecasts better.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
FORECASTS_TABLE_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"
DM_TABLE_PATH = REPO_ROOT / "results" / "tables" / "diebold_mariano.csv"

HORIZON = 5                 # forecast horizon in trading days
HAC_LAGS = HORIZON - 1      # 4 -- the headline estimator
LAG_SENSITIVITY = (0, 1, 2, 3, 4)

PAIRS = [
    ("GJR-GARCH", "RandomForest"),
    ("GJR-GARCH", "Hybrid (symmetric)"),
]


def loss_absolute(actual, forecast):
    """Per-observation loss underlying MAE."""
    return np.abs(actual - forecast)


def loss_squared(actual, forecast):
    """Per-observation loss underlying RMSE (squared error, not rooted --
    rooting is a monotone transform applied after averaging and would not be a
    per-observation loss)."""
    return (actual - forecast) ** 2


def loss_qlike(actual, forecast):
    """Per-observation QLIKE, on VARIANCE -- both series squared first, exactly
    as evaluation.qlike does. Squaring is the definition, not a stylistic
    choice (CLAUDE.md)."""
    ratio = (actual ** 2) / (forecast ** 2)
    return ratio - np.log(ratio) - 1


LOSSES = {"MAE": loss_absolute, "RMSE": loss_squared, "QLIKE": loss_qlike}


def newey_west_variance(d, lags):
    """Long-run variance of the mean of `d`, Newey-West with a Bartlett kernel.

    Returns the variance OF THE MEAN (i.e. already divided by n), which is what
    the DM statistic's denominator needs. lags=0 reduces to the i.i.d. case.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    dev = d - d.mean()
    gamma0 = float(np.dot(dev, dev) / n)
    total = gamma0
    for j in range(1, lags + 1):
        gamma_j = float(np.dot(dev[j:], dev[:-j]) / n)
        weight = 1.0 - j / (lags + 1.0)      # Bartlett
        total += 2.0 * weight * gamma_j
    return total / n


def diebold_mariano(loss_a, loss_b, lags=HAC_LAGS, horizon=HORIZON):
    """DM test on the loss differential d = loss_a - loss_b.

    Negative statistic => model_a's loss is lower => model_a forecasts better.
    Returns the HAC-based statistic/p-value plus the HLN-corrected pair.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    dbar = float(d.mean())
    var_dbar = newey_west_variance(d, lags)

    if var_dbar <= 0:
        raise ValueError("non-positive HAC variance -- cannot form the DM statistic")

    dm = dbar / np.sqrt(var_dbar)
    p_dm = float(2 * (1 - norm.cdf(abs(dm))))

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    adj = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_hln = dm * adj
    p_hln = float(2 * (1 - student_t.cdf(abs(dm_hln), df=n - 1)))

    return {
        "n": n,
        "mean_loss_diff": dbar,
        "hac_lags": lags,
        "se_mean_diff": float(np.sqrt(var_dbar)),
        "dm_stat": float(dm),
        "dm_pvalue": p_dm,
        "dm_stat_hln": float(dm_hln),
        "dm_pvalue_hln": p_hln,
    }


def run_all(forecasts, pairs=PAIRS, losses=LOSSES):
    clean = forecasts.dropna(subset=["actual"]).reset_index(drop=True)
    actual = clean["actual"].to_numpy()

    rows = []
    for model_a, model_b in pairs:
        fa, fb = clean[model_a].to_numpy(), clean[model_b].to_numpy()
        for loss_name, loss_fn in losses.items():
            la, lb = loss_fn(actual, fa), loss_fn(actual, fb)
            base = diebold_mariano(la, lb)
            better = model_a if base["mean_loss_diff"] < 0 else model_b
            row = {"model_a": model_a, "model_b": model_b, "loss": loss_name,
                   "mean_loss_a": float(la.mean()), "mean_loss_b": float(lb.mean()),
                   "lower_loss": better}
            row.update(base)
            row["significant_5pct"] = bool(base["dm_pvalue"] < 0.05)
            # lag sensitivity, q=0 being the naive i.i.d. assumption
            for q in LAG_SENSITIVITY:
                alt = diebold_mariano(la, lb, lags=q)
                row[f"dm_stat_lag{q}"] = alt["dm_stat"]
                row[f"dm_pvalue_lag{q}"] = alt["dm_pvalue"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    forecasts = pd.read_csv(FORECASTS_TABLE_PATH, parse_dates=["Date"])
    table = run_all(forecasts)

    print(f"Diebold-Mariano, Newey-West HAC (Bartlett) at {HAC_LAGS} lags, "
          f"h={HORIZON}, n={int(table['n'].iloc[0])}")
    print("d = loss(model_a) - loss(model_b); negative => model_a better\n")
    for (a, b), grp in table.groupby(["model_a", "model_b"], sort=False):
        print(f"{a}  vs  {b}")
        for _, r in grp.iterrows():
            sig = "significant" if r.significant_5pct else "NOT significant"
            print(f"  {r.loss:6s} mean d={r.mean_loss_diff:+.6e}  DM={r.dm_stat:+7.4f}  "
                  f"p={r.dm_pvalue:.4f}  -> lower loss: {r.lower_loss:20s} ({sig} at 5%)")
        print()

    print("Lag sensitivity (q=0 is the naive i.i.d. assumption):")
    for _, r in table.iterrows():
        cells = "  ".join(f"q={q}: {r[f'dm_pvalue_lag{q}']:.4f}" for q in LAG_SENSITIVITY)
        print(f"  {r.model_a} vs {r.model_b} [{r.loss}]  {cells}")

    DM_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(DM_TABLE_PATH, index=False)
    print(f"\nSaved: {DM_TABLE_PATH}")


if __name__ == "__main__":
    sys.exit(main())

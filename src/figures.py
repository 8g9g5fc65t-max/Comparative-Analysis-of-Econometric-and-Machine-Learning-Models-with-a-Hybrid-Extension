"""Thesis figures, saved as vector PDF to results/figures/.

- returns_timeseries.pdf: daily log returns, full sample, COVID-crash
  point annotated.
- forecast_error_test.pdf: test-period forecast error (actual - forecast)
  for GJR-GARCH and XGBoost (results/tables/forecasts_all_models.csv --
  no walk-forward re-run needed). Positive error means actual > forecast,
  i.e. the model UNDERpredicted realised volatility (forecast too low) --
  XGBoost's single largest underprediction is annotated.
- xgboost_tree_example.pdf: one tree from an XGBoost ensemble, purely
  illustrative (see plot_xgboost_tree).

All styled plain: white background, no gridlines, minimal decoration. The
first two are sized for a single-column thesis page (13cm x 7cm); the tree
figure needs a taller canvas (see TREE_FIGSIZE).

Rendered with matplotlib rather than xgboost.plot_tree/to_graphviz on
purpose: those need the graphviz Python package AND a `dot` binary on PATH,
a system-level dependency a reader reproducing this repo would have to
install separately. Drawing from trees_to_dataframe() keeps the figure
reproducible with the pinned requirements alone, and matching the other
figures' fonts and palette.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (build_features, load_processed, TrainTestSplit,
                      RQ1_ML_FEATURES, TARGET_COL)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "gspc_processed.csv"
FORECASTS_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

CM_TO_INCH = 1 / 2.54
FIGSIZE = (13 * CM_TO_INCH, 7 * CM_TO_INCH)
# The tree is 29 nodes / 15 leaves: it does not fit the 13x7cm format at a
# legible font size. Laid out left-to-right instead of top-down (depth runs
# across, leaves stack down), which gives each node a full-width single-line
# label. 16cm is the text width of the thesis's A4/2.5cm-margin geometry, so
# this still needs no shrinking on the page.
TREE_FIGSIZE = (16 * CM_TO_INCH, 12 * CM_TO_INCH)
TREE_INDEX = 0  # which tree of the 200 to draw

COVID_CRASH_DATE = "2020-03-16"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.size": 8,
})


def plot_returns_timeseries(df, path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["Date"], df["log_return"], color="#1f77b4", linewidth=0.5)

    crash = df.loc[df["Date"] == COVID_CRASH_DATE].iloc[0]
    ax.plot(crash["Date"], crash["log_return"], "o", color="#d62728",
             markersize=4, zorder=5)
    # Text sits up and to the left of the point (inside the plot area, well
    # clear of the x-axis tick labels below) with the arrow pointing back
    # down to the actual data point.
    ax.annotate(
        f"2020-03-16 ({crash['log_return']:.1%})",
        xy=(crash["Date"], crash["log_return"]),
        xytext=(-90, 35), textcoords="offset points",
        fontsize=7, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily log return")
    ax.set_xlim(df["Date"].min(), df["Date"].max())
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def plot_forecast_error(df, path):
    """df: forecasts_all_models.csv, already restricted to rows with a
    valid actual (tail NaN rows dropped by the caller). Error = actual -
    forecast: positive means the model underpredicted realised volatility
    (forecast too low, the dangerous direction for risk management);
    negative means it overpredicted."""
    error_gjr = df["actual"] - df["GJR-GARCH"]
    error_xgb = df["actual"] - df["XGBoost"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(0, color="gray", linewidth=0.8, zorder=1)
    ax.plot(df["Date"], error_gjr, color="#2ca02c", linewidth=0.8,
             label="GJR-GARCH error")
    ax.plot(df["Date"], error_xgb, color="#ff7f0e", linewidth=0.8,
             label="XGBoost error")

    worst_idx = error_xgb.idxmax()  # largest underprediction: max(actual - forecast)
    worst_date = df["Date"].loc[worst_idx]
    worst_value = error_xgb.loc[worst_idx]
    ax.plot(worst_date, worst_value, "o", color="#d62728", markersize=4, zorder=5)
    ax.annotate(
        f"{worst_date.date()} ({worst_value:+.3f})",
        xy=(worst_date, worst_value),
        xytext=(-95, -12), textcoords="offset points",
        fontsize=7, color="#d62728",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast error (actual − forecast)")
    ax.set_xlim(df["Date"].min(), df["Date"].max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def fit_illustrative_xgboost(df):
    """One-off XGBoost fit on the whole 2011-2022 training split, for the tree
    figure only.

    This is NOT part of the walk-forward evaluation and is not one of its 184
    weekly refits: it is a single static fit whose only purpose is to show what
    one tree in the ensemble looks like. Hyperparameters and seed are copied
    from models/ml_models.XGBoostModel so the illustration matches the model
    actually evaluated. Returns (fitted model, training frame).
    """
    train, _ = TrainTestSplit().split(df)
    train = train.dropna(subset=list(RQ1_ML_FEATURES) + [TARGET_COL])
    model = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
    )
    model.fit(train[RQ1_ML_FEATURES], train[TARGET_COL])
    return model, train


def _tree_layout(nodes, root):
    """Assign (depth, y) to every node. Leaves take consecutive y slots in
    traversal order so no edges cross; an internal node sits at the midpoint
    of its two children."""
    pos, counter = {}, [0]

    def walk(nid, depth):
        left, right = nodes[nid]["yes"], nodes[nid]["no"]
        if left is None:                       # leaf
            y = counter[0]
            counter[0] += 1
        else:
            y = (walk(left, depth + 1) + walk(right, depth + 1)) / 2
        pos[nid] = (depth, y)
        return y

    walk(root, 0)
    return pos


def plot_xgboost_tree(model, path, tree_index=TREE_INDEX):
    """Draw one tree of the ensemble as a left-to-right diagram.

    Internal nodes show the split as `feature < threshold` (XGBoost's "yes"
    branch is the condition being true); leaves show the additive contribution
    that tree makes to the prediction. Returns (n_nodes, n_leaves, depth).
    """
    frame = model.get_booster().trees_to_dataframe()
    frame = frame[frame["Tree"] == tree_index]

    nodes = {}
    for _, r in frame.iterrows():
        is_leaf = r["Feature"] == "Leaf"
        nodes[r["ID"]] = {
            "leaf": is_leaf,
            "feature": None if is_leaf else r["Feature"],
            "split": None if is_leaf else float(r["Split"]),
            "value": float(r["Gain"]) if is_leaf else None,  # Gain holds the leaf value
            "yes": None if is_leaf else r["Yes"],
            "no": None if is_leaf else r["No"],
        }
    root = f"{tree_index}-0"
    pos = _tree_layout(nodes, root)
    max_depth = max(d for d, _ in pos.values())
    n_leaves = sum(1 for n in nodes.values() if n["leaf"])

    fig, ax = plt.subplots(figsize=TREE_FIGSIZE)

    # Edges first, so the node boxes sit on top of them. _tree_layout walks the
    # "yes" child first, so it always lands on the upper branch -- stated once
    # in the subtitle rather than labelled on all 28 edges, which at this size
    # collided with the node boxes.
    for nid, node in nodes.items():
        if node["leaf"]:
            continue
        x0, y0 = pos[nid]
        for child in (node["yes"], node["no"]):
            x1, y1 = pos[child]
            ax.annotate("", xy=(x1 - 0.02, y1), xytext=(x0 + 0.02, y0),
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6,
                                        connectionstyle="angle,angleA=0,angleB=90,rad=4"))

    for nid, node in nodes.items():
        x, y = pos[nid]
        if node["leaf"]:
            txt, face, edge = f"{node['value']:+.4f}", "#eaf3ea", "#2ca02c"
        else:
            txt = f"{node['feature']} < {node['split']:.4f}"
            face, edge = "#eaf0f7", "#1f77b4"
        ax.text(x, y, txt, fontsize=6, ha="center", va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=edge,
                          linewidth=0.5))

    ax.set_xlim(-0.45, max_depth + 0.55)
    ax.set_ylim(n_leaves - 0.4, -1.1)          # inverted: first leaf at the top
    ax.axis("off")
    ax.set_title(f"XGBoost ensemble, tree {tree_index} of 200 "
                 f"({n_leaves} leaves, depth {max_depth})", fontsize=7.5, pad=10)
    ax.text(0.5, 1.005, "upper branch = split condition true; leaf values are "
                        "that tree's additive contribution to the forecast",
            transform=ax.transAxes, fontsize=5.5, color="#555555",
            ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return len(nodes), n_leaves, max_depth


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])
    path1 = FIGURES_DIR / "returns_timeseries.pdf"
    plot_returns_timeseries(processed, path1)
    print(f"Saved: {path1}")

    forecasts = pd.read_csv(FORECASTS_PATH, parse_dates=["Date"])
    forecasts = forecasts.dropna(subset=["actual"]).reset_index(drop=True)
    path2 = FIGURES_DIR / "forecast_error_test.pdf"
    plot_forecast_error(forecasts, path2)
    print(f"Saved: {path2}")

    model, train = fit_illustrative_xgboost(build_features(load_processed()))
    path3 = FIGURES_DIR / "xgboost_tree_example.pdf"
    n_nodes, n_leaves, depth = plot_xgboost_tree(model, path3)
    print(f"Saved: {path3}")
    print(f"  illustrative fit: {len(train)} training rows, "
          f"{train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"  tree {TREE_INDEX}: {n_nodes} nodes, {n_leaves} leaves, depth {depth}")


if __name__ == "__main__":
    sys.exit(main())

"""Thesis figures, saved as vector PDF to results/figures/.

- returns_timeseries.pdf: daily log returns, full sample, COVID-crash
  point annotated.
- forecast_vs_actual_test.pdf: test-period target_rv_5d vs. GJR-GARCH and
  XGBoost forecasts (results/tables/forecasts_all_models.csv -- no
  walk-forward re-run needed).

Both sized for a single-column thesis page (~13cm wide) and styled plain:
white background, no gridlines, minimal decoration.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "gspc_processed.csv"
FORECASTS_PATH = REPO_ROOT / "results" / "tables" / "forecasts_all_models.csv"
FIGURES_DIR = REPO_ROOT / "results" / "figures"

CM_TO_INCH = 1 / 2.54
FIGSIZE = (13 * CM_TO_INCH, 7 * CM_TO_INCH)

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


def plot_forecast_vs_actual(df, path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(df["Date"], df["actual"], color="black", linewidth=0.9,
             label="Actual (target_rv_5d)")
    ax.plot(df["Date"], df["GJR-GARCH"], color="#2ca02c", linewidth=0.8,
             label="GJR-GARCH forecast")
    ax.plot(df["Date"], df["XGBoost"], color="#ff7f0e", linewidth=0.8,
             label="XGBoost forecast")

    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised volatility")
    ax.set_xlim(df["Date"].min(), df["Date"].max())
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30, ha="right")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])
    path1 = FIGURES_DIR / "returns_timeseries.pdf"
    plot_returns_timeseries(processed, path1)
    print(f"Saved: {path1}")

    forecasts = pd.read_csv(FORECASTS_PATH, parse_dates=["Date"])
    path2 = FIGURES_DIR / "forecast_vs_actual_test.pdf"
    plot_forecast_vs_actual(forecasts, path2)
    print(f"Saved: {path2}")


if __name__ == "__main__":
    sys.exit(main())

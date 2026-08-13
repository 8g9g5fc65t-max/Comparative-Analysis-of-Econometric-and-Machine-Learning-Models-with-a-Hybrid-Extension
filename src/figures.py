"""Thesis figures, saved as vector PDF to results/figures/.

- returns_timeseries.pdf: daily log returns, full sample, COVID-crash
  point annotated.
- forecast_error_test.pdf: test-period forecast error (actual - forecast)
  for GJR-GARCH and XGBoost (results/tables/forecasts_all_models.csv --
  no walk-forward re-run needed). Positive error means actual > forecast,
  i.e. the model UNDERpredicted realised volatility (forecast too low) --
  XGBoost's single largest underprediction is annotated.

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


if __name__ == "__main__":
    sys.exit(main())

"""Descriptive statistics for daily log returns, full cleaned sample.

Reads data/processed/gspc_processed.csv (log_return column) and computes
mean, std (daily + annualised), skewness, excess kurtosis, min, max, and a
Jarque-Bera normality test. Saves the raw numbers to
results/tables/returns_descriptive_stats.csv and prints a booktabs LaTeX
table (caption below the tabular, per thesis convention) for the Results
section.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import TRADING_DAYS_PER_YEAR

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "gspc_processed.csv"
STATS_TABLE_PATH = REPO_ROOT / "results" / "tables" / "returns_descriptive_stats.csv"


def compute_stats(log_returns):
    """log_returns: array-like, NaN-free. Returns a dict of the stats."""
    r = np.asarray(log_returns, dtype=float)
    mean_daily = float(np.mean(r))
    std_daily = float(np.std(r, ddof=1))
    jb_stat, jb_pvalue = jarque_bera(r)
    return {
        "n": len(r),
        "mean_daily": mean_daily,
        "std_daily": std_daily,
        "mean_annualized": mean_daily * TRADING_DAYS_PER_YEAR,
        "std_annualized": std_daily * np.sqrt(TRADING_DAYS_PER_YEAR),
        "skewness": float(skew(r, bias=True)),
        "excess_kurtosis": float(kurtosis(r, fisher=True, bias=True)),
        "min": float(np.min(r)),
        "max": float(np.max(r)),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
    }


def to_latex(stats):
    def fmt(x, nd=4):
        return f"{x:.{nd}f}"

    def fmt_pvalue(p, nd=4):
        # A statistic this extreme drives the p-value below float64's
        # representable range (genuine underflow, not a bug) -- show a
        # bound instead of a misleading literal "0".
        threshold = 10 ** (-nd)
        return f"< {threshold:g}" if p < threshold else fmt(p, nd)

    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\begin{tabular}{lc}\n"
        "\\toprule\n"
        "Statistic & Value \\\\\n"
        "\\midrule\n"
        f"Mean (daily) & {fmt(stats['mean_daily'], 6)} \\\\\n"
        f"Mean (annualised) & {fmt(stats['mean_annualized'])} \\\\\n"
        f"Std.\\ dev.\\ (daily) & {fmt(stats['std_daily'], 6)} \\\\\n"
        f"Std.\\ dev.\\ (annualised) & {fmt(stats['std_annualized'])} \\\\\n"
        f"Skewness & {fmt(stats['skewness'])} \\\\\n"
        f"Excess kurtosis & {fmt(stats['excess_kurtosis'])} \\\\\n"
        f"Minimum & {fmt(stats['min'])} \\\\\n"
        f"Maximum & {fmt(stats['max'])} \\\\\n"
        f"Jarque--Bera statistic & {fmt(stats['jarque_bera_stat'], 2)} \\\\\n"
        f"Jarque--Bera $p$-value & {fmt_pvalue(stats['jarque_bera_pvalue'])} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Descriptive statistics for daily log returns, "
        "S\\&P 500 index (ticker \\texttt{\\textasciicircum GSPC}), "
        f"full cleaned sample ($n={stats['n']}$).}}\n"
        "\\label{tab:returns_descriptive_stats}\n"
        "\\end{table}"
    )


def main():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["Date"])
    r = df["log_return"].dropna()

    stats = compute_stats(r)
    print("Descriptive statistics (log_return, full cleaned sample):")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    STATS_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(STATS_TABLE_PATH, index=False)
    print(f"\nSaved: {STATS_TABLE_PATH}")

    print("\nLaTeX table:\n")
    print(to_latex(stats))


if __name__ == "__main__":
    sys.exit(main())

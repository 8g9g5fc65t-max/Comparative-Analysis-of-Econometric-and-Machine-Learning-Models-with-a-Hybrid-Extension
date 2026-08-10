"""
Data pipeline — Stage 1 of the TFM (see CLAUDE.md).

Downloads daily OHLCV for the S&P 500 index (^GSPC) via yfinance, saves the
raw pull untouched, then produces a cleaned series with computed log returns.

Usage:
    python src/data_pipeline.py

Outputs:
    data/raw/gspc_raw.csv                  - untouched pull from yfinance
    data/processed/gspc_processed.csv      - cleaned series + log returns + flags
    docs/data_provenance.md                - source/date/period/cleaning note

Notes:
    - ^GSPC is a price index, not a total-return index: it is NOT
      dividend-adjusted. This is standard practice for volatility work but
      is a real caveat for anything return-level (e.g. buy-and-hold P&L).
      See docs/data_provenance.md for the full caveat.
    - No data leakage: every quantity computed here (log returns, flags) is
      constructible from information available strictly at that date.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TICKER = "^GSPC"
START_DATE = "2011-01-01"
END_DATE = None  # None => through the most recent available session
OUTLIER_THRESHOLD = 0.10  # |log return| beyond this is flagged, not dropped

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DOCS_DIR = REPO_ROOT / "docs"

RAW_PATH = RAW_DIR / "gspc_raw.csv"
PROCESSED_PATH = PROCESSED_DIR / "gspc_processed.csv"
PROVENANCE_PATH = DOCS_DIR / "data_provenance.md"

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_raw(ticker=TICKER, start=START_DATE, end=END_DATE):
    """Pull daily OHLCV from yfinance and return it exactly as received
    (aside from flattening the column index yfinance now returns)."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,   # keep Close and Adj Close as separate columns
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(
            f"yfinance returned no data for {ticker}. Check ticker/network/"
            f"rate limiting before proceeding."
        )

    # yfinance >=0.2 returns MultiIndex columns (Field, Ticker) even for a
    # single symbol; flatten to plain field names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    df = df.reset_index()[["Date"] + REQUIRED_COLS]
    return df


def save_raw(df, path=RAW_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_and_flag(df_raw, ticker=TICKER):
    """Clean the raw pull and flag (never silently drop) suspicious rows.

    Returns (df_clean, report) where report is a dict of cleaning stats for
    the provenance note.
    """
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    report = {}

    # 1. Duplicates on Date -> drop, keep first, but count them.
    n_dupes = int(df["Date"].duplicated().sum())
    if n_dupes:
        df = df.drop_duplicates(subset="Date", keep="first")
    report["duplicate_rows_dropped"] = n_dupes

    # 2. Sort ascending, reset index.
    df = df.sort_values("Date").reset_index(drop=True)

    # 3. Non-trading days: yfinance should only return trading sessions, but
    #    verify no Saturday/Sunday rows slipped in.
    weekday = df["Date"].dt.weekday
    n_weekend = int((weekday >= 5).sum())
    if n_weekend:
        df = df[weekday < 5].reset_index(drop=True)
    report["weekend_rows_dropped"] = n_weekend

    # 4. Gap check against the NYSE trading calendar (flag only, no fill).
    if mcal is not None:
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(
            start_date=df["Date"].min(), end_date=df["Date"].max()
        )
        expected = pd.DatetimeIndex(sched.index.date).normalize()
        actual = pd.DatetimeIndex(df["Date"].dt.normalize())
        missing_sessions = expected.difference(actual)
        extra_sessions = actual.difference(expected)
        report["missing_trading_sessions"] = [
            d.strftime("%Y-%m-%d") for d in missing_sessions
        ]
        report["unexpected_non_trading_dates"] = [
            d.strftime("%Y-%m-%d") for d in extra_sessions
        ]
    else:
        report["missing_trading_sessions"] = None
        report["unexpected_non_trading_dates"] = None

    # 5. Missing values in OHLCV -> flag, don't drop.
    df["flag_missing_value"] = df[REQUIRED_COLS].isna().any(axis=1)
    report["rows_with_missing_values"] = int(df["flag_missing_value"].sum())

    # 6. Zero-volume days -> flag, don't drop.
    df["flag_zero_volume"] = df["Volume"] == 0
    report["zero_volume_rows"] = int(df["flag_zero_volume"].sum())

    # 7. Log returns from Adjusted Close: r_t = ln(P_t / P_{t-1}).
    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

    # 8. Outlier flag on |log_return| (single-day, informational only).
    df["flag_outlier_return"] = df["log_return"].abs() > OUTLIER_THRESHOLD
    report["outlier_return_rows"] = int(df["flag_outlier_return"].sum())
    report["outlier_threshold"] = OUTLIER_THRESHOLD

    report["n_rows_clean"] = len(df)
    report["date_min"] = df["Date"].min().strftime("%Y-%m-%d")
    report["date_max"] = df["Date"].max().strftime("%Y-%m-%d")

    return df, report


def save_processed(df, path=PROCESSED_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# Provenance note
# --------------------------------------------------------------------------
def write_provenance(report, raw_path, processed_path, path=PROVENANCE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    flagged_dates = df_flagged_dates_note(report)

    text = f"""# Data provenance

- **Source**: Yahoo! Finance, via the `yfinance` Python library
  (v{yf.__version__}).
- **Ticker**: `{TICKER}` (S&P 500 index).
- **Download date**: {download_date}
- **Requested period**: {START_DATE} to {END_DATE or "most recent available session"}
- **Actual period returned**: {report['date_min']} to {report['date_max']}
  ({report['n_rows_clean']} trading-day rows after cleaning)
- **Fields kept**: Date, Open, High, Low, Close, Adj Close, Volume.
- **Raw file** (untouched pull): `{raw_path.relative_to(REPO_ROOT).as_posix()}`
- **Processed file** (cleaned + log returns): `{processed_path.relative_to(REPO_ROOT).as_posix()}`

## Caveat: price index, not total return

`^GSPC` is the S&P 500 **price** index. It is not adjusted for dividends,
unlike a total-return index. This is standard practice for volatility
modelling (dividends are a low-volatility, near-deterministic drag on the
level series and have negligible effect on realised volatility), but it is
technically not what a total-return index would give you, so it should not
be used to draw conclusions about buy-and-hold P&L.

## Cleaning rules applied

1. **Duplicates**: exact duplicate rows on `Date` are dropped, keeping the
   first occurrence. Dropped: {report['duplicate_rows_dropped']}.
2. **Non-trading days**: any Saturday/Sunday rows are dropped (yfinance
   should not return these, so this is a safety check). Dropped:
   {report['weekend_rows_dropped']}.
3. **Trading-calendar gap check** (NYSE calendar via
   `pandas_market_calendars`): sessions present on the NYSE calendar but
   missing from the pull, and rows present in the pull that don't correspond
   to an NYSE session, are both flagged for manual review (not filled or
   dropped).
4. **Missing values**: rows with a NaN in any of Open/High/Low/Close/Adj
   Close/Volume are flagged via `flag_missing_value`, not dropped. Flagged:
   {report['rows_with_missing_values']}.
5. **Zero-volume days**: flagged via `flag_zero_volume`, not dropped.
   Flagged: {report['zero_volume_rows']}.
6. **Outlier single-day returns**: any `|log_return| > {report['outlier_threshold']:.0%}`
   is flagged via `flag_outlier_return`, not dropped — these are usually
   genuine market events (e.g. 2011 downgrade volatility, Dec 2018, Mar 2020
   COVID crash) and are exactly the kind of observation a volatility model
   needs to see. Flagged: {report['outlier_return_rows']}.

{flagged_dates}

## Return definition

Log returns are computed on **Adjusted Close**:

    r_t = ln(Adj Close_t / Adj Close_{{t-1}})

The first row of the processed series has `log_return = NaN` by
construction (no prior observation).

## Reproducing this pull

    python src/data_pipeline.py

Re-running will re-download the current Yahoo Finance history for `{TICKER}`
and overwrite both CSVs; the "download date" above should be updated to
match whenever that happens (this file is regenerated by the script, not
hand-edited).
"""
    path.write_text(text, encoding="utf-8")
    return path


def df_flagged_dates_note(report):
    missing = report.get("missing_trading_sessions")
    extra = report.get("unexpected_non_trading_dates")
    if missing is None:
        return (
            "*(NYSE calendar gap check skipped: `pandas_market_calendars` "
            "not installed.)*"
        )
    lines = ["## Flagged calendar gaps (NYSE sessions vs. pulled dates)", ""]
    if missing:
        lines.append(f"- Missing sessions (on NYSE calendar, absent from pull): "
                      f"{', '.join(missing)}")
    else:
        lines.append("- Missing sessions: none.")
    if extra:
        lines.append(f"- Unexpected dates (in pull, not an NYSE session): "
                      f"{', '.join(extra)}")
    else:
        lines.append("- Unexpected non-session dates: none.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"Downloading {TICKER} from {START_DATE} to "
          f"{END_DATE or 'latest'} via yfinance...")
    df_raw = download_raw()
    raw_path = save_raw(df_raw)
    print(f"Raw pull saved: {raw_path} ({len(df_raw)} rows, "
          f"{df_raw['Date'].min().date()} to {df_raw['Date'].max().date()})")

    df_clean, report = clean_and_flag(df_raw)
    processed_path = save_processed(df_clean)
    print(f"Processed data saved: {processed_path} ({report['n_rows_clean']} rows)")

    prov_path = write_provenance(report, raw_path, processed_path)
    print(f"Provenance note saved: {prov_path}")

    print("\nSummary:")
    for k, v in report.items():
        if isinstance(v, list):
            v = f"{len(v)} item(s)" if v else "none"
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())

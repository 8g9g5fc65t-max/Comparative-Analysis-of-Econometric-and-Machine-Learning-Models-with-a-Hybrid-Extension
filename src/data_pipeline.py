"""Download, clean and flag S&P 500 (^GSPC) OHLCV data. See docs/data_provenance.md."""
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

TICKER = "^GSPC"
START_DATE = "2011-01-01"
# Fixed, not "today": keeps the dataset (and every downstream result) identical
# on every re-run instead of silently growing each time the pipeline is run.
END_DATE = "2026-07-07"
OUTLIER_THRESHOLD = 0.10

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = REPO_ROOT / "data" / "raw" / "gspc_raw.csv"
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "gspc_processed.csv"
PROVENANCE_PATH = REPO_ROOT / "docs" / "data_provenance.md"

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_raw(ticker=TICKER, start=START_DATE, end=END_DATE):
    # yfinance's `end` is exclusive, so bump by a day to make END_DATE itself
    # the last possible session included.
    yf_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None
    df = yf.download(ticker, start=start, end=yf_end, auto_adjust=False,
                      actions=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df.reset_index()[["Date"] + REQUIRED_COLS]


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def clean_and_flag(df_raw):
    df = df_raw.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    report = {}

    n_dupes = int(df["Date"].duplicated().sum())
    df = df.drop_duplicates(subset="Date", keep="first")
    report["duplicate_rows_dropped"] = n_dupes

    df = df.sort_values("Date").reset_index(drop=True)

    weekday = df["Date"].dt.weekday
    n_weekend = int((weekday >= 5).sum())
    df = df[weekday < 5].reset_index(drop=True)
    report["weekend_rows_dropped"] = n_weekend

    if mcal is not None:
        sched = mcal.get_calendar("NYSE").schedule(
            start_date=df["Date"].min(), end_date=df["Date"].max())
        expected = pd.DatetimeIndex(sched.index.date).normalize()
        actual = pd.DatetimeIndex(df["Date"].dt.normalize())
        report["missing_trading_sessions"] = [
            d.strftime("%Y-%m-%d") for d in expected.difference(actual)]
        report["unexpected_non_trading_dates"] = [
            d.strftime("%Y-%m-%d") for d in actual.difference(expected)]
    else:
        report["missing_trading_sessions"] = None
        report["unexpected_non_trading_dates"] = None

    df["flag_missing_value"] = df[REQUIRED_COLS].isna().any(axis=1)
    report["rows_with_missing_values"] = int(df["flag_missing_value"].sum())

    df["flag_zero_volume"] = df["Volume"] == 0
    report["zero_volume_rows"] = int(df["flag_zero_volume"].sum())

    df["log_return"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))

    df["flag_outlier_return"] = df["log_return"].abs() > OUTLIER_THRESHOLD
    report["outlier_return_rows"] = int(df["flag_outlier_return"].sum())
    report["outlier_threshold"] = OUTLIER_THRESHOLD

    report["n_rows_clean"] = len(df)
    report["date_min"] = df["Date"].min().strftime("%Y-%m-%d")
    report["date_max"] = df["Date"].max().strftime("%Y-%m-%d")

    return df, report


def flagged_dates_section(report):
    missing = report.get("missing_trading_sessions")
    extra = report.get("unexpected_non_trading_dates")
    if missing is None:
        return ("*(NYSE calendar gap check skipped: `pandas_market_calendars` "
                 "not installed.)*")
    lines = ["## Flagged calendar gaps (NYSE sessions vs. pulled dates)", ""]
    lines.append("- Missing sessions (on NYSE calendar, absent from pull): "
                  + (", ".join(missing) if missing else "none."))
    lines.append("- Unexpected dates (in pull, not an NYSE session): "
                  + (", ".join(extra) if extra else "none."))
    return "\n".join(lines)


def write_provenance(report, raw_path, processed_path, path=PROVENANCE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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

{flagged_dates_section(report)}

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


def main():
    print(f"Downloading {TICKER} from {START_DATE} to {END_DATE or 'latest'}...")
    df_raw = download_raw()
    raw_path = save_csv(df_raw, RAW_PATH)
    print(f"Raw pull saved: {raw_path} ({len(df_raw)} rows)")

    df_clean, report = clean_and_flag(df_raw)
    processed_path = save_csv(df_clean, PROCESSED_PATH)
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

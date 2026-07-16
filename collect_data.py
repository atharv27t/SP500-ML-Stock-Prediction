"""
S&P 500 Stock Data Collection
==============================
Downloads 6 years of daily OHLCV data for 10 S&P 500 stocks using yfinance,
adds sector labels, computes basic technical indicators, and saves the result
as ``sp500_stocks.csv``.

Requirements:
    pip install yfinance pandas

Usage:
    python collect_data.py
"""

from __future__ import annotations

import yfinance as yf
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = PROJECT_DIR / "sp500_stocks.csv"

STOCKS: dict[str, str] = {
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "GOOGL": "Communication Services",
    "AMZN":  "Consumer Discretionary",
    "NVDA":  "Semiconductors",
    "JPM":   "Financials",
    "JNJ":   "Healthcare",
    "XOM":   "Energy",
    "PG":    "Consumer Staples",
    "HD":    "Consumer Discretionary",
}

START = "2020-01-01"
END = "2026-06-04"


def main() -> None:
    frames = []
    for ticker, sector in STOCKS.items():
        print(f"Downloading {ticker} ({sector}) ...")
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            print(f"  WARNING: no data for {ticker}, skipping.")
            continue

        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["ticker"] = ticker
        df["sector"] = sector
        frames.append(df)
        print(f"  {len(df)} rows")

    if not frames:
        print("No data downloaded. Exiting.")
        return

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["ticker", "Date"]).reset_index(drop=True)

    full.rename(columns={"Date": "date", "Open": "open", "High": "high",
                         "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)

    full.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(full)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

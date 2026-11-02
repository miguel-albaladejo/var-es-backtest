"""
Download, cleaning and transformation of price data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log-returns r_t = ln(P_t / P_{t-1}).
    """
    log_prices = np.log(prices)
    return log_prices.diff().iloc[1:]


def to_losses(returns: pd.Series) -> pd.Series:
    """
    Compute losses: L_t = -r_t
    """
    return -returns


def clean_prices(prices: pd.Series) -> pd.Series:
    """
    Return a clean copy of a raw price series.
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.sort_index()
    prices = prices.dropna()
    prices = prices[prices > 0]

    return prices.astype(float)


def load_prices(
    ticker: str,
    start: str = "2000-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    Load daily adjusted closing prices from Yahoo Finance, with a local cache.
    """
    cache_file = DATA_DIR / f"{ticker.replace('^', '')}.csv"

    if use_cache and cache_file.exists():
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker!r}. Check the ticker.")

        prices = df["Close"]
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        DATA_DIR.mkdir(exist_ok=True)
        prices.to_csv(cache_file)

    prices.name = ticker
    prices.index.name = "Date"
    return clean_prices(prices)

"""
Download, cleaning and transformation of price data.

Conventions
-----------
Log-returns: r_t = ln(P_t / P_{t-1}). They are additive over time and
approximate simple returns R_t to first order.
Losses: L_t = -r_t.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Folder where downloaded prices are cached: <repo>/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log-returns r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.Series
        Clean price series indexed by date (no NaN, strictly positive).

    Returns
    -------
    pd.Series
        Log-returns starting at the second date.
    """
    log_prices = np.log(prices)
    return log_prices.diff().iloc[1:]


def to_losses(returns: pd.Series) -> pd.Series:
    """
    Compute losses as L_t = -r_t

    Parameters
    ----------
    returns : pd.Series
        Daily log-returns.

    Returns
    -------
    pd.Series
        Daily losses.
    """
    return -returns


def clean_prices(prices: pd.Series) -> pd.Series:
    """
    Return a clean copy of a raw price series.

    Parameters
    ----------
    prices : pd.Series
        Raw prices indexed by date.

    Returns
    -------
    pd.Series
        Prices sorted by date, without duplicated dates, NaN or
        non-positive values, as floats and with the same name as the input.
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

    The first call downloads the prices and saves them to data/<ticker>.csv.
    Later calls read that file, so results are reproducible and do not
    depend on Yahoo Finance being available.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, e.g. "^GSPC" (S&P 500) or "^IBEX" (IBEX 35).
    start, end : str
        Dates in "YYYY-MM-DD" format. ``end=None`` means up to today.
    use_cache : bool
        If True, read the local CSV when it exists instead of downloading.

    Returns
    -------
    pd.Series
        Clean adjusted closing prices, named after the ticker.
    """
    cache_file = DATA_DIR / f"{ticker.replace('^', '')}.csv"

    if use_cache and cache_file.exists():
        # Read the cached prices: first column = dates, second = prices.
        prices = pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        # Download. auto_adjust=True adjusts prices for splits and dividends.
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

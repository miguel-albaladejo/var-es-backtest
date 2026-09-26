"""
Download, cleaning and transformation of price data.

Conventions
-----------
Log-returns: r_t = ln(P_t / P_{t-1}). They are additive over time and
approximate simple returns R_t to first order.
Losses: L_t = -r_t.
"""

import numpy as np
import pandas as pd


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

"""
Download, clean and transformation of data prices.
Let us define the logaritmic returns as r_t = log(Pt/Pt-1). It is
well-known that this log-returns are additive and they are a first order
approximation of simple returns.
Let L_t = -r_t be the losses.
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
        Log-returns starting at the second date, same name as `prices`.
    """
    log_prices = np.log(prices)
    return log_prices.diff().iloc[1:]


def to_losses(returns: pd.Series) -> pd.Series:
    """
    Compute losses as L_t = -r_t
    """
    return -returns

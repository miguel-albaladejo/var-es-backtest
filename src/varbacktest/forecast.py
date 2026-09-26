"""
Rolling-window engine that turns a VaR/ES model into daily out-of-sample
forecast.
A Risk measure at time t is a prediction based on what happened in the past.
We use a 'window' to include the information about the past.
"""

from typing import Callable
import numpy as np
import pandas as pd


def rolling_var_es(
    losses: pd.Series,
    model: Callable[[np.ndarray, float], tuple[float, float]],
    window: int = 500,
    alpha: float = 0.99,
) -> pd.DataFrame:
    """
    Rolling one-day-ahead VaR and ES forecasts. The forecast for day t is
    computed with the losses of days t-window, ..., t-1 only.

    Parameters
    ----------
    losses : pd.Series
        Daily losses indexed by date.
    model : callable
        Function (window_losses, alpha) -> (var, es)
    window : int
        Number of past observations used for each forecast.
    alpha : float
        Confidence level (e.g. 0.99).

    Returns
    -------
    pd.DataFrame
        Columns "var" and "es", indexed by forecast date, starting at the
        date in position `window`.
    """
    results = []
    for i in range(window, len(losses)):
        w = losses.iloc[i - window : i].to_numpy()
        results.append(model(w, alpha))
    return pd.DataFrame(results, columns=["var", "es"], index=losses.index[window:])

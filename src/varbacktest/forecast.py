"""Turn a VaR/ES model into daily forecasts with a rolling window."""

from typing import Callable
import numpy as np
import pandas as pd


def rolling_var_es(
    losses: pd.Series,
    model: Callable[[np.ndarray, float], tuple[float, float]],
    window: int = 500,
    alpha: float = 0.99,
) -> pd.DataFrame:
    """One-day-ahead VaR and ES, each one using only the previous `window` days."""
    results = []
    for i in range(window, len(losses)):
        w = losses.iloc[i - window : i].to_numpy()
        results.append(model(w, alpha))
    return pd.DataFrame(results, columns=["var", "es"], index=losses.index[window:])

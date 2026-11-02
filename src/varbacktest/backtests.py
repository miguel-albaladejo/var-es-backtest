"""Statistical backtests of VaR and ES forecasts."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def kupiec_test(exceptions: np.ndarray, alpha: float = 0.99) -> dict:
    """Kupiec (1995): is the share of exceptions compatible with 1 - alpha?"""
    exceptions = np.asarray(exceptions, dtype=int)
    n = len(exceptions)
    x = int(exceptions.sum())
    p = 1 - alpha
    pi_hat = x / n

    log_l0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    log_l1 = xlogy(n - x, 1 - pi_hat) + xlogy(x, pi_hat)

    lr = -2 * (log_l0 - log_l1)
    p_value = stats.chi2.sf(lr, df=1)

    return {
        "n": n,
        "x": x,
        "rate": pi_hat,
        "lr": float(lr),
        "p_value": float(p_value),
    }


def christoffersen_test(exceptions: np.ndarray, alpha: float = 0.99) -> dict:
    """Christoffersen (1998): independence and conditional coverage tests."""
    exceptions = np.asarray(exceptions, dtype=int)

    # Transition counts: yesterday (prev) -> today (curr)
    prev, curr = exceptions[:-1], exceptions[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    log_l1 = (
        xlogy(n00, 1 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11)
        + xlogy(n11, pi11)
    )

    # max(..., 0) avoids tiny negative values caused by floating-point rounding
    lr_ind = max(-2 * (log_l0 - log_l1), 0.0)
    p_ind = stats.chi2.sf(lr_ind, df=1)

    lr_cc = kupiec_test(exceptions, alpha)["lr"] + lr_ind
    p_cc = stats.chi2.sf(lr_cc, df=2)

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi01": pi01,
        "pi11": pi11,
        "lr_ind": float(lr_ind),
        "p_ind": float(p_ind),
        "lr_cc": float(lr_cc),
        "p_cc": float(p_cc),
    }


# Basel plus factor added to the capital multiplier (base 3), by number of
# exceptions in the last 250 days (Basel Committee, 1996).
BASEL_PLUS_FACTOR = {
    0: 0.0,
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.40,
    6: 0.50,
    7: 0.65,
    8: 0.75,
    9: 0.85,
}


def basel_traffic_light(exceptions: pd.Series, window: int = 250) -> pd.DataFrame:
    """Basel zone and capital multiplier of a 99% VaR model, day by day."""
    count = exceptions.astype(int).rolling(window).sum().dropna().astype(int)
    zone = pd.cut(count, bins=[-1, 4, 9, np.inf], labels=["green", "yellow", "red"])
    multiplier = 3 + count.map(lambda c: BASEL_PLUS_FACTOR.get(c, 1.0))
    return pd.DataFrame(
        {"count": count, "zone": zone.astype(str), "multiplier": multiplier}
    )


def acerbi_szekely_test(
    losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float = 0.99
) -> dict:
    """
    Acerbi and Szekely (2014) backtests for Expected Shortfall.

    With p = 1 - alpha and I_t = 1{L_t > VaR_t}, a correct model satisfies
    E[L_t I_t] = p * ES_t. Two statistics (both ~0 under H0, negative when
    the tail risk is underestimated):

        Z1 = 1 - (1/N) * sum_{t: I_t = 1} L_t / ES_t
             (size of the losses on exception days; N = number of exceptions)

        Z2 = 1 - (1/T) * sum_t L_t I_t / (p * ES_t)
             (joint test of the frequency and the size of exceptions)

    The one-sided p-value of Z2 uses a CLT approximation:
        Z2 / (s / sqrt(T)) ~ N(0, 1),  s = std of L_t I_t / (p ES_t)
    so small p-values mean that ES is significantly underestimated.

    Parameters
    ----------
    losses, var, es : np.ndarray
        Realized losses and the VaR and ES forecasts for the same days.
    alpha : float
        Confidence level of VaR and ES (e.g. 0.99).

    Returns
    -------
    dict
        n : number of days
        exceptions : number of VaR exceptions
        mean_loss_to_es : average L_t / ES_t on exception days (1 if correct)
        z1, z2 : Acerbi-Szekely statistics
        p_z2 : one-sided p-value of Z2 (CLT approximation)
    """
    losses = np.asarray(losses, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    p = 1 - alpha
    n = len(losses)

    hits = losses > var
    n_exc = int(hits.sum())

    # Realized loss relative to predicted ES on exception days (0 on other days)
    ratio = np.where(hits, losses / es, 0.0)
    mean_ratio = ratio[hits].mean() if n_exc > 0 else np.nan

    z1 = 1 - mean_ratio
    y = ratio / p
    z2 = 1 - y.mean()

    se = y.std(ddof=1) / np.sqrt(n)
    p_z2 = stats.norm.cdf(z2 / se) if se > 0 else np.nan

    return {
        "n": n,
        "exceptions": n_exc,
        "mean_loss_to_es": float(mean_ratio),
        "z1": float(z1),
        "z2": float(z2),
        "p_z2": float(p_z2),
    }

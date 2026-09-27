"""
Statistical backtests.

Given the exception sequence I_t = 1{L_t > VaR_t}, a correct VaR model at
confidence level alpha must satisfy two properties:

- Unconditional coverage: P(I_t = 1) = 1 - alpha. We test formally with Kupiec (1995)
- Independence: exceptions do not cluster in time. We test formally with Christoffersen (1998)

The Basel traffic-light approach is also included, since it is the rule
used in practice by banking supervisors.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy


def kupiec_test(exceptions: np.ndarray, alpha: float = 0.99) -> dict:
    """
    Kupiec test.
    If the VaR model is correct, each I_t ~ Bernoulli(p) with p = 1 - alpha.
    With x exceptions in n days and observed rate pi_hat = x / n:

        H0: P(I_t = 1) = p      vs      H1: P(I_t = 1) != p

        LR_uc = -2 ln[ p^x (1-p)^(n-x) / (pi_hat^x (1-pi_hat)^(n-x)) ] ~ chi2(1)

    Parameters
    ----------
    exceptions : np.ndarray
        Exception sequence I_t.
    alpha : float
        Confidence level.

    Returns
    -------
    dict
        n : number of days
        x : number of exceptions
        rate : observed exception rate x / n
        lr : likelihood-ratio statistic LR_uc
        p_value : p-value under chi2(1)
    """
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
    """
    Christoffersen test.
    The exception sequence I_t is modelled as a first-order Markov chain.
    Let n_ij be the number of days with I_{t-1} = i and I_t = j, and

        pi_01 = n_01 / (n_00 + n_01)   P(exception today | no exception yesterday)
        pi_11 = n_11 / (n_10 + n_11)   P(exception today | exception yesterday)
        pi    = (n_01 + n_11) / (n_00 + n_01 + n_10 + n_11)

    Independence test (H0: pi_01 = pi_11 = pi):

        LR_ind = -2 ln[ (1-pi)^(n_00+n_10) pi^(n_01+n_11)
                        / ((1-pi_01)^n_00 pi_01^n_01 (1-pi_11)^n_10 pi_11^n_11) ] ~ chi2(1)

    Conditional coverage test (correct rate AND independence):

        LR_cc = LR_uc + LR_ind ~ chi2(2)

    where LR_uc is the Kupiec statistic. Clustered exceptions (pi_11 >> pi_01)
    are rejected by LR_ind even when the total number of exceptions is correct.

    Parameters
    ----------
    exceptions : np.ndarray
        Exception sequence I_t (0/1 or bool), in chronological order.
    alpha : float
        Confidence level of the VaR (e.g. 0.99).

    Returns
    -------
    dict
        n00, n01, n10, n11 : transition counts
        pi01, pi11 : conditional exception probabilities
        lr_ind, p_ind : independence statistic and p-value under chi2(1)
        lr_cc, p_cc : conditional coverage statistic and p-value under chi2(2)
    """
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
    """
    Basel traffic-light zone of a 99% VaR model, day by day.

    On each day, the supervisor counts the VaR exceptions over the last
    `window` trading days (250 = one year, including the current day):

        Green   0-4 exceptions   model accepted, capital multiplier 3
        Yellow  5-9 exceptions   multiplier 3 + plus factor (0.40 to 0.85)
        Red     10+ exceptions   multiplier 4, model under review

    Parameters
    ----------
    exceptions : pd.Series
        Exception sequence I_t (0/1 or bool) indexed by date.
    window : int
        Number of trading days in the regulatory look-back window.

    Returns
    -------
    pd.DataFrame
        Indexed by date (from the first full window on), with columns
        "count" (exceptions in the window), "zone" ("green", "yellow" or
        "red") and "multiplier" (3 + plus factor).
    """
    count = exceptions.astype(int).rolling(window).sum().dropna().astype(int)
    zone = pd.cut(count, bins=[-1, 4, 9, np.inf], labels=["green", "yellow", "red"])
    multiplier = 3 + count.map(lambda c: BASEL_PLUS_FACTOR.get(c, 1.0))
    return pd.DataFrame(
        {"count": count, "zone": zone.astype(str), "multiplier": multiplier}
    )

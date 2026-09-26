"""
Let us consider L the losses of our portfolio and alpha the confidence
level. Then, we define:
    - VaR_alpha(L) as the quantile alpha of L
    - ES_alpha(L) as the mean of the losses beyond VaR_alpha(L)
These risk measures can be calculated either empirically, or theoretically
if a distribution for L is assumed.
"""

import numpy as np
from scipy import stats


def historical_var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    No distribution for losses is assumed here.

    Parameters
    ----------
    losses : np.ndarray
        Vector of losses
    alpha : float
        Confidence level

    Returns
    -------
    (var, es) : tuple[float, float]
        VaR is the empirical quantile
        ES is the mean of losses greater than or equal to VaR
    """
    losses = np.asarray(losses)
    var = np.quantile(losses, alpha)
    es = losses[losses >= var].mean()
    return (float(var), float(es))


def normal_var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    Assume L ~ N(mu, sigma^2). Let z_alpha be such that Phi(z_alpha) = alpha.
    Then, one can prove that
        VaR = mu + sigma*z_alpha
        ES = mu + sigma*phi(z_alpha)/(1-alpha)
    where Phi and phi are the CDF and the pdf, respectively.

    Parameters
    ----------
    losses : np.ndarray
        Array of losses
    alpha : float
        Confidence level

    Returns
    -------
    (var, es) : tuple[float, float]
    """
    mu = losses.mean()
    sd_l = losses.std(ddof=1)
    z = stats.norm.ppf(alpha)
    phi = stats.norm.pdf(z)

    var = mu + sd_l * z
    es = mu + sd_l * phi / (1 - alpha)
    return (float(var), float(es))


def student_t_var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    Assume L = m + s*t_nu. Let q be such that t_nu(q) = alpha.
    Then, one can prove that
        VaR = m + s*q
        ES = m + s*f(q)/(1-alpha)*(nu + q^2)/(nu - 1)
    where t_nu and f are the CDF and the pdf, respectively.

    Parameters
    ----------
    losses : np.ndarray
        Array of losses
    alpha : float
        Confidence level

    Returns
    -------
    (var, es) : tuple[float, float]
    """
    nu, m, s = stats.t.fit(losses)
    q = stats.t.ppf(alpha, nu)
    f = stats.t.pdf(q, nu)

    var = m + s * q
    es = m + s * f * (nu + q**2) / ((1 - alpha) * (nu - 1))
    return (float(var), float(es))

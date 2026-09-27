"""
Let us consider L the losses of our portfolio and alpha the confidence
level. Then, we define:
    - VaR_alpha(L) as the quantile alpha of L
    - ES_alpha(L) as the mean of the losses beyond VaR_alpha(L)
These risk measures can be calculated either empirically, or theoretically
if a distribution for L is assumed.
"""

import warnings
import numpy as np
from scipy import stats
from arch import arch_model


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


def evt_var_es(
    losses: np.ndarray,
    alpha: float = 0.99,
    threshold_quantile: float = 0.90,
) -> tuple[float, float]:
    """
    VaR and ES with Extreme Value Theory (peaks over threshold).

    Extreme Value Theory models only the tail of the loss distribution.
    By the Pickands-Balkema-de Haan theorem, the excesses over a high
    threshold u converge to a Generalized Pareto Distribution (GPD):

        F_u(y) = P(L - u <= y | L > u) ~ G(y) = 1 - (1 + xi * y / beta)^(-1/xi)

    where beta > 0 is the scale and xi is the shape (tail index):
    xi > 0 heavy (power-law) tail, xi = 0 exponential tail (e.g. Normal),
    xi < 0 bounded tail.

    With n losses in the window and N_u of them above u (Smith, 1987):

        VaR = u + (beta / xi) * [ (n / N_u * (1 - alpha))^(-xi) - 1 ]
        ES  = (VaR + beta - xi * u) / (1 - xi),          valid for xi < 1

    and, in the limit xi -> 0 (exponential tail):

        VaR = u + beta * ln( N_u / (n * (1 - alpha)) )
        ES  = VaR + beta

    Choice of threshold (bias-variance trade-off): a low u includes
    non-extreme losses, where the GPD approximation is poor (bias); a high u
    leaves too few excesses to estimate xi and beta (variance). By default
    u is the 90% empirical quantile of the window: 50 excesses out of 500.

    Parameters
    ----------
    losses : np.ndarray
        Losses in the estimation window.
    alpha : float
        Confidence level.
    threshold_quantile : float
        Quantile of the losses used as threshold u.

    Returns
    -------
    (var, es) : tuple[float, float]
        Value-at-Risk and Expected Shortfall at level alpha.

    Raises
    ------
    ValueError
        If the fitted xi >= 1, since the ES is then infinite.
    """
    losses = np.asarray(losses)
    n = len(losses)

    u = np.quantile(losses, threshold_quantile)
    excesses = losses[losses > u] - u
    n_u = len(excesses)

    xi, _, beta = stats.genpareto.fit(excesses, floc=0)

    if xi >= 1:
        raise ValueError(f"xi = {xi:.3f} >= 1: the Expected Shortfall is infinite.")

    tail_prob_ratio = n / n_u * (1 - alpha)  # (1 - alpha) / P(L > u)

    if abs(xi) < 1e-8:
        var = u - beta * np.log(tail_prob_ratio)
        es = var + beta
    else:
        var = u + beta / xi * (tail_prob_ratio ** (-xi) - 1)
        es = (var + beta - xi * u) / (1 - xi)

    return float(var), float(es)


def _fit_garch_t(losses: np.ndarray):
    """
    Fit a GARCH(1,1) with Student-t innovations and forecast tomorrow's volatility.

    Losses are multiplied by 100 (percent units) for numerical stability of
    the optimizer, as recommended by the `arch` package; results are scaled back.

    Returns
    -------
    res : ARCHModelResult
        Fitted model (parameters, standardized residuals, ...).
    mu : float
        Estimated mean loss.
    sigma_next : float
        One-day-ahead conditional volatility forecast sigma_{t+1|t}.
    """
    model = arch_model(
        100 * np.asarray(losses), mean="Constant", vol="GARCH", p=1, q=1, dist="t"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence occasional convergence warnings
        res = model.fit(disp="off")

    mu = res.params["mu"] / 100
    variance_next = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
    sigma_next = np.sqrt(variance_next) / 100
    return res, mu, sigma_next


def _standardized_t_var_es(nu: float, alpha: float) -> tuple[float, float]:
    """VaR and ES of a Student-t rescaled to unit variance."""
    q = stats.t.ppf(alpha, nu)
    scale = np.sqrt((nu - 2) / nu)
    var_z = scale * q
    es_z = scale * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return var_z, es_z


def garch_t_var_es(losses: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    VaR and ES from a GARCH(1,1) model with Student-t innovations.

    Model:
        L_t = mu + sigma_t * z_t,   z_t ~ standardized t_nu (unit variance)
        sigma_t^2 = omega + a * eps_{t-1}^2 + b * sigma_{t-1}^2

    Parameters are estimated by maximum likelihood on the window, and the
    risk measures use tomorrow's conditional volatility sigma_{t+1|t}:

        VaR = mu + sigma_{t+1|t} * q_alpha(z)
        ES  = mu + sigma_{t+1|t} * ES_alpha(z)

    Parameters
    ----------
    losses : np.ndarray
        Losses in the estimation window, in chronological order.
    alpha : float
        Confidence level (e.g. 0.99).

    Returns
    -------
    (var, es) : tuple[float, float]
    """
    res, mu, sigma_next = _fit_garch_t(losses)
    var_z, es_z = _standardized_t_var_es(res.params["nu"], alpha)
    return float(mu + sigma_next * var_z), float(mu + sigma_next * es_z)


def garch_evt_var_es(
    losses: np.ndarray, alpha: float = 0.99, threshold_quantile: float = 0.90
) -> tuple[float, float]:
    """
    VaR and ES with GARCH-filtered EVT (McNeil and Frey, 2000).

    1. Fit a GARCH(1,1) to the losses and compute the standardized residuals
       z_t = (L_t - mu) / sigma_t, which are much closer to i.i.d.
    2. Apply EVT (peaks over threshold, GPD) to the residuals to obtain
       q_alpha(z) and ES_alpha(z) without assuming their distribution.
    3. Rescale with tomorrow's conditional volatility:

        VaR = mu + sigma_{t+1|t} * q_alpha(z)
        ES  = mu + sigma_{t+1|t} * ES_alpha(z)

    GARCH captures the dynamics (volatility clustering) and EVT the tail.

    Parameters
    ----------
    losses : np.ndarray
        Losses in the estimation window, in chronological order.
    alpha : float
        Confidence level (e.g. 0.99).
    threshold_quantile : float
        Quantile of the residuals used as EVT threshold (default 0.90).

    Returns
    -------
    (var, es) : tuple[float, float]
    """
    res, mu, sigma_next = _fit_garch_t(losses)
    z = np.asarray(res.std_resid)
    z = z[~np.isnan(z)]
    var_z, es_z = evt_var_es(z, alpha, threshold_quantile)
    return float(mu + sigma_next * var_z), float(mu + sigma_next * es_z)

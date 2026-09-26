"""
We here document three aspects of financial returns:
    1. Heavy tails
    2. Volatility clusters
    3. Lack of normality
"""

import numpy as np
import pandas as pd
from scipy import stats
from varbacktest.data import load_prices, log_returns
import matplotlib.pyplot as plt
from pathlib import Path


def summary_statistics(r: pd.Series) -> pd.Series:
    """
    Parameters
    ----------
    r : pd.Series
        financial returns

    Returns
    -------
    pd.Series
        number observations, annualized mean, annualized std, skewness,
        kurtosis, minimum, minimum date, maximum, maximum date,
        Jarque-Bera p-value
    """
    return pd.Series(
        {
            "n_obs": len(r),
            "mean_ann": r.mean() * 252,
            "vol_ann": r.std() * np.sqrt(252),
            "skewness": r.skew(),
            "excess_kurtosis": r.kurt(),
            "min_return": r.min(),
            "min_date": r.idxmin(),
            "max_return": r.max(),
            "max_date": r.idxmax(),
            "jarque_bera_p": stats.jarque_bera(r).pvalue,
        },
        name=r.name,
    )


if __name__ == "__main__":
    returns = {}
    prices = {}
    for t in ["^GSPC", "^IBEX"]:
        p = load_prices(t)
        r = log_returns(p)
        returns[t] = r
        prices[t] = p

    table = pd.concat([summary_statistics(r) for r in returns.values()], axis=1)
    print(table)

    FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    # Rolling 21-day annualised volatility
    fig, ax = plt.subplots(figsize=(10, 4))
    for ticker, r in returns.items():
        vol = r.rolling(21).std() * np.sqrt(252)
        ax.plot(vol.index, vol, label=ticker, linewidth=0.8)
    ax.set_title("Rolling 21-day annualised volatility")
    ax.set_ylabel("Volatility")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rolling_volatility.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for ticker, p in prices.items():
        ax.plot(p.index, p, label=ticker, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_title("Prices evolution")
    ax.set_ylabel("Price (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "log_prices.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (ticker, r) in zip(axes, returns.items()):
        ax.plot(r.index, r, linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{ticker} daily log-returns")
        ax.set_ylabel("Log-return")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "returns.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (ticker, r) in zip(axes, returns.items()):
        x = np.linspace(r.min(), r.max(), 500)
        normal_pdf = stats.norm.pdf(x, r.mean(), r.std())
        for ax, scale in zip(row, ["linear", "log"]):
            ax.hist(r, bins=200, density=True, alpha=0.6, label="Empirical")
            ax.plot(x, normal_pdf, color="red", linewidth=1.2, label="Normal fit")
            ax.set_yscale(scale)
            if scale == "log":
                ax.set_ylim(bottom=1e-2)
            ax.set_title(f"{ticker} - {scale} scale")
            ax.set_xlabel("Daily log-return")
            ax.set_ylabel("Density")
            ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "histogram.png", dpi=150)
    plt.close(fig)

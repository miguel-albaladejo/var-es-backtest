"""
 Formal backtests of the rolling 99% VaR forecasts.

Reads the forecasts produced by scripts/02_rolling_var.py and applies:
  * Kupiec unconditional coverage test
  * Christoffersen independence and conditional coverage tests
  * Basel traffic-light zones (250-day window)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from varbacktest.backtests import (
    basel_traffic_light,
    christoffersen_test,
    kupiec_test,
    acerbi_szekely_test,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

ALPHA = 0.99
MODELS = {
    "hist": "Historical",
    "norm": "Normal",
    "t": "Student-t",
    "evt": "EVT (POT-GPD)",
    "garch_t": "GARCH-t",
    "garch_evt": "GARCH-EVT",
}
TICKERS = ["^GSPC", "^IBEX"]


def backtest_table(forecasts: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Run all backtests for every model in a forecasts table.

    Parameters
    ----------
    forecasts : pd.DataFrame
        Output of step 2: a "loss" column and one "var_<model>" column per model.
    alpha : float
        Confidence level of the VaR.

    Returns
    -------
    pd.DataFrame
        One row per model with exception counts, test statistics, p-values
        and the share of days spent in each Basel zone.
    """
    rows = {}
    for key, name in MODELS.items():
        exceptions = (forecasts["loss"] > forecasts[f"var_{key}"]).astype(int)

        kup = kupiec_test(exceptions, alpha)
        chr_ = christoffersen_test(exceptions, alpha)
        basel = basel_traffic_light(exceptions)
        zone_share = basel["zone"].value_counts(normalize=True)
        acs = acerbi_szekely_test(
            forecasts["loss"], forecasts[f"var_{key}"], forecasts[f"es_{key}"], alpha
        )

        rows[name] = {
            "exceptions": kup["x"],
            "expected": round(kup["n"] * (1 - alpha), 1),
            "rate_%": round(100 * kup["rate"], 2),
            "p_kupiec": kup["p_value"],
            "pi01_%": round(100 * chr_["pi01"], 2),
            "pi11_%": round(100 * chr_["pi11"], 2),
            "p_indep": chr_["p_ind"],
            "p_cc": chr_["p_cc"],
            "green_%": round(100 * zone_share.get("green", 0.0), 1),
            "yellow_%": round(100 * zone_share.get("yellow", 0.0), 1),
            "red_%": round(100 * zone_share.get("red", 0.0), 1),
            "max_250d": int(basel["count"].max()),
            "loss/ES": round(acs["mean_loss_to_es"], 2),
            "z2": round(acs["z2"], 3),
            "p_z2": acs["p_z2"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_basel_counts(forecasts: pd.DataFrame, ticker: str) -> None:
    """
    Plot the rolling 250-day exception count of each model with Basel zones.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axhspan(0, 4.5, color="green", alpha=0.08)
    ax.axhspan(4.5, 9.5, color="gold", alpha=0.15)
    ax.axhspan(9.5, 100, color="red", alpha=0.08)

    top = 0
    for key, name in MODELS.items():
        exceptions = (forecasts["loss"] > forecasts[f"var_{key}"]).astype(int)
        count = basel_traffic_light(exceptions)["count"]
        ax.plot(count.index, count, linewidth=1.0, label=name)
        top = max(top, count.max())

    ax.set_ylim(0, top + 2)
    ax.set_title(
        f"{ticker}: 99% VaR exceptions in the last 250 days (Basel traffic light)"
    )
    ax.set_ylabel("Exceptions")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"basel_{ticker.replace('^', '')}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    for ticker in TICKERS:
        name = ticker.replace("^", "")
        forecasts = pd.read_csv(
            RESULTS_DIR / f"forecasts_{name}.csv", index_col=0, parse_dates=True
        )

        table = backtest_table(forecasts)
        table.to_csv(RESULTS_DIR / f"backtests_{name}.csv")

        print(
            f"\n{ticker} - backtests of the 99% VaR "
            f"({forecasts.index[0].date()} to {forecasts.index[-1].date()})"
        )
        print(table.to_string(float_format=lambda v: f"{v:.3g}"))

        plot_basel_counts(forecasts, ticker)

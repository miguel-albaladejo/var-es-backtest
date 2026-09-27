"""
Crisis analysis and final scorecard.

Reads the forecasts and backtest tables and produces:
  * exceptions and average VaR of every model in each crisis period
  * the Basel market-risk capital charge implied by each model
  * a final scorecard combining all backtests
  * crisis figures with three representative models
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from varbacktest.backtests import basel_traffic_light

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

ALPHA = 0.99
SIGNIFICANCE = 0.05
MODELS = {
    "hist": "Historical",
    "norm": "Normal",
    "t": "Student-t",
    "evt": "EVT (POT-GPD)",
    "garch_t": "GARCH-t",
    "garch_evt": "GARCH-EVT",
}
TICKERS = ["^GSPC", "^IBEX"]
PERIODS = {
    "GFC (Sep 2008 - Jun 2009)": ("2008-09-01", "2009-06-30"),
    "Euro crisis (H2 2011)": ("2011-07-01", "2011-12-31"),
    "COVID (Feb - Jun 2020)": ("2020-02-20", "2020-06-30"),
    "Rate hikes (2022)": ("2022-01-01", "2022-12-31"),
    "Calm year (2017)": ("2017-01-01", "2017-12-31"),
}
# Models shown in the crisis figures (one per family, to keep them readable)
PLOT_MODELS = {"norm": "Normal", "hist": "Historical", "garch_evt": "GARCH-EVT"}


def crisis_table(forecasts: pd.DataFrame) -> pd.DataFrame:
    """
    Exceptions (vs. expected) of each model in each period.

    Returns a table with one row per period and one column per model; each
    cell is "exceptions / expected".
    """
    rows = {}
    for period, (start, end) in PERIODS.items():
        df = forecasts.loc[start:end]
        expected = (1 - ALPHA) * len(df)
        row = {"days": len(df)}
        for key, name in MODELS.items():
            exceptions = int((df["loss"] > df[f"var_{key}"]).sum())
            row[name] = f"{exceptions} / {expected:.1f}"
        rows[period] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def basel_capital_charge(var: pd.Series, exceptions: pd.Series) -> pd.Series:
    """
    Daily Basel market-risk capital charge implied by a VaR model.

    Basel (1996) internal-models formula, without the stressed-VaR add-on:

        capital_t = max( VaR_t , m_t * mean(VaR over the last 60 days) )

    where m_t = 3 + plus factor is the traffic-light multiplier
    (3 in the green zone, up to 4 in the red zone).
    """
    multiplier = basel_traffic_light(exceptions)["multiplier"]
    var_60 = var.rolling(60).mean()
    capital = pd.concat([var, multiplier * var_60], axis=1).max(axis=1, skipna=False)
    return capital.loc[multiplier.index].dropna()


def scorecard(forecasts: pd.DataFrame, backtests: pd.DataFrame) -> pd.DataFrame:
    """Final summary of every model: tests passed and cost in capital."""
    rows = {}
    for key, name in MODELS.items():
        exceptions = (forecasts["loss"] > forecasts[f"var_{key}"]).astype(int)
        capital = basel_capital_charge(forecasts[f"var_{key}"], exceptions)
        bt = backtests.loc[name]
        passed = {
            "Kupiec": bt["p_kupiec"] > SIGNIFICANCE,
            "Indep.": bt["p_indep"] > SIGNIFICANCE,
            "ES (Z2)": bt["p_z2"] > SIGNIFICANCE,
        }
        rows[name] = {
            **{test: "pass" if ok else "fail" for test, ok in passed.items()},
            "tests_passed": f"{sum(passed.values())}/3",
            "red_%": bt["red_%"],
            "avg_VaR_%": round(100 * forecasts[f"var_{key}"].mean(), 2),
            "avg_capital_%": round(100 * capital.mean(), 2),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_crisis(forecasts: pd.DataFrame, ticker: str) -> None:
    """Losses, VaR of three representative models and exceptions in 2008 and 2020."""
    panels = {
        "GFC (2008-2009)": ("2008-06-01", "2009-06-30"),
        "COVID (2020)": ("2020-01-01", "2020-06-30"),
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.5), sharey=True, gridspec_kw={"width_ratios": [2, 1]}
    )
    for ax, (title, (start, end)) in zip(axes, panels.items()):
        df = forecasts.loc[start:end]
        ax.bar(
            df.index, 100 * df["loss"], width=1.0, color="lightgrey", label="Daily loss"
        )
        # Decreasing marker sizes: a day that breaches several models shows
        # concentric dots instead of one dot hiding the others.
        for (key, name), size in zip(PLOT_MODELS.items(), [45, 22, 8]):
            line = ax.plot(
                df.index,
                100 * df[f"var_{key}"],
                linewidth=1.2,
                label=f"99% VaR - {name}",
            )
            exc = df[df["loss"] > df[f"var_{key}"]]
            ax.scatter(
                exc.index,
                100 * exc["loss"],
                s=size,
                zorder=3,
                color=line[0].get_color(),
            )
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Loss (%)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{ticker}: 99% VaR in crises (dots = exceptions of each model)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"crisis_{ticker.replace('^', '')}.png", dpi=150)
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
        backtests = pd.read_csv(RESULTS_DIR / f"backtests_{name}.csv", index_col=0)

        crises = crisis_table(forecasts)
        summary = scorecard(forecasts, backtests)
        crises.to_csv(RESULTS_DIR / f"crises_{name}.csv")
        summary.to_csv(RESULTS_DIR / f"scorecard_{name}.csv")

        print(f"\n===== {ticker} =====")
        print("\nExceptions / expected in each period:")
        print(crises.to_string())
        print("\nFinal scorecard:")
        print(summary.to_string())

        plot_crisis(forecasts, ticker)

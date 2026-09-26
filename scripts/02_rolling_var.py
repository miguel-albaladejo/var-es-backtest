"""
Rolling one-day 99% VaR/ES forecasts and exception counts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from varbacktest.data import load_prices, log_returns, to_losses
from varbacktest.forecast import rolling_var_es
from varbacktest.models import historical_var_es, normal_var_es, student_t_var_es

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

ALPHA = 0.99
WINDOW = 500
MODELS = {
    "hist": ("Historical", historical_var_es),
    "norm": ("Normal", normal_var_es),
    "t": ("Student-t", student_t_var_es),
}
PERIODS = {"2008": ("2007", "2009"), "2020": ("2020", "2020")}


def count_exceptions(table: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """
    Count VaR exceptions (loss > VaR) for each model in `table`.

    Parameters
    ----------
    table : pd.DataFrame
        Must contain a "loss" column and one "var_<model>" column per model.
    alpha : float
        Confidence level of the VaR.

    Returns
    -------
    pd.DataFrame
        One row per model with the observed and expected number of
        exceptions, the ratio between them and the observed exception rate.
    """
    n = len(table)
    expected = (1 - alpha) * n
    rows = {}
    for key, (name, _) in MODELS.items():
        exceptions = int((table["loss"] > table[f"var_{key}"]).sum())
        rows[name] = {
            "exceptions": exceptions,
            "expected": round(expected, 1),
            "ratio": round(exceptions / expected, 2),
            "rate_%": round(100 * exceptions / n, 2),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    for ticker in ["^GSPC", "^IBEX"]:
        loss = to_losses(log_returns(load_prices(ticker))).rename("loss")

        # --- Rolling forecasts for every model ---
        forecasts = [loss]
        for key, (name, model) in MODELS.items():
            print(f"{ticker}: computing {name} VaR/ES...")
            out = rolling_var_es(loss, model, window=WINDOW, alpha=ALPHA)
            forecasts.append(out.add_suffix(f"_{key}"))

        # Drop the first WINDOW days, which have no forecast
        table = pd.concat(forecasts, axis=1).dropna()
        table.to_csv(RESULTS_DIR / f"forecasts_{ticker.replace('^', '')}.csv")

        # --- Exception counts ---
        print(
            f"\n{ticker} - 99% VaR exceptions "
            f"({table.index[0].date()} to {table.index[-1].date()}, {len(table)} days)"
        )
        print(count_exceptions(table))
        print()

        # --- Figures: losses vs VaR in crisis periods ---
        for label, (start, end) in PERIODS.items():
            df = table.loc[start:end]
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.bar(
                df.index, df["loss"], width=1.0, color="lightgrey", label="Daily loss"
            )
            for key, (name, _) in MODELS.items():
                ax.plot(
                    df.index, df[f"var_{key}"], linewidth=1.0, label=f"99% VaR - {name}"
                )
            # Exceptions of the historical model
            exc = df[df["loss"] > df["var_hist"]]
            ax.scatter(
                exc.index,
                exc["loss"],
                color="red",
                s=12,
                zorder=3,
                label="Exception (historical VaR)",
            )
            ax.set_title(f"{ticker}: daily losses vs 99% VaR ({start}-{end})")
            ax.set_ylabel("Loss")
            ax.legend(loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"var_{ticker.replace('^', '')}_{label}.png", dpi=150)
            plt.close(fig)

# Empirical Backtesting of VaR and Expected Shortfall

One-day Value-at-Risk (VaR) and Expected Shortfall (ES) models for the
**S&P 500** and the **IBEX 35** (2000–2026), evaluated with formal
statistical backtests, with a focus on crisis periods (2008, 2020, 2022).

This project is the empirical counterpart of my BSc Mathematics thesis,
*Risk Measures*, a functional-analytic study of convex and coherent risk
measures. The thesis explains **why** VaR is not a coherent risk measure
and Expected Shortfall is; this project tests **how** they behave on real data.

> Work in progress — see [Roadmap](#roadmap).

## Data

- Daily adjusted closing prices from Yahoo Finance: `^GSPC` (S&P 500) and `^IBEX` (IBEX 35), Jan 2000 – Sep 2026.
- Both are **price indices** (dividends excluded).
- Each index is treated separately, since they follow different exchange holiday calendars.

## Conventions

- Log-returns: $r_t = \ln(P_t / P_{t-1})$.
- Losses: $L_t = -r_t$, so VaR and ES are reported as **positive numbers**.
- A VaR **exception** on day $t$ occurs when $L_t > \mathrm{VaR}_t$.
- Forecasts for day $t$ only use information available up to day $t-1$ (no look-ahead bias).

## Stylized facts

|                        |        S&P 500      |       IBEX 35       |
|------------------------|---------------------|---------------------|
| Number of observations |        6722         |       6800          |
| Annualized mean        |        6.3%         |       2.0%          |
| Annualized volatility  |       19.2%         |      22.1%          |
| Skewness               |       −0.35         |      −0.32          |
| Excess kurtosis        |       10.7          |       8.2           |
| Worst day              | −12.8% (2020-03-16) | −15.2% (2020-03-12) |
| Best day               | +11.0% (2008-10-13) | +13.5% (2010-05-10) |
| Jarque–Bera p-value    |      < 0.001        |     < 0.001         |

*Returns are log-returns.*

**Key takeaways**

- **Fat tails.** The worst S&P 500 day was a **10.5** $\sigma$ move. Under normality, such a day would be expected roughly once every $10^{23}$ years, yet it happened in this 26-year sample. A Gaussian VaR therefore underestimates tail risk.
- **Negative skewness.** Large losses are more frequent than large gains of the same size.
- **Volatility clustering.** The best days of both indices occurred in the middle of crises (Lehman 2008, the euro crisis 2010). Extreme moves come in clusters, which motivates GARCH-type models.
- **Volatility ≠ tail risk.** The IBEX is more volatile, but the S&P 500 has heavier tails.

**Volatility clustering.** Rolling one-month volatility ranges from below 10% in calm years (2017) to around 80% in October 2008 and close to 100% in March 2020.

![Rolling volatility](figures/rolling_volatility.png)

**Fat tails.** On a log scale, the Normal density (red) vanishes beyond ±5%, while observed daily returns reach −10% to −15%.

![Return distribution vs Normal](figures/histogram.png)

## Project structure

```
src/varbacktest/   library code (data, models, backtests)
tests/             unit tests (pytest)
scripts/           analysis scripts that produce tables and figures
figures/           generated charts
data/              local price cache (not tracked by git)
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/01_explore_data.py   # stylized facts table and figures
pytest                              # unit tests
```

## Roadmap

- [x] Data pipeline: download, cleaning, local cache
- [x] Exploratory analysis and stylized facts
- [ ] VaR/ES models: historical simulation, Normal, Student-t (rolling window)
- [ ] VaR backtests: Kupiec, Christoffersen, Basel traffic light
- [ ] GARCH(1,1) with Student-t innovations
- [ ] ES backtest: Acerbi–Szekely
- [ ] Crisis analysis (2008, 2020, 2022) and final results
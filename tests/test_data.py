import numpy as np
import pandas as pd

from varbacktest.data import log_returns, to_losses


def make_series(values, dates, name="TEST"):
    """Small helper to build a price series for the tests."""
    return pd.Series(values, index=pd.to_datetime(dates), name=name)


def test_log_returns_values():
    prices = make_series(
        [100.0, 110.0, 99.0], ["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    r = log_returns(prices)
    np.testing.assert_allclose(r.values, [np.log(1.1), np.log(0.9)])


def test_log_returns_drops_first_date():
    prices = make_series(
        [100.0, 110.0, 99.0], ["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    r = log_returns(prices)
    assert len(r) == 2
    assert r.index[0] == pd.Timestamp("2024-01-02")


def test_log_returns_keeps_name():
    prices = make_series(
        [100.0, 110.0, 99.0], ["2024-01-01", "2024-01-02", "2024-01-03"], name="SPX"
    )
    r = log_returns(prices)
    assert r.name == "SPX"

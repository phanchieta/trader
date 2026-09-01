import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_ohlcv():
    def _make(n: int = 500, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        px = 30.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
        idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
        idx.name = "time"
        return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                             "close": px, "volume": 1e6}, index=idx)
    return _make


@pytest.fixture
def intraday_ohlcv():
    """~1min de 10:00 às 11:00 BRT (13:00-14:00 UTC), vários dias."""
    def _make(days: int = 6, per_day: int = 60, seed: int = 2) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        stamps, prices = [], []
        for d in range(days):
            base = pd.Timestamp("2024-03-04", tz="UTC") + pd.Timedelta(days=d)
            p = 40.0 + np.cumsum(rng.normal(0.0, 0.05, per_day))
            for m in range(per_day):
                stamps.append(base + pd.Timedelta(hours=13, minutes=m))
            prices.extend(p.tolist())
        px = np.asarray(prices)
        idx = pd.DatetimeIndex(stamps, name="time")
        return pd.DataFrame({"open": px, "high": px + 0.02, "low": px - 0.02,
                             "close": px, "volume": 1e5}, index=idx)
    return _make

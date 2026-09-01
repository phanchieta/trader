"""Trava a correção do bug de resolução de datetime (parquet vinha em ms e o
cálculo assumia ns, estourando o CAGR)."""
import numpy as np
import pandas as pd

from trader.backtest.metrics import _periods_per_year, summarize

_EMPTY_TRADES = pd.DataFrame(columns=["price", "shares", "fee"])


def _equity(n_days=3650, start=100_000.0, cagr=0.08, unit="ns"):
    idx = pd.date_range("2010-01-01", periods=n_days, freq="D", tz="UTC").as_unit(unit)
    idx.name = "time"
    growth = (1.0 + cagr) ** (np.arange(n_days) / 365.25)
    return pd.Series(start * growth, index=idx, name="equity")


def test_periods_per_year_diario_qualquer_resolucao():
    for unit in ("ns", "us", "ms", "s"):
        idx = pd.date_range("2020-01-01", periods=400, freq="B", tz="UTC").as_unit(unit)
        assert abs(_periods_per_year(idx) - 252.0) < 1e-6


def test_cagr_recuperado_sem_overflow_com_indice_ms():
    stats = summarize(_equity(cagr=0.08, unit="ms"), _EMPTY_TRADES, 100_000.0)
    assert stats["cagr"] is not None
    assert abs(stats["cagr"] - 0.08) < 0.005


def test_cagr_identico_ns_e_ms():
    a = summarize(_equity(unit="ns"), _EMPTY_TRADES, 100_000.0)["cagr"]
    b = summarize(_equity(unit="ms"), _EMPTY_TRADES, 100_000.0)["cagr"]
    assert abs(a - b) < 1e-9

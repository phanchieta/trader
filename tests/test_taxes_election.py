import numpy as np
import pandas as pd

from trader.risk.election import haircut_series
from trader.taxes import after_tax_equity, cagr


def _daily(n=800, cagr_alvo=0.20, start="2016-01-04"):
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC")
    idx.name = "time"
    g = (1 + cagr_alvo) ** (np.arange(n) / 252.0)
    return pd.Series(100_000.0 * g, index=idx, name="equity")


def test_after_tax_reduz_o_cagr_quando_ha_lucro():
    eq = _daily(cagr_alvo=0.25)
    at = after_tax_equity(eq, rate=0.155)
    assert cagr(at) < cagr(eq)
    # ~15,5% do ganho vira imposto -> CAGR pós fica entre 80% e 100% do pré
    assert 0.80 < cagr(at) / cagr(eq) < 1.0


def test_after_tax_nao_taxa_prejuizo():
    idx = pd.date_range("2016-01-04", periods=500, freq="B", tz="UTC")
    idx.name = "time"
    eq = pd.Series(100_000.0 * (0.7) ** (np.arange(500) / 500), index=idx)  # só cai
    at = after_tax_equity(eq, rate=0.155)
    # sem lucro mensal -> sem imposto -> curva ~igual
    m = eq.resample("ME").last()
    assert abs(at.iloc[-1] - m.iloc[-1]) < 1.0


def test_haircut_series_corta_na_janela_eleitoral():
    idx = pd.date_range("2018-01-01", "2018-12-31", freq="D", tz="UTC")
    hc = haircut_series(idx, factor=0.5)
    assert hc.loc["2018-06-01"] == 1.0
    assert hc.loc["2018-10-15"] == 0.5
    assert hc.loc["2018-12-01"] == 1.0


def test_haircut_series_ano_nao_eleitoral_fica_em_1():
    idx = pd.date_range("2019-01-01", "2019-12-31", freq="D", tz="UTC")
    assert (haircut_series(idx, factor=0.5) == 1.0).all()

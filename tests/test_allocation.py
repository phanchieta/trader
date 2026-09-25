import numpy as np
import pandas as pd

from trader.allocation import dual_momentum, faber_sma, month_ends, run_allocation, splice, static_weights


def _cal(n=400, start="2020-01-01"):
    return pd.bdate_range(start, periods=n)


def test_emenda_usa_referencia_antes_e_etf_depois():
    cal = _cal(10)
    proxy = pd.Series(np.linspace(100, 109, 10), index=cal)
    etf = pd.Series([50.0, 55.0, 60.5, 60.5, 60.5], index=cal[5:])
    s = splice(etf, proxy, cal, drag_aa=0.0)
    assert np.isclose(s.iloc[0], 1.0)
    assert np.isclose(s.iloc[4] / s.iloc[3], 104 / 103)          # referência
    assert np.isclose(s.iloc[6] / s.iloc[5], 55 / 50)            # ETF
    d = splice(None, proxy, cal, drag_aa=0.252)
    assert np.isclose(d.iloc[1] / d.iloc[0], 101 / 100 - 0.001)  # desconto diário


def test_faber_so_segura_acima_da_media_de_10_meses():
    cal = _cal(600)
    up = pd.Series(100 * np.exp(np.linspace(0, 1, 600)), index=cal)
    down = pd.Series(100 * np.exp(np.linspace(0, -1, 600)), index=cal)
    w = faber_sma(pd.DataFrame({"A": up, "B": down}), months=10)
    assert w.index.equals(month_ends(cal))
    late = w.iloc[-1]
    assert late["A"] == 0.5 and late["B"] == 0.0                  # 1/N para quem passou
    assert (w.iloc[:9].sum(axis=1) == 0).all()                    # sem histórico: CDI


def test_dual_momentum_vai_para_o_cdi_quando_tudo_perde_do_juro():
    cal = _cal(600)
    flat = pd.Series(100.0, index=cal)
    slow = pd.Series(100 * np.exp(np.linspace(0, 0.02, 600)), index=cal)
    cdi = pd.Series(0.0004, index=cal)                            # ~10% a.a.
    w = dual_momentum(pd.DataFrame({"A": flat, "B": slow}), cdi, months=12)
    assert (w.iloc[-1] == 0).all()
    fast = pd.Series(100 * np.exp(np.linspace(0, 0.8, 600)), index=cal)
    w2 = dual_momentum(pd.DataFrame({"A": flat, "B": fast}), cdi, months=12)
    assert w2.iloc[-1]["B"] == 1.0


def test_motor_executa_no_dia_seguinte_e_cobra_custo():
    cal = _cal(60)
    px = pd.Series(100.0, index=cal)
    px.iloc[30:] = 110.0                                          # salta 10% no dia 30
    prices = pd.DataFrame({"A": px})
    me = month_ends(cal)
    tgt = pd.DataFrame({"A": 1.0}, index=me)
    cdi = pd.Series(0.0, index=cal)
    r = run_allocation(prices, tgt, cdi, cost_bps=10.0)
    first_exec = cal.get_loc(me[0]) + 1
    assert r.weights["A"].iloc[first_exec - 1] == 0.0 and r.weights["A"].iloc[first_exec] > 0.99
    assert np.isclose(r.equity.iloc[first_exec], 100_000 * (1 - 0.001))
    cash = run_allocation(prices, tgt * 0.0, pd.Series(0.001, index=cal))
    assert np.isclose(cash.equity.iloc[-1], 100_000 * 1.001 ** 59)
    assert abs(cash.stats["alpha"]) < 1e-3                        # 100% CDI: alpha zero


def test_alpha_contra_cdi_com_fuso_diferente():
    cal = _cal(300)
    prices = pd.DataFrame({"A": pd.Series(100.0, index=cal)})
    tgt = static_weights(prices, {})
    cdi_utc = pd.Series(0.0004, index=cal.tz_localize("UTC"))
    r = run_allocation(prices, tgt, cdi_utc)
    assert abs(r.stats["alpha"]) < 1e-3


def test_banda_evita_rebalanceamento_pequeno():
    cal = _cal(300)
    rng = np.random.default_rng(1)
    a = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=cal)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=cal)
    prices = pd.DataFrame({"A": a, "B": b})
    tgt = static_weights(prices, {"A": 0.5, "B": 0.5})
    cdi = pd.Series(0.0, index=cal)
    tight = run_allocation(prices, tgt, cdi, order_cost=4.9)
    loose = run_allocation(prices, tgt, cdi, order_cost=4.9, band=0.2)
    assert loose.stats["ordens_aa"] < tight.stats["ordens_aa"]

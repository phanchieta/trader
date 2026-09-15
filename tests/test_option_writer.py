import numpy as np
import pandas as pd

from trader.options import implied_vol, monthly_expiries, pick_series, run_option_writer
from trader.options.bs import bs_price


def _chain(dates, expiry, strikes, *, call_bid, put_bid, call_ask=None, put_ask=None):
    rows = []
    for d in dates:
        for k in strikes:
            rows.append({"date": d, "codneg": f"C{int(k)}", "tpmerc": "070", "strike": float(k),
                         "expiry": expiry, "bid": call_bid(d, k), "ask": (call_ask or call_bid)(d, k) + 0.02,
                         "close": call_bid(d, k), "n_trades": 10})
            rows.append({"date": d, "codneg": f"P{int(k)}", "tpmerc": "080", "strike": float(k),
                         "expiry": expiry, "bid": put_bid(d, k), "ask": (put_ask or put_bid)(d, k) + 0.02,
                         "close": put_bid(d, k), "n_trades": 10})
    return pd.DataFrame(rows)


DAYS = pd.DatetimeIndex(pd.date_range("2024-03-01", periods=6, freq="B"))
EXP = DAYS[4]
STRIKES = [80, 85, 90, 95, 100, 105, 110]
ZERO_CDI = pd.Series(0.0, index=DAYS.tz_localize("UTC"))
KW = dict(fee_option=0.0, fee_exercise=0.0, min_dte=0, max_dte=60)


def test_cadeia_mensal_e_escolha_de_strike():
    ch = _chain(DAYS[:1], EXP, STRIKES, call_bid=lambda d, k: 2.0, put_bid=lambda d, k: 1.5)
    assert list(monthly_expiries(ch)) == [EXP]
    row = pick_series(ch, kind="call", spot=100.0, moneyness=0.05, expiry=EXP)
    assert row["strike"] == 105.0
    row = pick_series(ch, kind="put", spot=100.0, moneyness=0.05, expiry=EXP)
    assert row["strike"] == 95.0


def test_venda_coberta_limita_a_alta():
    spot = pd.Series([100, 100, 100, 120, 120, 120], index=DAYS, dtype=float)
    tr = spot.pct_change().fillna(0.0)
    ch = _chain(DAYS[:4], EXP, STRIKES, call_bid=lambda d, k: 2.0, put_bid=lambda d, k: 1.0)
    r = run_option_writer(spot, tr, ch, ZERO_CDI, kind="call", moneyness=0.05, **KW)
    # 10.000 ações, prêmio 2, exercida a 105 com spot 120: 1,2 mi + 20 mil - 150 mil
    assert np.isclose(r.equity.iloc[4], 1_200_000 + 20_000 - 150_000)
    assert r.trades.iloc[0]["strike"] == 105.0


def test_put_coberta_por_caixa_exercida_e_trava_de_baixa():
    spot = pd.Series([100, 100, 90, 80, 80, 80], index=DAYS, dtype=float)
    tr = spot.pct_change().fillna(0.0)
    ch = _chain(DAYS[:4], EXP, STRIKES, call_bid=lambda d, k: 0.5, put_bid=lambda d, k: 1.5 if k == 95 else 0.5)
    naked = run_option_writer(spot, tr, ch, ZERO_CDI, kind="put", moneyness=0.05, **KW)
    qty = np.floor(1_000_000 / 95 / 100) * 100
    assert np.isclose(naked.equity.iloc[4], 1_000_000 + qty * 1.5 - qty * 15)
    spread = run_option_writer(spot, tr, ch, ZERO_CDI, kind="put", moneyness=0.05, hedge=0.15, **KW)
    # compra a put 85 pelo ask (0,52) e recebe 5 por ação no vencimento
    assert np.isclose(spread.equity.iloc[4], 1_000_000 + qty * (1.5 - 0.52) - qty * 15 + qty * 5)
    assert spread.equity.iloc[4] > naked.equity.iloc[4]


def test_venda_coberta_nunca_alavanca_depois_de_exercicio():
    days = pd.DatetimeIndex(pd.date_range("2024-03-01", periods=12, freq="B"))
    e1, e2 = days[4], days[9]
    spot = pd.Series([100, 100, 100, 120, 120, 120, 120, 60, 60, 60, 60, 60], index=days, dtype=float)
    tr = spot.pct_change().fillna(0.0)
    strikes = [95, 100, 105, 110, 115, 120, 126, 130]
    ch = pd.concat([_chain(days[:5], e1, strikes, call_bid=lambda d, k: 2.0, put_bid=lambda d, k: 1.0),
                    _chain(days[4:10], e2, strikes, call_bid=lambda d, k: 2.0, put_bid=lambda d, k: 1.0)])
    cdi = pd.Series(0.0, index=days.tz_localize("UTC"))
    r = run_option_writer(spot, tr, ch, cdi, kind="call", moneyness=0.05, stock_fee=0.0, **KW)
    stock_dd = (spot / spot.cummax() - 1).min()
    cc_dd = (r.equity / r.equity.cummax() - 1).min()
    assert cc_dd >= stock_dd                                    # coberta nunca cai mais que a ação
    assert len(r.trades) == 2


def test_oferta_lixo_nao_contamina_a_marcacao():
    spot = pd.Series([100, 100, 100, 100, 100, 100], index=DAYS, dtype=float)
    tr = spot.pct_change().fillna(0.0)
    ch = _chain(DAYS[:4], EXP, STRIKES, call_bid=lambda d, k: 0.5, put_bid=lambda d, k: 1.5)
    ch.loc[(ch["date"] == DAYS[2]) & (ch["codneg"] == "P95"), ["ask", "n_trades"]] = [60.0, 0]
    r = run_option_writer(spot, tr, ch, ZERO_CDI, kind="put", moneyness=0.05, **KW)
    qty = np.floor(1_000_000 / 95 / 100) * 100
    assert r.equity.iloc[2] > 1_000_000 - qty * 2.0          # ignorou o ask de R$60


def test_volatilidade_implicita_recupera_a_vol():
    p = bs_price(30.0, 31.0, 30 / 365, 0.10, 0.35, "call")
    assert abs(implied_vol(p, 30.0, 31.0, 30 / 365, 0.10, "call") - 0.35) < 1e-4
    p = bs_price(30.0, 28.0, 30 / 365, 0.10, 0.5, "put")
    assert abs(implied_vol(p, 30.0, 28.0, 30 / 365, 0.10, "put") - 0.5) < 1e-4

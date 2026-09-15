import numpy as np
import pandas as pd

from trader.data.b3_panel import build_panel, snap_factor, split_factors


def _rows(closes, opens=None, ev=None, qty=None, isin="BRXXXXACNOR0", start="2020-01-02"):
    n = len(closes)
    idx = pd.date_range(start, periods=n, freq="B")
    closes = np.asarray(closes, float)
    return pd.DataFrame({"date": idx, "codneg": "XXXX3", "codisi": isin,
                         "open": closes if opens is None else np.asarray(opens, float),
                         "high": closes, "low": closes, "close": closes,
                         "qty": np.full(n, 1000) if qty is None else np.asarray(qty),
                         "volume": closes * 1000, "ev": [""] * n if ev is None else ev})


def test_encaixa_fatores_redondos():
    assert snap_factor(0.5) == 2.0                  # desdobramento 2:1
    assert abs(snap_factor(0.8) - 1.25) < 1e-9     # bonificação 25%
    assert snap_factor(10.0) == 0.1                # grupamento 10:1
    assert snap_factor(0.97) is None or abs(np.log(snap_factor(0.97))) < 0.06


def test_bonificacao_marcada_e_grupamento_sem_marca_com_quantidade():
    closes = [10.0] * 30 + [8.1] * 30 + [81.0] * 30
    ev = [""] * 30 + ["B"] * 30 + ["B"] * 30
    qty = [1000] * 30 + [1250] * 30 + [125] * 30
    g = _rows(closes, ev=ev, qty=qty)
    f = split_factors(g)
    assert abs(f[30] - 1.25) < 1e-9                 # bonificação 25% (preço 10 -> 8,1 ~ -19%)
    assert abs(f[60] - 0.1) < 1e-9                  # grupamento 10:1 sem marca nova, qtd /10
    assert np.count_nonzero(f != 1.0) == 2


def test_grupamento_gigante_e_cisao_usam_o_salto():
    closes = [0.001] * 10 + [250.0] * 10 + [100.0] * 10
    ev = [""] * 10 + ["G"] * 10 + ["C"] * 10            # grupamento 250000:1, depois cisão
    f = split_factors(_rows(closes, ev=ev))
    assert np.isclose(f[10], 0.001 / 250.0)
    assert np.isclose(f[20], 250.0 / 100.0)


def test_provento_implicito_desconta_o_mercado_e_entra_no_retorno_total():
    closes = [100.0] * 10 + [97.0] * 10
    opens = [100.0] * 10 + [97.0] * 10
    ev = [""] * 10 + ["J"] * 10                        # dia ex de JCP no pregão 10
    g = _rows(closes, opens=opens, ev=ev)
    mkt = pd.Series(0.01, index=pd.DatetimeIndex(g["date"]))   # mercado abriu +1%
    cal = pd.DatetimeIndex(g["date"])
    price = build_panel(g, cal, market_gap=mkt)["BRXXXXACNOR0"]
    tr = build_panel(g, cal, market_gap=mkt, total_return=True)["BRXXXXACNOR0"]
    assert np.isclose(price["provento"].iloc[10], 0.04)          # 100·1,01 − 97 = 4
    assert np.isclose(price["close"].pct_change().iloc[10], -0.03)
    assert np.isclose(tr["close"].pct_change().iloc[10], 0.01)   # (97 + 4) / 100 − 1


def test_queda_forte_sem_marca_nem_quantidade_nao_vira_desdobramento():
    closes = [10.0] * 30 + [5.0] * 30                # crash de 50%, volume em ações igual
    g = _rows(closes)
    assert (split_factors(g) == 1.0).all()


def test_painel_ajusta_preco_e_preenche_so_dentro_da_vida():
    closes = [10.0] * 5 + [5.0] * 5
    ev = [""] * 5 + ["X"] * 0 + ["B"] * 5
    g = _rows(closes, ev=ev, qty=[1000] * 5 + [2000] * 5)
    cal = pd.date_range("2019-12-30", periods=16, freq="B")
    p = build_panel(g, cal)["BRXXXXACNOR0"]
    assert pd.Timestamp("2019-12-31") not in p.index             # antes do 1º negócio: fora
    alive = p.dropna(subset=["close"])
    assert np.allclose(alive["close"].pct_change().dropna(), 0.0)   # 10 -> 5 com 2:1 = retorno 0
    assert alive["raw_close"].iloc[-1] == 5.0

import numpy as np
import pandas as pd

from trader.factors.library import FACTORS, combined_score, factor_score


def _frame(px):
    n = len(px)
    idx = pd.date_range("2010-01-04", periods=n, freq="B", tz="UTC")
    idx.name = "time"
    px = np.asarray(px, float)
    return pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": 1e6}, index=idx)


def _universe(n=1400):
    rng = np.random.default_rng(1)
    return {f"S{k}": _frame(50 * np.exp(np.cumsum(rng.normal(0.0002, 0.01 + 0.004 * k, n))))
            for k in range(6)}


def test_factor_score_e_percentil():
    u = _universe()
    for name in FACTORS:
        sc = factor_score(u, name, mkt=u["S0"]["close"])
        assert sc.shape[1] == len(u)
        v = sc.iloc[300:].to_numpy()
        v = v[np.isfinite(v)]
        if name == "div_yield":            # sem parquet de proventos -> tudo NaN, ok
            assert v.size == 0
            continue
        assert v.min() >= 0.0 and v.max() <= 1.0


def test_div_yield_com_proventos():
    u = _universe()
    div = pd.DataFrame({
        "symbol": ["S0", "S0", "S1"],
        "ex_date": pd.to_datetime(["2010-06-01", "2011-06-01", "2011-06-01"], utc=True),
        "amount": [3.0, 3.0, 0.5],
    })
    sc = factor_score(u, "div_yield", dividends=div)
    late = sc.loc["2012-01-01":]
    assert late["S0"].mean() > late["S1"].mean()      # S0 paga mais -> percentil maior


def test_low_vol_favorece_ativo_calmo():
    u = _universe()                     # S0 tem a menor vol, S5 a maior
    sc = factor_score(u, "low_vol")
    assert sc["S0"].iloc[400:].mean() > sc["S5"].iloc[400:].mean()


def test_combined_score_media_dos_percentis():
    u = _universe()
    c = combined_score(u, ["momentum", "low_vol"], mkt=u["S0"]["close"])
    m = (factor_score(u, "momentum") + factor_score(u, "low_vol")) / 2
    pd.testing.assert_frame_equal(c.iloc[400:], m.iloc[400:])


def test_causal():
    u = _universe(1000)
    full = combined_score(u, list(FACTORS), mkt=u["S0"]["close"])
    cut = 700
    trunc = combined_score({k: v.iloc[:cut] for k, v in u.items()},
                           list(FACTORS), mkt=u["S0"]["close"].iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], trunc, check_dtype=False)

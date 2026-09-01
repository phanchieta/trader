import numpy as np
import pandas as pd

from trader.risk.overlay import vol_target_multiplier


def _frames(n=400, seed=0):
    out = {}
    for k in range(3):
        rng = np.random.default_rng(seed + k)
        px = 40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n)))
        idx = pd.date_range("2016-01-04", periods=n, freq="B", tz="UTC")
        idx.name = "time"
        out[f"S{k}"] = pd.DataFrame({"open": px, "high": px, "low": px,
                                     "close": px, "volume": 1.0}, index=idx)
    return out


def test_multiplicador_limitado_e_nao_negativo():
    m = vol_target_multiplier(_frames(), target_ann_vol=0.15, lookback=20,
                              max_leverage=1.0)
    assert (m <= 1.0 + 1e-9).all()
    assert (m >= 0.0).all()


def test_warmup_e_zero():
    m = vol_target_multiplier(_frames(), lookback=20)
    assert (m.iloc[:20] == 0.0).all()


def test_causal_usa_dados_ate_t_menos_1():
    frames = _frames(300)
    full = vol_target_multiplier(frames, lookback=20)
    cut = 200
    trunc = vol_target_multiplier({s: f.iloc[:cut] for s, f in frames.items()},
                                  lookback=20)
    pd.testing.assert_series_equal(full.iloc[:cut], trunc, check_names=False)


def test_vol_maior_reduz_multiplicador():
    calmo = _frames(400, seed=10)
    # dobra a escala dos retornos -> vol ~2x -> multiplicador ~metade
    turbulento = {s: f.assign(close=40.0 * np.exp(
        np.cumsum(np.diff(np.log(f["close"]), prepend=np.log(40.0)) * 2)))
        for s, f in calmo.items()}
    mc = vol_target_multiplier(calmo, lookback=20).iloc[50:].mean()
    mt = vol_target_multiplier(turbulento, lookback=20).iloc[50:].mean()
    assert mt < mc

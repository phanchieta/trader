import pandas as pd
import pytest

from trader.strategy import MeanReversion, Momentum, OpeningRangeBreakout


@pytest.mark.parametrize("strat", [
    MeanReversion(lookback=20, entry_z=1.5, exit_z=0.5),
    MeanReversion(lookback=15, entry_z=1.0, exit_z=0.0, allow_short=True),
    Momentum(lookback=60),
    Momentum(lookback=40, band=0.02, trend_filter=100, allow_short=True),
])
def test_sinal_valido_e_causal(strat, daily_ohlcv):
    df = daily_ohlcv(600)
    full = strat.generate_signals(df)

    assert full.index.equals(df.index)
    assert set(full.unique()) <= {-1.0, 0.0, 1.0}

    cut = 400
    trunc = strat.generate_signals(df.iloc[:cut])
    pd.testing.assert_series_equal(full.iloc[:cut], trunc, check_names=False)


def test_opening_range_exige_intraday(daily_ohlcv):
    with pytest.raises(ValueError, match="intraday"):
        OpeningRangeBreakout().generate_signals(daily_ohlcv(300))


def test_opening_range_roda_em_intraday(intraday_ohlcv):
    df = intraday_ohlcv(days=6, per_day=60)
    sig = OpeningRangeBreakout(opening_minutes=10, session_end="10:55").generate_signals(df)
    assert sig.index.equals(df.index)
    assert set(sig.unique()) <= {-1.0, 0.0, 1.0}
    # durante o range de abertura de cada dia a posição é sempre 0
    local = df.index.tz_convert("America/Sao_Paulo")
    minute = local.hour * 60 + local.minute
    day = local.normalize()
    first_min = pd.Series(minute, index=df.index).groupby(day.values).transform("min")
    in_or = pd.Series(minute, index=df.index) < first_min + 10
    assert (sig[in_or.values] == 0.0).all()

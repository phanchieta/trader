"""Fonte de barras via MT5 — READ-ONLY. Este módulo nunca envia ordem.

Recebe o módulo `mt5` já inicializado (ver core.mt5_session).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .schema import OHLCV, coerce

_TF = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
       "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "D1": "TIMEFRAME_D1"}


def _windows(start: datetime, end: datetime, chunk_days: int):
    """Divide [start, end] em janelas contíguas de até `chunk_days` dias."""
    step = timedelta(days=chunk_days)
    cur = start
    out = []
    while cur < end:
        nxt = min(cur + step, end)
        out.append((cur, nxt))
        cur = nxt
    return out


def bars(mt5, symbol: str, timeframe: str, start: datetime,
         end: datetime | None = None, server_utc_offset_hours: float = -3.0,
         retries: int = 5) -> pd.DataFrame:
    """Barras [start, end] no contrato OHLCV. Faz retry: o terminal baixa
    histórico sob demanda e a 1a chamada às vezes volta vazia / -2."""
    if timeframe not in _TF:
        raise ValueError(f"timeframe {timeframe!r} não suportado: {list(_TF)}")
    end = end or datetime.now(timezone.utc)
    tf = getattr(mt5, _TF[timeframe])
    mt5.symbol_select(symbol, True)

    rates = None
    for _ in range(retries):
        rates = mt5.copy_rates_range(symbol, tf, start, end)
        if rates is not None and len(rates):
            break
        time.sleep(1.0)
    if rates is None or not len(rates):
        raise RuntimeError(
            f"sem barras {timeframe} para {symbol}: {mt5.last_error()} "
            "(abra um gráfico do símbolo no terminal e role para trás)"
        )

    df = pd.DataFrame(rates).rename(columns={"tick_volume": "volume"})
    # 'time' vem em segundos no horário do SERVIDOR -> converte para UTC real
    t = pd.to_datetime(df["time"], unit="s") - pd.Timedelta(hours=server_utc_offset_hours)
    df.index = pd.DatetimeIndex(t).tz_localize("UTC")
    return coerce(df[OHLCV])


def bars_range_chunked(mt5, symbol: str, timeframe: str, start: datetime,
                       end: datetime | None = None, server_utc_offset_hours: float = -3.0,
                       chunk_days: int = 30) -> pd.DataFrame:
    """`bars` em janelas, para driblar o teto de request do terminal ao puxar
    histórico longo de M1/M5. Janelas sem dados (antes do início do histórico)
    são ignoradas."""
    end = end or datetime.now(timezone.utc)
    frames = []
    for w_start, w_end in _windows(start, end, chunk_days):
        try:
            frames.append(bars(mt5, symbol, timeframe, w_start, w_end,
                               server_utc_offset_hours=server_utc_offset_hours,
                               retries=2))
        except RuntimeError:
            continue
    if not frames:
        raise RuntimeError(
            f"sem barras {timeframe} para {symbol} em {start:%Y-%m-%d}..{end:%Y-%m-%d}"
        )
    return coerce(pd.concat(frames)[OHLCV])

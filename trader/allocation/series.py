"""Série diária de retorno total em reais: referência (descontada da diferença medida
contra o ETF) antes do ETF existir, o próprio ETF depois."""
from __future__ import annotations

import pandas as pd


def splice(etf: pd.Series | None, proxy: pd.Series | None, calendar: pd.DatetimeIndex, *,
           drag_aa: float = 0.0, etf_start: str | pd.Timestamp | None = None) -> pd.Series:
    """Índice (começa em 1) no `calendar`. Antes de `etf_start` usa o retorno da
    referência menos `drag_aa`/252 por dia; a partir dele, o do ETF. Dias sem
    cotação repetem o último preço (retorno zero)."""
    parts = []
    start = pd.Timestamp(etf_start) if etf_start is not None else (etf.index[0] if etf is not None else None)
    if proxy is not None:
        rp = proxy.reindex(calendar).ffill().pct_change() - drag_aa / 252.0
        parts.append(rp[calendar < start] if start is not None else rp)
    if etf is not None:
        re = etf.reindex(calendar).ffill().pct_change()
        parts.append(re[calendar >= start])
    r = pd.concat(parts).sort_index()
    r = r[~r.index.duplicated()].reindex(calendar)
    src = proxy if proxy is not None else etf
    first = src.reindex(calendar).first_valid_index()
    r = r.loc[first:].copy()
    r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod().rename("tr")

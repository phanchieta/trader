"""Alocação tática ao vivo: sinal do último mês fechado e ordens em cotas inteiras.

Mesma regra do backtest (`faber_sma` + banda do `run_allocation`): o sinal usa o
fechamento do último pregão do mês e a ordem sai no pregão seguinte.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .signals import faber_sma


def brl(x: float, casas: int = 2) -> str:
    """R$ no padrão brasileiro (1.234,56)."""
    return f"R$ {x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fetch_etf_closes(etfs: list[str], *, start: str = "2019-01-01") -> pd.DataFrame:
    """Fechamentos diários (Yahoo, `<ETF>.SA`), índice de datas sem fuso."""
    from ..data.yahoo import daily

    closes = pd.DataFrame({e: daily(f"{e}.SA", start=start)["close"] for e in etfs})
    closes.index = closes.index.tz_convert(None).normalize()
    return closes.dropna()


@dataclass
class Signal:
    month_end: pd.Timestamp
    weights: dict
    detail: pd.DataFrame     # por ETF: fechamento do mês, média, acima, peso


def last_completed_month_end(index: pd.DatetimeIndex, today: pd.Timestamp) -> pd.Timestamp | None:
    """Último pregão de um mês que já terminou (anterior ao mês de `today`)."""
    first = pd.Timestamp(today.year, today.month, 1)
    past = index[index < first]
    return past.max() if len(past) else None


def faber_signal(closes: pd.DataFrame, month_end: pd.Timestamp, *, months: int = 10) -> Signal:
    """Sinal com os preços até `month_end` (use o último dia disponível para uma prévia
    do mês corrente)."""
    upto = closes.loc[:month_end]
    w = faber_sma(upto, months=months)
    me = upto.reindex(w.index)
    sma = me.rolling(months, min_periods=months).mean()
    detail = pd.DataFrame({"fechamento": me.iloc[-1], "media": sma.iloc[-1],
                           "acima": me.iloc[-1] > sma.iloc[-1], "peso": w.iloc[-1]})
    return Signal(w.index[-1], w.iloc[-1].to_dict(), detail)


def plan_orders(weights: dict, prices: dict, units: dict, cash: float, *,
                band: float = 0.20, lot: int = 1) -> tuple[dict, dict, float]:
    """(cotas-alvo, ordens {etf: +compra/−venda}, patrimônio). Com banda: só mexe no ETF
    se o sinal mudou (entrou/saiu) ou se o peso atual desviou do alvo mais que `band`
    do alvo — igual ao `run_allocation`."""
    equity = cash + sum(float(units.get(a, 0)) * p for a, p in prices.items())
    target, orders = {}, {}
    for a, p in prices.items():
        w_t = float(weights.get(a, 0.0))
        cur = float(units.get(a, 0))
        w_c = cur * p / equity if equity > 0 else 0.0
        flip = (w_t > 0) != (cur > 0)
        dev = abs(w_t - w_c) > (band * w_t if band > 0 else 1e-9)
        tgt = float(np.floor(equity * w_t / p / lot) * lot) if (flip or dev) and p > 0 else cur
        target[a] = tgt
        if tgt != cur:
            orders[a] = tgt - cur
    return target, orders, equity

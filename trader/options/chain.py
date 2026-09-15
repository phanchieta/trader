"""Seleção de séries na cadeia de opções de um dia."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_WEEKLY = re.compile(r"W\d$")


def is_weekly(codneg: pd.Series) -> pd.Series:
    """Séries semanais têm sufixo W1..W5 no código (ex.: PETRJ325W2)."""
    return codneg.str.contains(_WEEKLY)


def monthly_expiries(opts: pd.DataFrame, min_series: int = 5) -> pd.DatetimeIndex:
    """Vencimentos mensais = datas de vencimento das séries não semanais, com pelo
    menos `min_series` séries diferentes negociadas."""
    m = opts[~is_weekly(opts["codneg"]) & opts["expiry"].notna()]
    counts = m.groupby("expiry")["codneg"].nunique()
    return pd.DatetimeIndex(sorted(counts[counts >= min_series].index))


def sane_book(bid: pd.Series, ask: pd.Series, max_rel_spread: float = 0.5) -> pd.Series:
    """Livro utilizável: bid e ask positivos e spread relativo ao meio limitado (o
    COTAHIST tem ofertas-lixo, ex.: ask R$60 numa opção de R$2)."""
    mid = (bid + ask) / 2.0
    return (bid > 0) & (ask >= bid) & ((ask - bid) <= max_rel_spread * mid)


def pick_series(day: pd.DataFrame, *, kind: str, spot: float, moneyness: float,
                expiry: pd.Timestamp, need_bid: bool = True) -> pd.Series | None:
    """Série do `expiry` com strike mais perto de spot·(1+m) (call) ou spot·(1−m) (put).
    `need_bid`: para vender — exige livro são. Senão (para comprar), exige ask
    positivo e, havendo bid, spread até 100%. Empate: mais negócios."""
    tp = "070" if kind == "call" else "080"
    c = day[(day["tpmerc"] == tp) & (day["expiry"] == expiry) & ~is_weekly(day["codneg"])]
    if need_bid:
        c = c[sane_book(c["bid"], c["ask"])]
    else:
        c = c[(c["ask"] > 0) & ((c["bid"] <= 0) | sane_book(c["bid"], c["ask"], 1.0))]
    if c.empty or spot <= 0:
        return None
    target = spot * (1.0 + moneyness if kind == "call" else 1.0 - moneyness)
    dist = (c["strike"] - target).abs()
    best = c.assign(_d=dist.round(4)).sort_values(["_d", "n_trades"], ascending=[True, False])
    row = best.iloc[0]
    return None if abs(row["strike"] / target - 1.0) > 0.05 else row.drop("_d")

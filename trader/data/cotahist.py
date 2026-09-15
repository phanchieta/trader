"""Leitor do COTAHIST (séries históricas da B3), registro tipo 01, largura fixa.

Mercados (TPMERC): 010 à vista, 070 opção de compra, 080 opção de venda.
Preços em centavos com 2 decimais implícitos; divididos por FATCOT (1 ou 1000).
`bid`/`ask` = melhor oferta de compra/venda no fechamento. `strike` e `expiry` só
fazem sentido nas opções; o strike já vem ajustado por proventos na data.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

_FIELDS = {"date": (2, 10), "codbdi": (10, 12), "codneg": (12, 24), "tpmerc": (24, 27),
           "especi": (39, 49), "open": (56, 69), "high": (69, 82), "low": (82, 95),
           "close": (108, 121), "bid": (121, 134), "ask": (134, 147), "n_trades": (147, 152),
           "qty": (152, 170), "volume": (170, 188), "strike": (188, 201), "expiry": (202, 210),
           "fatcot": (210, 217), "codisi": (230, 242)}
_PRICES = ("open", "high", "low", "close", "bid", "ask")


def parse_lines(lines, *, roots: tuple[str, ...] | None,
                markets: tuple[str, ...] = ("010", "070", "080"),
                codbdi: tuple[str, ...] | None = None) -> pd.DataFrame:
    """`roots=None` = todas as raízes; `codbdi` filtra o código BDI (02 = lote padrão)."""
    rows = {k: [] for k in _FIELDS}
    for raw in lines:
        line = raw.decode("latin-1") if isinstance(raw, bytes) else raw
        if not line.startswith("01") or line[24:27] not in markets:
            continue
        if roots is not None and line[12:16] not in roots:
            continue
        if codbdi is not None and line[10:12] not in codbdi:
            continue
        for k, (a, b) in _FIELDS.items():
            rows[k].append(line[a:b])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for k in ("codbdi", "codneg", "tpmerc", "especi", "codisi"):
        df[k] = df[k].str.strip()
    fat = df["fatcot"].astype(np.int64).replace(0, 1)
    for k in _PRICES:
        df[k] = df[k].astype(np.int64) / 100.0 / fat
    df["strike"] = df["strike"].astype(np.int64) / 100.0
    df["volume"] = df["volume"].astype(np.int64) / 100.0
    df["n_trades"] = df["n_trades"].astype(np.int64)
    df["qty"] = df["qty"].astype(np.int64)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    exp = df["expiry"].where(df["expiry"] != "99991231")
    df["expiry"] = pd.to_datetime(exp, format="%Y%m%d", errors="coerce")
    return df.drop(columns="fatcot")


def read_cotahist(path: str | Path, *, roots: tuple[str, ...] | None,
                  markets: tuple[str, ...] = ("010", "070", "080"),
                  codbdi: tuple[str, ...] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.upper().endswith(".TXT") or "COTAHIST" in n.upper())
        with z.open(name) as fh:
            return parse_lines(io.BufferedReader(fh), roots=roots, markets=markets, codbdi=codbdi)

"""Painel diário de TODAS as ações da B3 a partir do COTAHIST (inclusive deslistadas).

Filtra classes de ação (ON, PN, PNA..PNH, UNT) e tickers de ação (final 3–8 ou 11);
fora BDR, ETF, índice. Papel = ISIN (sobrevive a troca de ticker).

Desdobramento / grupamento / bonificação: o COTAHIST dá preço bruto, mas marca o
primeiro dia "ex" na especificação (EB bonificação, EG grupamento, EC…). Nesses
dias o salto de abertura é encaixado no fator redondo mais próximo; saltos acima de
45% sem marca só contam se a quantidade negociada mudar na mesma proporção.
Proventos em dinheiro: o COTAHIST marca o dia ex (D dividendo, J JCP, R restituição)
mas não o valor — `implied_distributions` estima pela queda da abertura menos o gap
noturno do mercado. Com `total_return=True` o painel reinveste essa estimativa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = {"ON", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "PNG", "PNH", "UNT"}
TICKER = r"^[A-Z0-9]{4}(3|4|5|6|7|8|11)$"
_BONUS = [0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5, 0.6, 2 / 3, 0.75, 0.8, 1.0,
          1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 9.0, 19.0, 24.0, 49.0, 99.0]
_REVERSE = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 200, 500, 1000]
# fator = ações novas por ação antiga (preço cai para 1/fator)
CANDIDATES = np.array(sorted({1.0 + b for b in _BONUS} | {1.0 / k for k in _REVERSE}))


def load_rows(b3_dir: str | Path) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p) for p in sorted(Path(b3_dir).glob("acoes_*.parquet"))],
                   ignore_index=True)
    tok = df["especi"].str.split()
    df = df[tok.str[0].isin(CLASSES) & df["codneg"].str.match(TICKER)].copy()
    df["ev"] = df["especi"].str.split().apply(lambda t: "".join(x[1:] for x in t[1:] if x.startswith("E")))
    return df.sort_values(["codisi", "date"]).reset_index(drop=True)


def snap_factor(gap_ratio: float, tol: float = 0.06) -> float | None:
    """Fator (ações novas/antiga) cujo preço teórico 1/fator está a menos de `tol` em
    log do salto observado. None se nenhum encaixa."""
    if not np.isfinite(gap_ratio) or gap_ratio <= 0:
        return None
    err = np.abs(np.log(gap_ratio) + np.log(CANDIDATES))
    k = int(np.argmin(err))
    return float(CANDIDATES[k]) if err[k] < tol else None


def split_factors(g: pd.DataFrame) -> np.ndarray:
    """Fator por linha de um papel (1 = sem evento), no primeiro dia "ex":
      G grupamento   -> fator redondo (tol. 15%) ou o próprio salto
      B bonificação  -> fator redondo (tol. 15%); sem encaixe, o salto se > 25%
      C cisão / R restituição de capital -> a queda inteira como distribuição
    Sem marca: salto > 45% que encaixa num fator ≥ 1,8 e a quantidade confirma."""
    close, open_ = g["close"].to_numpy(), g["open"].to_numpy()
    qty, ev = g["qty"].to_numpy(dtype="float64"), g["ev"].to_numpy()
    f = np.ones(len(g))
    for i in range(1, len(g)):
        if close[i - 1] <= 0 or open_[i] <= 0:
            continue
        ratio = open_[i] / close[i - 1]
        new_ev = ev[i] if ev[i] != ev[i - 1] else ""
        if "G" in new_ev:
            fac = snap_factor(ratio, tol=0.15)
            f[i] = fac if fac is not None else 1.0 / ratio
        elif "B" in new_ev:
            fac = snap_factor(ratio, tol=0.15)
            if fac is not None and abs(np.log(fac)) > 0.03:
                f[i] = fac
            elif abs(np.log(ratio)) > np.log(1.25):
                f[i] = 1.0 / ratio
        elif ("C" in new_ev or "R" in new_ev) and ratio < 1.0:
            if "C" in new_ev or ratio < 0.85:
                f[i] = 1.0 / ratio
        elif not new_ev and abs(np.log(ratio)) > np.log(1.45):
            fac = snap_factor(ratio, tol=0.04)
            if fac is None or abs(np.log(fac)) < np.log(1.8):
                continue
            before = np.median(qty[max(0, i - 20):i]) if i > 0 else np.nan
            after = np.median(qty[i:i + 20])
            if before > 0 and after > 0 and 0.4 < np.log(after / before) / np.log(fac) < 2.0:
                f[i] = fac
    return f


def implied_distributions(g: pd.DataFrame, f: np.ndarray, market_gap: pd.Series | None) -> np.ndarray:
    """Provento em dinheiro estimado, como fração do fechamento anterior, no primeiro
    dia "ex" de dividendo/JCP/restituição (marcas D, J, R): a queda da abertura
    descontado o gap noturno do mercado no mesmo dia. Sem truncar em zero (o ruído
    fica simétrico); limitado a ±20%."""
    close, open_, ev = g["close"].to_numpy(), g["open"].to_numpy(), g["ev"].to_numpy()
    dates = g["date"].to_numpy()
    mg = (market_gap.reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy()
          if market_gap is not None else np.zeros(len(g)))
    d = np.zeros(len(g))
    for i in range(1, len(g)):
        new_ev = ev[i] if ev[i] != ev[i - 1] else ""
        if f[i] != 1.0 or close[i - 1] <= 0 or open_[i] <= 0 or not any(c in new_ev for c in "DJR"):
            continue
        d[i] = float(np.clip((close[i - 1] * (1.0 + mg[i]) - open_[i]) / close[i - 1], -0.2, 0.2))
    return d


def build_panel(rows: pd.DataFrame, calendar: pd.DatetimeIndex, *, fill_limit: int = 5,
                market_gap: pd.Series | None = None, total_return: bool = False) -> dict:
    """{ISIN: DataFrame no calendário do mercado, só entre o primeiro e o último
    negócio} com open/high/low/close ajustados por desdobramento, `raw_close`,
    `volume` (R$, 0 sem negócio), `traded` (bool), `codneg` (ticker do dia) e
    `provento` (fração estimada no dia ex). `total_return=True` reinveste o provento
    estimado no fechamento do dia ex (como o preço ajustado do Yahoo); senão, retorno
    só de preço. Preço repetido em até `fill_limit` pregões sem negócio."""
    out = {}
    for isin, g in rows.groupby("codisi", sort=False):
        g = g.drop_duplicates("date", keep="last").set_index("date")
        gr = g.reset_index()
        f = split_factors(gr)
        dist = implied_distributions(gr, f, market_gap)
        close = g["close"].to_numpy()
        prev = np.r_[np.nan, close[:-1]]
        grow = np.ones(len(g))
        if total_return:
            with np.errstate(invalid="ignore", divide="ignore"):
                grow = np.where(dist != 0.0, 1.0 + dist * prev / close, 1.0)
        step = f * grow
        cum = np.cumprod(step[::-1])[::-1]
        adj = np.r_[cum[1:], 1.0]                          # eventos DEPOIS de cada data
        px = g[["open", "high", "low", "close"]].div(adj, axis=0)
        px["provento"] = dist
        px["raw_close"] = g["close"]
        px["volume"] = g["volume"]
        px["qty_adj"] = g["qty"] * adj
        px["codneg"] = g["codneg"]
        life = calendar[(calendar >= g.index[0]) & (calendar <= g.index[-1])]
        fr = px.reindex(life)
        traded = fr["close"].notna()
        fr[["open", "high", "low", "close", "raw_close"]] = fr[["open", "high", "low", "close", "raw_close"]].ffill(limit=fill_limit)
        fr["volume"] = fr["volume"].fillna(0.0)
        fr["codneg"] = fr["codneg"].ffill()
        fr["traded"] = traded
        out[isin] = fr
    return out

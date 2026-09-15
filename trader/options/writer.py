"""Venda sistemática de opção mensal, com série real do COTAHIST.

kind="call": venda coberta — carteira comprada na ação (retorno total, com
             proventos) e vendida em calls sobre o mesmo número de ações. A call
             exercida é liquidada pela diferença e, a cada rolagem, a carteira volta
             a 100% em ação (paga `stock_fee` sobre o ajuste) — sem caixa negativo.
kind="put":  put coberta por caixa — patrimônio no CDI, puts vendidas até o
             notional (strike × quantidade) igualar o patrimônio.
`hedge`: opcional, compra uma put `hedge` mais fora do dinheiro na mesma data
(trava de baixa: limita a cauda da put vendida).

Todo vencimento: liquida pelo valor intrínseco no fechamento do dia (strike do
dia, já ajustado por proventos), cobra custo de exercício se exercida, e vende a
série do próximo vencimento mensal que esteja entre `min_dte` e `max_dte` dias.
`execution="book"`: vende pelo bid e compra pelo ask; `"mid"`: pelo meio do livro
(otimista). Marca a mercado pelo último negócio, pelo meio de um livro são ou
pelo valor anterior, sempre entre o intrínseco e o teto teórico.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.metrics import vs_benchmark
from ..taxes import after_tax_summary
from .chain import monthly_expiries, pick_series


@dataclass
class WriterResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def _mark(row, last: float, intrinsic: float, cap: float) -> float:
    """Último negócio do dia; senão meio de um livro são; senão valor anterior.
    Sempre entre o intrínseco e o teto teórico (call ≤ spot, put ≤ strike)."""
    v = last
    if row is not None:
        if row["n_trades"] > 0 and row["close"] > 0:
            v = row["close"]
        elif row["bid"] > 0 and row["ask"] >= row["bid"] and (row["ask"] - row["bid"]) <= 0.5 * (row["bid"] + row["ask"]) / 2:
            v = (row["bid"] + row["ask"]) / 2.0
    return float(min(max(v, intrinsic), cap))


def run_option_writer(spot_raw: pd.Series, total_return: pd.Series, opts: pd.DataFrame,
                      cdi: pd.Series, *, kind: str = "call", moneyness: float = 0.05,
                      hedge: float | None = None, fee_option: float = 0.00064,
                      fee_exercise: float = 0.0008, stock_fee: float = 0.0005,
                      execution: str = "book", min_dte: int = 15, max_dte: int = 50,
                      lot: int = 100, equity0: float = 1_000_000.0) -> WriterResult:
    days = spot_raw.index
    rate = cdi.reindex(days.tz_localize("UTC") if days.tz is None else days).ffill().fillna(0.0).to_numpy()
    tr = total_return.reindex(days).fillna(0.0).to_numpy()
    S = spot_raw.to_numpy(dtype="float64")
    by_day = {d: g.set_index("codneg") for d, g in opts.groupby("date")}
    expiries = monthly_expiries(opts)

    cash = float(equity0)
    stock = float(equity0) if kind == "call" else 0.0
    if kind == "call":
        cash = 0.0
    legs: list[dict] = []                      # {'code','kind','side',qty,strike,expiry,value}
    equity = np.empty(len(days))
    log = []

    for i, d in enumerate(days):
        cash *= 1.0 + rate[i]
        if kind == "call":
            stock *= 1.0 + tr[i]
        if i > 0 and S[i - 1] > 0 and S[i] > 0:
            split = S[i - 1] / S[i] * (1.0 + tr[i])
            if abs(split - 1.0) > 0.25:
                for leg in legs:
                    leg["qty"] = round(leg["qty"] * split / lot) * lot
                    leg["strike"] /= split
        today = by_day.get(d)

        for leg in list(legs):
            row = today.loc[leg["code"]] if (today is not None and leg["code"] in today.index) else None
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if row is not None and row["strike"] > 0:
                leg["strike"] = row["strike"]
            intr = (max(S[i] - leg["strike"], 0.0) if leg["kind"] == "call"
                    else max(leg["strike"] - S[i], 0.0))
            if d >= leg["expiry"]:
                cash += leg["side"] * leg["qty"] * intr
                if intr > 0:
                    cash -= leg["qty"] * leg["strike"] * fee_exercise
                key = "pago_no_vencimento" if leg["side"] < 0 else "recebido_protecao"
                log[leg["log"]][key] = intr
                legs.remove(leg)
            else:
                cap = S[i] if leg["kind"] == "call" else leg["strike"]
                leg["value"] = _mark(row, leg["value"], intr, cap)

        if not legs and today is not None and S[i] > 0:
            ahead = expiries[(expiries > d) & ((expiries - d).days >= min_dte)]
            ahead = ahead[(ahead - d).days <= max_dte]
            if len(ahead):
                exp = ahead[0]
                short = pick_series(today.reset_index(), kind=kind, spot=S[i], moneyness=moneyness,
                                    expiry=exp)
                prot = (pick_series(today.reset_index(), kind="put", spot=S[i], moneyness=hedge,
                                    expiry=exp, need_bid=False) if hedge is not None else None)
                if short is not None and (hedge is None or (prot is not None and prot["ask"] > 0)):
                    eq_now = cash + stock
                    if kind == "call":                    # sem alavancagem: volta a 100% em ação
                        stock = eq_now - abs(eq_now - stock) * stock_fee
                        cash = 0.0
                    base = stock if kind == "call" else eq_now
                    qty = np.floor(base / (S[i] if kind == "call" else short["strike"]) / lot) * lot
                    if qty > 0:
                        mid = (short["bid"] + short["ask"]) / 2.0
                        premium = short["bid"] if execution == "book" else mid
                        cash += qty * premium * (1.0 - fee_option)
                        log.append({"data": d, "serie": short["codneg"], "strike": short["strike"],
                                    "vencimento": exp, "spot": S[i], "premio": premium, "qtd": qty})
                        legs.append({"code": short["codneg"], "kind": kind, "side": -1, "qty": qty,
                                     "strike": short["strike"], "expiry": exp, "value": premium,
                                     "log": len(log) - 1})
                        if prot is not None:
                            cost = (prot["ask"] if execution == "book" or prot["bid"] <= 0
                                    else (prot["bid"] + prot["ask"]) / 2.0)
                            cash -= qty * cost * (1.0 + fee_option)
                            log[-1]["protecao"] = cost
                            legs.append({"code": prot["codneg"], "kind": "put", "side": +1, "qty": qty,
                                         "strike": prot["strike"], "expiry": exp, "value": cost,
                                         "log": len(log) - 1})

        equity[i] = cash + stock + sum(leg["side"] * leg["qty"] * leg["value"] for leg in legs)

    idx = days.tz_localize("UTC") if days.tz is None else days
    eq = pd.Series(equity, index=idx, name="equity")
    trades = pd.DataFrame(log)
    b = vs_benchmark(eq, cdi)
    ex = trades["pago_no_vencimento"].fillna(0.0) if "pago_no_vencimento" in trades else pd.Series(dtype=float)
    stats = {"cagr": b["cagr_estrategia"], "alpha": b["alpha_aa"],
             "alpha_pos_ir": after_tax_summary(eq, cdi)["alpha_pos_ir"],
             "sharpe_cdi": b["sharpe_sobre_cdi"],
             "max_dd": round(float((eq / eq.cummax() - 1.0).min()), 4),
             "meses": len(trades),
             "exercidas": round(float((ex > 0).mean()), 3) if len(ex) else None,
             "premio_medio_pct_spot": round(float((trades["premio"] / trades["spot"]).mean()), 4) if len(trades) else None}
    return WriterResult(eq, trades, stats)

"""Teste de ida-e-volta na conta DEMO: compra e fecha 100 PETR4 a mercado.

Rodar da raiz do repo, DURANTE o pregão (10h-17h).
Segurança: aborta se a conta não for demo ou se o AutoTrading estiver desligado.
Vira RUN=True só quando quiser de fato enviar as ordens.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roda sem pip install -e .

import MetaTrader5 as mt5

from trader.core.safety import preflight_trading

SYMBOL, LOT, MAGIC, RUN = "PETR4", 100.0, 990001, False
RET = {10009: "DONE", 10008: "PLACED", 10018: "MARKET_CLOSED",
       10027: "ALGO_DISABLED", 10030: "BAD_FILLING", 10019: "NO_MONEY",
       10014: "BAD_VOLUME", 10021: "NO_PRICES"}

if not mt5.initialize():
    raise SystemExit(mt5.last_error())
preflight_trading(mt5)                     # <- trava demo + AutoTrading
mt5.symbol_select(SYMBOL, True)

acc = mt5.account_info()
tk = mt5.symbol_info_tick(SYMBOL)
print(f"Conta {acc.login} @ {acc.server} (DEMO)  saldo {acc.balance:.2f} {acc.currency}")
print(f"{SYMBOL}  bid={tk.bid} ask={tk.ask}")


def _filling():
    fm = mt5.symbol_info(SYMBOL).filling_mode
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def market(side: str, lot: float):
    q = mt5.symbol_info_tick(SYMBOL)
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": float(lot),
           "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
           "price": q.ask if side == "buy" else q.bid, "deviation": 20,
           "magic": MAGIC, "comment": "teste round-trip",
           "type_time": mt5.ORDER_TIME_DAY, "type_filling": _filling()}
    r = mt5.order_send(req)
    print(f"  {side:4} vol={lot} -> {r.retcode} {RET.get(r.retcode, '?')} "
          f"| {r.comment} | fill={r.price}")
    return r


if RUN:
    print("\n-- round-trip na DEMO --")
    market("buy", LOT)
    time.sleep(2)
    for p in mt5.positions_get(symbol=SYMBOL) or []:
        if p.magic == MAGIC:
            market("sell" if p.type == mt5.POSITION_TYPE_BUY else "buy", p.volume)
    time.sleep(2)
    print("  posições após fechar:", mt5.positions_get(symbol=SYMBOL) or [])
else:
    print("\nRUN=False — nenhuma ordem enviada. Vire para True para testar.")

mt5.shutdown()

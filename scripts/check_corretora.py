"""Diagnóstico da conexão MT5 + símbolo PETR4. Read-only. Rodar da raiz do repo."""
from __future__ import annotations

import datetime as dt

import MetaTrader5 as mt5

SYMBOL = "PETR4"
MODE = {0: "DEMO", 1: "CONTEST", 2: "REAL"}

if not mt5.initialize():
    raise SystemExit(f"initialize() falhou: {mt5.last_error()}")

acc = mt5.account_info()
term = mt5.terminal_info()
print(f"Pacote MT5 .: {mt5.__version__}  | terminal build {term.build}")
print(f"Terminal ...: {term.path}")
print(f"Conta ......: {acc.login} @ {acc.server} -> {MODE.get(acc.trade_mode, '?')}")
print(f"Modo .......: {'netting' if acc.margin_mode == mt5.ACCOUNT_MARGIN_MODE_RETAIL_NETTING else 'hedging'}")
print(f"AutoTrade ..: conta={acc.trade_allowed}  terminal={term.trade_allowed}")

if not mt5.symbol_select(SYMBOL, True):
    print(f"\n{SYMBOL} INDISPONÍVEL:", mt5.last_error())
else:
    s = mt5.symbol_info(SYMBOL)
    modes = {mt5.SYMBOL_TRADE_MODE_DISABLED: "DISABLED",
             mt5.SYMBOL_TRADE_MODE_LONGONLY: "LONG ONLY",
             mt5.SYMBOL_TRADE_MODE_SHORTONLY: "SHORT ONLY",
             mt5.SYMBOL_TRADE_MODE_CLOSEONLY: "CLOSE ONLY",
             mt5.SYMBOL_TRADE_MODE_FULL: "FULL"}
    print(f"\ntrade_mode .....: {modes.get(s.trade_mode)}")
    print(f"volume min/step : {s.volume_min} / {s.volume_step}")
    print(f"tick_size ......: {s.trade_tick_size} | digits: {s.digits}")
    print(f"filling_mask ...: {s.filling_mode}  (bit0=FOK bit1=IOC)")
    print(f"grupo ..........: {s.path}")

    # copy_rates_from_pos devolve -2 se `count` passa do que o terminal tem em
    # cache / do limite "Máx. barras no gráfico". Sonda com contagens decrescentes.
    def depth(tf_name):
        tf = getattr(mt5, f"TIMEFRAME_{tf_name}")
        err = None
        for cnt in (200_000, 50_000, 10_000, 2_000, 500, 100):
            r = mt5.copy_rates_from_pos(SYMBOL, tf, 0, cnt)
            if r is not None and len(r):
                first = dt.datetime.fromtimestamp(r[0]["time"])
                mais = "" if len(r) < cnt else f"  (>= {cnt}, teto de request entre {cnt} e o próximo)"
                return f"{len(r)} barras, desde {first:%Y-%m-%d %H:%M}{mais}"
            err = mt5.last_error()
        return f"vazio {err}"

    for tf_name in ("D1", "M15", "M5", "M1"):
        print(f"{tf_name} ............: {depth(tf_name)}")

mt5.shutdown()

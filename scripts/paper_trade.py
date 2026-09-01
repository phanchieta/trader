"""Executor swing da cesta na conta DEMO. Rodar uma vez por dia, APÓS o fechamento.

    python scripts/paper_trade.py            # dry-run: calcula e mostra, NÃO envia
    python scripts/paper_trade.py --send     # envia as ordens (aborta se não for demo)

Estado em data/paper_state.json (reconciliado com a corretora no boot).
Diário em data/paper_journal.csv. Alertas no Telegram se as env vars estiverem setadas.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.core.dotenv import load_dotenv
from trader.core.mt5_session import mt5_session

load_dotenv()                                   # Telegram etc. do .env
from trader.data.mt5_source import bars
from trader.indicators import yang_zhang
from trader.live import alerts, journal
from trader.live.broker import MT5Broker
from trader.live.planner import plan_targets
from trader.live.state import load as load_state
from trader.live.state import reconcile
from trader.live.state import save as save_state
from trader.strategy import SuperTrend

SEND = "--send" in sys.argv
cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
SYMS = cfg["basket"]["symbols"]
LOT = cfg["backtest"]["lot_size"]
OFFSET = cfg["mt5"]["server_utc_offset_hours"]
TERM = cfg["mt5"]["terminal_path"] or None
DATA = Path(cfg["data"]["dir"])
STATE_PATH = DATA / "paper_state.json"
JOURNAL_PATH = DATA / "paper_journal.csv"

# CAMPEÃO: SuperTrend + vol-target Yang-Zhang. Sem trailing (atrapalha o SuperTrend).
STRAT = SuperTrend(period=10, multiplier=3.0)
TRAIL = None
TARGET_VOL = cfg["risk_overlay"]["target_ann_vol"]
VOL_LB = cfg["risk_overlay"]["vol_lookback"]
MAX_LEV = cfg["risk_overlay"]["max_leverage"]
HIST_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
_alert_lines: list[str] = []

with mt5_session(TERM) as mt5:
    broker = MT5Broker(mt5)                      # preflight: demo + AutoTrading
    acc = mt5.account_info()
    is_demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    cap_per_name = acc.equity / len(SYMS)
    print(f"conta {acc.login} @ {acc.server} ({'DEMO' if is_demo else '!!'}) | "
          f"equity {acc.equity:.2f} {acc.currency} | por ação {cap_per_name:.2f}")

    state = load_state(STATE_PATH)
    broker_pos = broker.positions()
    state, disc = reconcile(state, {s: broker_pos.get(s, 0.0) for s in SYMS})
    for d in disc:
        print(f"[reconcile] {d}")
        _alert_lines.append(f"⚠ reconcile {d}")

    frames = {s: bars(mt5, s, "D1", HIST_START, server_utc_offset_hours=OFFSET)
              for s in SYMS}
    stale = [s for s, f in frames.items()
             if (datetime.now(timezone.utc) - f.index[-1].to_pydatetime()).days > 5]
    if stale:
        print(f"[aviso] histórico velho para: {stale}")
        _alert_lines.append(f"⚠ histórico velho: {stale}")

    # multiplicador de exposição (vol-target Yang-Zhang da cesta, até o último fechamento)
    import pandas as pd
    _yz = pd.DataFrame({s: yang_zhang(frames[s], window=VOL_LB) for s in SYMS}).mean(axis=1)
    overlay = min(MAX_LEV, TARGET_VOL / float(_yz.iloc[-1])) if _yz.iloc[-1] > 0 else 0.0
    print(f"vol-target: vol cesta {float(_yz.iloc[-1]):.1%} a.a. -> multiplicador {overlay:.2f}")

    plan = plan_targets(frames, STRAT, state, capital_per_name=cap_per_name,
                        lot_size=LOT, trailing_stop=TRAIL, overlay_mult=overlay)
    for note in plan.notes:
        print(f"[trailing] {note}")
        _alert_lines.append(f"🛑 {note}")

    print(f"\n{'ação':7} {'atual':>8} {'alvo':>8} {'delta':>8}")
    orders = []
    for s in SYMS:
        cur = state.get(s).shares
        delta = plan.targets[s] - cur
        print(f"{s:7} {cur:>8.0f} {plan.targets[s]:>8.0f} {delta:>+8.0f}")
        if delta != 0.0:
            orders.append((s, delta))

    if not SEND:
        print(f"\ndry-run — {len(orders)} ordem(ns) NÃO enviada(s). Use --send.")
    else:
        for s, delta in orders:
            r = broker.market(s, delta)
            line = (f"{s} {delta:+.0f} -> {getattr(r, 'retcode', None)} "
                    f"{getattr(r, 'comment', '')}")
            print(f"  {line}")
            _alert_lines.append(f"📈 {line}")
            if broker.ok(r):
                ss = state.get(s)
                fill = float(getattr(r, "price", 0.0)) or float(frames[s]["close"].iloc[-1])
                new = ss.shares + delta
                if ss.shares == 0.0 and new != 0.0:
                    ss.entry = ss.peak = fill
                ss.shares = new
                if new == 0.0:
                    ss.entry = ss.peak = 0.0
                    ss.blocked = False

    state.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_state(state, STATE_PATH)

    pos_value = sum(state.get(s).shares * float(frames[s]["close"].iloc[-1]) for s in SYMS)
    n_pos = sum(1 for s in SYMS if state.get(s).shares != 0.0)
    journal.append(JOURNAL_PATH, {
        "ts": state.updated, "equity": round(acc.equity, 2),
        "pos_value": round(pos_value, 2), "n_positions": n_pos,
        "detail": json.dumps({s: state.get(s).shares for s in SYMS}),
    })
    print(f"estado -> {STATE_PATH} | diário -> {JOURNAL_PATH}")

    if _alert_lines:
        head = f"paper-trade {acc.login} {'[SEND]' if SEND else '[dry]'}"
        sent = alerts.notify(head + "\n" + "\n".join(_alert_lines))
        print(f"telegram: {'enviado' if sent else 'não configurado / falhou'}")

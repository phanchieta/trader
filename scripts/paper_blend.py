"""Executor do BLEND (campeão + fatores) na conta DEMO. Rodar 1×/dia após o
fechamento. Sleeve campeão via MT5; sleeve de fatores lê os parquets do Yahoo
(mantenha `pull_factor_universe.py` rodando diariamente).

    python scripts/paper_blend.py            # dry-run
    python scripts/paper_blend.py --send     # envia (aborta se não for demo)

Estado em data/blend_state.json (separado do paper_trade.py do campeão).
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.core.dotenv import load_dotenv
from trader.core.mt5_session import mt5_session

load_dotenv()
from trader.data.mt5_source import bars
from trader.data.store import load
from trader.live import alerts, journal
from trader.live.blend_planner import plan_blend
from trader.live.broker import MT5Broker
from trader.live.state import load as load_state
from trader.live.state import reconcile
from trader.live.state import save as save_state

SEND = "--send" in sys.argv
cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
DATA = Path(cfg["data"]["dir"])
BASKET = cfg["basket"]["symbols"]
UNIVERSE = cfg["factor_universe"]["symbols"]
LOT = cfg["backtest"]["lot_size"]
OFFSET = cfg["mt5"]["server_utc_offset_hours"]
TERM = cfg["mt5"]["terminal_path"] or None
CHAMP_W = cfg["blend"]["champion_weight"]
N_FRAC = cfg["blend"]["factor_quintile"]
TV, VLB, MLEV = (cfg["risk_overlay"]["target_ann_vol"], cfg["risk_overlay"]["vol_lookback"],
                 cfg["risk_overlay"]["max_leverage"])
STATE_PATH = DATA / "blend_state.json"
JOURNAL_PATH = DATA / "blend_journal.csv"
HIST_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
alert_lines: list[str] = []

factor_frames = {}
for s in UNIVERSE:
    p = DATA / f"{s.lower()}_d1.parquet"
    if p.exists():
        factor_frames[s] = load(p)
IBOV = load(DATA / "ibov_d1.parquet")
stale = max((datetime.now(timezone.utc) - f.index[-1].to_pydatetime()).days
            for f in factor_frames.values())
if stale > 5:
    alert_lines.append(f"⚠ parquets de fatores {stale} dias atrasados — rode pull_factor_universe.py")
    print(alert_lines[-1])

with mt5_session(TERM) as mt5:
    broker = MT5Broker(mt5)
    acc = mt5.account_info()
    is_demo = acc.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    print(f"conta {acc.login} @ {acc.server} ({'DEMO' if is_demo else '!!'}) | "
          f"equity {acc.equity:.2f} {acc.currency}")

    champ_frames = {s: bars(mt5, s, "D1", HIST_START, server_utc_offset_hours=OFFSET)
                    for s in BASKET}

    state = load_state(STATE_PATH)
    bpos = broker.positions()
    all_syms = set(champ_frames) | set(factor_frames)
    state, disc = reconcile(state, {s: bpos.get(s, 0.0) for s in all_syms})
    for d in disc:
        print(f"[reconcile] {d}")
        alert_lines.append(f"⚠ {d}")

    plan = plan_blend(champ_frames, factor_frames, IBOV, equity=acc.equity,
                      today=date.today(), prev_quintile=state.factor.get("quintile", []),
                      last_rebalance=state.factor.get("last_rebalance"), champ_w=CHAMP_W,
                      n_frac=N_FRAC, lot=LOT, target_vol=TV, vol_lb=VLB, max_lev=MLEV,
                      min_names=20)
    for n in plan.notes:
        print(f"[fatores] {n}")
        alert_lines.append(f"🔁 {n}")
    print(f"vol-target: campeão {plan.champ_mult:.2f} | fatores {plan.factor_mult:.2f}")

    orders = []
    print(f"\n{'ação':7} {'atual':>8} {'alvo':>8} {'delta':>8}")
    for s in sorted(set(plan.target_shares) | {k for k in state.symbols if state.get(k).shares}):
        cur = state.get(s).shares
        tg = plan.target_shares.get(s, 0.0)
        delta = tg - cur
        if cur or tg:
            print(f"{s:7} {cur:>8.0f} {tg:>8.0f} {delta:>+8.0f}")
        if delta != 0.0:
            orders.append((s, delta))

    if not SEND:
        print(f"\ndry-run — {len(orders)} ordem(ns) NÃO enviada(s). Use --send.")
    else:
        for s, delta in orders:
            r = broker.market(s, delta)
            line = f"{s} {delta:+.0f} -> {getattr(r, 'retcode', None)} {getattr(r, 'comment', '')}"
            print(f"  {line}")
            alert_lines.append(f"📈 {line}")
            if broker.ok(r):
                state.get(s).shares = state.get(s).shares + delta

    if plan.rebalanced:
        state.factor = {"quintile": plan.quintile,
                        "last_rebalance": date.today().isoformat()}
    state.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save_state(state, STATE_PATH)

    pv = sum(state.get(s).shares * float(
        (champ_frames if s in champ_frames else factor_frames)[s]["close"].iloc[-1])
        for s in state.symbols if state.get(s).shares and s in (set(champ_frames) | set(factor_frames)))
    journal.append(JOURNAL_PATH, {
        "ts": state.updated, "equity": round(acc.equity, 2), "pos_value": round(pv, 2),
        "n_positions": sum(1 for s in state.symbols if state.get(s).shares),
        "detail": json.dumps({s: state.get(s).shares for s in state.symbols
                              if state.get(s).shares})})
    print(f"estado -> {STATE_PATH} | diário -> {JOURNAL_PATH}")

    if alert_lines:
        sent = alerts.notify(f"blend {acc.login} {'[SEND]' if SEND else '[dry]'}\n"
                             + "\n".join(alert_lines))
        print(f"telegram: {'enviado' if sent else 'não configurado / falhou'}")

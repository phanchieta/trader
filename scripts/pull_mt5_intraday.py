"""Histórico intraday do MT5 (XP) para mini-índice, mini-dólar e PETR4. Read-only.

Séries contínuas da XP: `$N` = sem ajuste (preço real do contrato vigente, salta
na rolagem); `$D` = ajustada por diferença (movimento em pontos idêntico ao do
contrato, sem salto na rolagem). Resultado em pontos sai da `$D`; notional da `$N`.
Profundidade na XP: M5 desde 2021-09, M1 ~13 meses.

    python scripts/pull_mt5_intraday.py              # M5 + M1
    python scripts/pull_mt5_intraday.py M5
    python scripts/pull_mt5_intraday.py D1 M5 --symbols VALE3,ITUB4,BBDC4,ABEV3
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.core.mt5_session import mt5_session
from trader.data.mt5_source import bars_range_chunked
from trader.data.store import save

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
OFFSET = cfg["mt5"]["server_utc_offset_hours"]
TERM = cfg["mt5"]["terminal_path"] or None
OUT = Path(cfg["data"]["dir"])
args = sys.argv[1:]
SYMBOLS = ["WIN$D", "WIN$N", "WDO$D", "WDO$N", "PETR4"]
if "--symbols" in args:
    k = args.index("--symbols")
    SYMBOLS = args[k + 1].split(",")
    args = args[:k] + args[k + 2:]
TIMEFRAMES = [tf.upper() for tf in args] or ["M5", "M1"]
CHUNK = {"D1": 3650, "M5": 60, "M1": 20}
START = datetime(2021, 1, 1, tzinfo=timezone.utc)


def fname(sym: str, tf: str) -> Path:
    return OUT / f"{sym.lower().replace('$', '_')}_mt5_{tf.lower()}.parquet"


with mt5_session(TERM) as mt5:
    for tf in TIMEFRAMES:
        for sym in SYMBOLS:
            try:
                df = bars_range_chunked(mt5, sym, tf, START, server_utc_offset_hours=OFFSET,
                                        chunk_days=CHUNK.get(tf, 60))
            except (RuntimeError, ValueError) as e:
                print(f"{sym:6} {tf}: {e}")
                continue
            save(df, fname(sym, tf))
            dias = len(set(df.index.tz_convert("America/Sao_Paulo").date))
            print(f"{sym:6} {tf}: {len(df):>7} barras | {dias} pregões | "
                  f"{df.index[0].date()} .. {df.index[-1].date()} -> {fname(sym, tf).name}")

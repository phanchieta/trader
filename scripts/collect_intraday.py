"""Acumula barras intraday da PETR4 ao vivo. Rodar da raiz do repo, DURANTE o pregão.

Deixe rodando; a cada 60s puxa as barras recentes e faz merge no Parquet.
Read-only — não envia ordem. Ctrl+C para parar.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # roda sem pip install -e .

import yaml

from trader.core.mt5_session import mt5_session
from trader.data.collector import collect_once

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
path = str(Path(cfg["data"]["dir"]) / cfg["data"]["intraday_file"])
tf = cfg["data"]["intraday_timeframe"]
offset = cfg["mt5"]["server_utc_offset_hours"]
term_path = cfg["mt5"]["terminal_path"] or None

with mt5_session(term_path) as mt5:
    print(f"coletando {cfg['symbol']} {tf} -> {path}  (Ctrl+C para parar)")
    while True:
        try:
            total = collect_once(mt5, cfg["symbol"], tf, path,
                                 server_utc_offset_hours=offset)
            print(f"{datetime.now():%H:%M:%S}  {total} barras acumuladas")
        except Exception as e:  # noqa: BLE001  — loop resiliente
            print(f"{datetime.now():%H:%M:%S}  aviso: {e}")
        time.sleep(60)

"""Puxa o histórico de barras do MT5 (XP) para Parquet. Read-only — não envia ordem.

Profundidade observada na XP: D1 e M15 ~5 anos (desde 2021); M5/M1 mais raso mas
puxável em janelas. Rodar com o MT5 aberto e logado, da raiz do repo.

    python scripts/pull_mt5_history.py            # D1 + M15
    python scripts/pull_mt5_history.py M15 M5 M1  # timeframes específicos
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
symbol = cfg["symbol"]
offset = cfg["mt5"]["server_utc_offset_hours"]
term_path = cfg["mt5"]["terminal_path"] or None
outdir = Path(cfg["data"]["dir"])

timeframes = [tf.upper() for tf in sys.argv[1:]] or ["D1", "M15"]
CHUNK_DAYS = {"D1": 3650, "H1": 365, "M15": 365, "M5": 90, "M1": 30}
START = datetime(2018, 1, 1, tzinfo=timezone.utc)     # antes do início; janelas vazias são puladas

with mt5_session(term_path) as mt5:
    for tf in timeframes:
        try:
            df = bars_range_chunked(mt5, symbol, tf, START,
                                    server_utc_offset_hours=offset,
                                    chunk_days=CHUNK_DAYS.get(tf, 90))
        except (RuntimeError, ValueError) as e:
            print(f"{tf:4}: {e}")
            continue
        out = outdir / f"petr4_mt5_{tf.lower()}.parquet"
        save(df, out)
        pregoes = len(set(df.index.tz_convert("America/Sao_Paulo").date))
        print(f"{tf:4}: {len(df):>7} barras | {pregoes} pregões | "
              f"{df.index[0].date()} .. {df.index[-1].date()} -> {out}")

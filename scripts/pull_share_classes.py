"""Baixa o diário (Yahoo, ajustado por proventos) das duas classes de cada par em
`share_classes.pairs`. Pula arquivos atualizados há menos de 3 dias. Tickers que
o Yahoo não tem (classes extintas por unificação, deslistadas) são reportados.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
OUT = Path(cfg["data"]["dir"])
SUFFIX = cfg["basket"]["yahoo_suffix"]
syms = sorted({s for pair in cfg["share_classes"]["pairs"] for s in pair})

ok, fail = [], []
for s in syms:
    p = OUT / f"{s.lower()}_d1.parquet"
    if p.exists() and time.time() - p.stat().st_mtime < 3 * 86400:
        ok.append(s)
        continue
    try:
        save(daily(f"{s}{SUFFIX}"), p)
        ok.append(s)
    except Exception:  # noqa: BLE001
        fail.append(s)

print(f"{len(ok)} ok, {len(fail)} sem dados no Yahoo")
if fail:
    print("sem dados:", ", ".join(fail))

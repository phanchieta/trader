"""Baixa o histórico diário do universo grande (Yahoo) para o cross-sectional.
Tickers que falharem são pulados e reportados. Rodar da raiz do repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
syms = cfg["universe"]["symbols"]
suffix = cfg["basket"]["yahoo_suffix"]
outdir = Path(cfg["data"]["dir"])

ok, fail = [], []
for sym in syms:
    try:
        df = daily(f"{sym}{suffix}")
        save(df, outdir / f"{sym.lower()}_d1.parquet")
        print(f"{sym:6} {len(df):>5} barras | {df.index[0].date()} .. {df.index[-1].date()}")
        ok.append(sym)
    except Exception as e:  # noqa: BLE001
        print(f"{sym:6} FALHOU: {e}")
        fail.append(sym)

print(f"\n{len(ok)} ok, {len(fail)} falharam{': ' + ', '.join(fail) if fail else ''}")

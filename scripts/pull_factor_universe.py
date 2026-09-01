"""Baixa o histórico diário do universo de fatores (Yahoo). Tickers que falharem
são pulados. Rodar da raiz do repo. Pode demorar (110+ nomes)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
syms = cfg["factor_universe"]["symbols"]
suffix = cfg["basket"]["yahoo_suffix"]
outdir = Path(cfg["data"]["dir"])

ok, fail = [], []
for i, sym in enumerate(syms, 1):
    p = outdir / f"{sym.lower()}_d1.parquet"
    try:
        df = daily(f"{sym}{suffix}")
        save(df, p)
        ok.append((sym, len(df), df.index[0].year))
    except Exception:  # noqa: BLE001
        fail.append(sym)
    if i % 20 == 0:
        print(f"  ... {i}/{len(syms)}")

print(f"\n{len(ok)} ok, {len(fail)} falharam")
if fail:
    print("falharam:", ", ".join(fail))
longos = [s for s, n, y in ok if y <= 2012]
print(f"com histórico desde <=2012: {len(longos)}")

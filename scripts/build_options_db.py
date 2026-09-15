"""COTAHIST anual (data/b3/COTAHIST_A*.ZIP) -> data/b3/cotahist_<ano>.parquet só com
à vista + opções das raízes de interesse. Pula anos já convertidos (menos o atual)."""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.data.cotahist import read_cotahist

ROOTS = ("PETR", "VALE", "BOVA", "ITUB", "BBDC")
B3 = Path("data/b3")

for z in sorted(B3.glob("COTAHIST_A*.ZIP")):
    year = int(z.stem[-4:])
    out = B3 / f"cotahist_{year}.parquet"
    if out.exists() and year != date.today().year and out.stat().st_mtime > z.stat().st_mtime:
        continue
    t = time.time()
    df = read_cotahist(z, roots=ROOTS)
    df.to_parquet(out, index=False)
    n_opt = int(df["tpmerc"].isin(["070", "080"]).sum())
    print(f"{year}: {len(df):>8} linhas ({n_opt} de opções) em {time.time() - t:.0f}s -> {out.name}", flush=True)

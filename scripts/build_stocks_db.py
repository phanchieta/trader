"""COTAHIST anual -> data/b3/acoes_<ano>.parquet: TODAS as ações do lote padrão
(mercado à vista, BDI 02), inclusive as que depois saíram da bolsa. Base para o
universo sem viés de sobrevivência."""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.data.cotahist import read_cotahist

B3 = Path("data/b3")
for z in sorted(B3.glob("COTAHIST_A*.ZIP")):
    year = int(z.stem[-4:])
    out = B3 / f"acoes_{year}.parquet"
    if out.exists() and year != date.today().year and out.stat().st_mtime > z.stat().st_mtime:
        continue
    t = time.time()
    df = read_cotahist(z, roots=None, markets=("010",), codbdi=("02",))
    df = df.drop(columns=["tpmerc", "codbdi", "bid", "ask", "strike", "expiry"])
    df.to_parquet(out, index=False)
    print(f"{year}: {len(df):>7} linhas, {df['codisi'].nunique()} papéis em {time.time() - t:.0f}s", flush=True)

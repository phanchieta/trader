"""Baixa a série de CDI diário do Banco Central (SGS série 12) -> data/cdi_d1.parquet.
`rate` = taxa diária (fração, ex. 0.00044 = 0,044%/dia). Rodar da raiz do repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

# o SGS rejeita janelas > ~10 anos (HTTP 406) -> puxa em pedaços
_URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sid}/dados"
        "?formato=json&dataInicial={ini}&dataFinal={fim}")
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_CHUNKS = [("01/01/2005", "31/12/2014"), ("01/01/2015", "31/12/2024"),
           ("01/01/2025", "31/12/2032")]
OUT = Path("data/cdi_d1.parquet")

for sid, name in [(12, "CDI"), (11, "Selic")]:
    try:
        recs = []
        for ini, fim in _CHUNKS:
            resp = requests.get(_URL.format(sid=sid, ini=ini, fim=fim),
                                headers=_HEADERS, timeout=60)
            resp.raise_for_status()
            recs.extend(resp.json())
        df = pd.DataFrame(recs)
        idx = pd.to_datetime(df["data"], format="%d/%m/%Y", utc=True)
        s = pd.Series(df["valor"].astype(float).to_numpy() / 100.0, index=idx, name="rate")
        s = s[~s.index.duplicated()].sort_index()
        s.index.name = "time"
        OUT.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(OUT)
        print(f"{name} (SGS {sid}): {len(s)} dias | {s.index[0].date()} .. {s.index[-1].date()}")
        print(f"  taxa diária média {s.mean():.5%} -> ~{(1 + s.mean()) ** 252 - 1:.1%} a.a. -> {OUT}")
        break
    except Exception as e:  # noqa: BLE001
        print(f"SGS {sid} falhou: {e}")
else:
    sys.exit("não consegui baixar CDI nem Selic do BACEN")

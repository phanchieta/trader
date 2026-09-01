"""Retenta os tickers que falharam no pull do universo de fatores + mapeia
mudanças de ticker conhecidas (fusões/renomeações 2021-2025).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from trader.data.store import save
from trader.data.yahoo import daily

cfg = yaml.safe_load(Path("config/settings.yaml").read_text(encoding="utf-8"))
outdir = Path(cfg["data"]["dir"])

RETRY = ["RENN3", "CPLE6", "ELET3", "ELET6", "EMBR3", "JBSS3", "GUAR3", "GOLL4",
         "ALSO3", "ODPV3", "STBP3", "PORT3", "AZUL4", "PETZ3"]
# ticker antigo -> novo (histórico curto pós-fusão, mas melhor que nada)
RENAMED = {"DTEX3": "DXCO3", "ARZZ3": "AZZA3", "SOMA3": "AZZA3",
           "MRFG3": "MBRF3", "BRFS3": "MBRF3", "CCRO3": "MOTV3"}

ok, fail = [], []
for sym in RETRY + list(RENAMED.values()):
    try:
        df = daily(f"{sym}.SA")
        save(df, outdir / f"{sym.lower()}_d1.parquet")
        ok.append(f"{sym}({df.index[0].year})")
    except Exception:  # noqa: BLE001
        fail.append(sym)

print(f"recuperados: {', '.join(ok)}")
print(f"ainda falham: {', '.join(fail)}")
print("\nDeslistados de verdade sem recuperação (viés de sobrevivência inerente): "
      "CIEL3, CRFB3, e possivelmente outros — o Yahoo não guarda deslistados.")

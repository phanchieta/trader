"""Baixa as séries históricas anuais da B3 (COTAHIST) — todos os negócios do dia,
inclusive opções, com melhor oferta de compra e de venda no fechamento.

O servidor da B3 às vezes estrangula uma conexão longa: baixa em paralelo e, se a
velocidade cair demais, reconecta retomando de onde parou (Range).

    python scripts/pull_cotahist.py              # 2008 até o ano atual
    python scripts/pull_cotahist.py 2019 2026    # intervalo
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import requests

OUT = Path("data/b3")
URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{y}.ZIP"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
MIN_BPS, WINDOW_S = 40_000, 20


def download(year: int) -> str:
    final = OUT / f"COTAHIST_A{year}.ZIP"
    part = final.with_suffix(".part")
    total = None
    for attempt in range(30):
        have = part.stat().st_size if part.exists() else 0
        headers = dict(HEADERS) | ({"Range": f"bytes={have}-"} if have else {})
        try:
            with requests.get(URL.format(y=year), headers=headers, timeout=60, stream=True) as r:
                if r.status_code == 404:
                    return f"{year}: não existe"
                if have and r.status_code != 206:
                    have = 0                                       # servidor ignorou o Range
                length = int(r.headers.get("Content-Length", 0))
                total = total or (have + length)
                t0, got0, got = time.time(), 0, 0
                with open(part, "ab" if have else "wb") as fh:
                    for chunk in r.iter_content(1 << 16):
                        fh.write(chunk)
                        got += len(chunk)
                        if time.time() - t0 > WINDOW_S:
                            if (got - got0) / (time.time() - t0) < MIN_BPS:
                                break                              # estrangulada: reconecta
                            t0, got0 = time.time(), got
            if part.stat().st_size >= total:
                part.replace(final)
                return f"{year}: {total / 1e6:.1f} MB ({attempt + 1} conexões)"
        except requests.RequestException:
            time.sleep(2)
    return f"{year}: FALHOU ({part.stat().st_size if part.exists() else 0} de {total} bytes)"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    y0 = int(sys.argv[1]) if len(sys.argv) > 1 else 2008
    y1 = int(sys.argv[2]) if len(sys.argv) > 2 else date.today().year
    todo = [y for y in range(y0, y1 + 1)
            if not (OUT / f"COTAHIST_A{y}.ZIP").exists() or y == date.today().year]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for msg in pool.map(download, todo):
            print(msg, flush=True)

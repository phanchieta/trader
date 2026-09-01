"""Diário do paper-trade: uma linha por execução, em CSV.

Serve para reconstruir a curva de equity real e comparar com o backtest
(scripts/paper_report.py).
"""
from __future__ import annotations

import csv
from pathlib import Path

COLUMNS = ["ts", "equity", "pos_value", "n_positions", "detail"]


def append(path, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def read(path):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, parse_dates=["ts"])

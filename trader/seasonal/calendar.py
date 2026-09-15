"""Posição do pregão dentro do mês, a partir do próprio calendário de negociação.

Índice diário do projeto = meia-noite UTC da data do pregão: usa ano/mês direto,
sem converter fuso (converter para São Paulo jogaria para o dia anterior).
Saber que amanhã é o último pregão do mês usa o calendário de feriados da B3, que
é público com antecedência. O último mês da série pode estar incompleto, então
lá o `from_end` fica indefinido.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def month_offsets(index: pd.DatetimeIndex) -> pd.DataFrame:
    """from_start: 1 = primeiro pregão do mês; from_end: −1 = último pregão do mês."""
    ym = pd.Series(np.asarray(index.year * 100 + index.month), index=index)
    g = ym.groupby(ym.to_numpy())
    out = pd.DataFrame({"from_start": g.cumcount() + 1,
                        "from_end": -(g.cumcount(ascending=False) + 1)}, index=index).astype(float)
    out.loc[ym == ym.iloc[-1], "from_end"] = np.nan
    return out


def tom_flag(offsets: pd.DataFrame, *, start: int = -1, end: int = 3) -> pd.Series:
    """True nos pregões da janela da virada: do `start`-ésimo antes do fim do mês
    (−1 = último) até o `end`-ésimo do mês seguinte."""
    return ((offsets["from_end"] >= start) | (offsets["from_start"] <= end)).rename("tom")

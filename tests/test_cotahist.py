from trader.data.cotahist import parse_lines


def _record(**kw):
    f = {"tipreg": "01", "date": "20240315", "codbdi": "02", "codneg": "PETR4", "tpmerc": "010",
         "nomres": "PETROBRAS", "especi": "PN", "prazot": "", "modref": "R$",
         "open": 3850, "high": 3900, "low": 3800, "avg": 3860, "close": 3880, "bid": 3879,
         "ask": 3881, "n_trades": 123, "qty": 456000, "volume": 1768800000, "strike": 0,
         "indopc": "0", "expiry": "99991231", "fatcot": 1, "ptoexe": 0, "codisi": "BRPETRACNPR6",
         "dismes": "123"}
    f.update(kw)
    num = lambda v, n: str(int(v)).zfill(n)
    line = (f["tipreg"] + f["date"] + f["codbdi"] + f["codneg"].ljust(12) + f["tpmerc"]
            + f["nomres"].ljust(12) + f["especi"].ljust(10) + f["prazot"].ljust(3) + f["modref"].ljust(4)
            + "".join(num(f[k], 13) for k in ("open", "high", "low", "avg", "close", "bid", "ask"))
            + num(f["n_trades"], 5) + num(f["qty"], 18) + num(f["volume"], 18) + num(f["strike"], 13)
            + f["indopc"] + f["expiry"] + num(f["fatcot"], 7) + num(f["ptoexe"], 13)
            + f["codisi"].ljust(12) + f["dismes"].ljust(3))
    assert len(line) == 245
    return line


def test_le_acao_e_opcao_e_filtra_raiz_e_mercado():
    lines = [
        "00COTAHIST.2024BOVESPA 20240315" + " " * 214,
        _record(),
        _record(codneg="PETRC400", tpmerc="070", codbdi="78", open=150, high=160, low=140,
                close=155, bid=154, ask=156, strike=4000, expiry="20240315"),
        _record(codneg="PETRO360", tpmerc="080", codbdi="82", close=45, bid=44, ask=46,
                strike=3600, expiry="20240315"),
        _record(codneg="VALE3", especi="ON"),
        _record(codneg="PETR4", tpmerc="020"),                    # fracionário: fora
    ]
    df = parse_lines(lines, roots=("PETR",))
    assert list(df["codneg"]) == ["PETR4", "PETRC400", "PETRO360"]
    stock = df.iloc[0]
    assert stock["close"] == 38.80 and stock["bid"] == 38.79 and stock["qty"] == 456000
    assert str(stock["expiry"]) == "NaT"
    call, put = df.iloc[1], df.iloc[2]
    assert call["tpmerc"] == "070" and call["strike"] == 40.0 and call["close"] == 1.55
    assert put["tpmerc"] == "080" and put["ask"] == 0.46 and str(put["expiry"].date()) == "2024-03-15"
    assert stock["codisi"] == "BRPETRACNPR6"
    allstocks = parse_lines(lines, roots=None, markets=("010",), codbdi=("02",))
    assert list(allstocks["codneg"]) == ["PETR4", "VALE3"]

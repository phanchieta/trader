import trader.live.alerts as alerts
from trader.live.journal import append, read


def test_alerts_noop_sem_credenciais(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert alerts.enabled() is False
    assert alerts.notify("oi") is False          # não levanta, não envia


def test_journal_cria_cabecalho_e_acumula(tmp_path):
    p = tmp_path / "j.csv"
    append(p, {"ts": "2026-08-31T20:00:00", "equity": 100000.0,
               "pos_value": 0.0, "n_positions": 0, "detail": "{}"})
    append(p, {"ts": "2026-09-01T20:00:00", "equity": 100420.0,
               "pos_value": 98000.0, "n_positions": 2, "detail": "{}"})
    df = read(p)
    assert list(df.columns) == ["ts", "equity", "pos_value", "n_positions", "detail"]
    assert len(df) == 2
    assert df["equity"].iloc[-1] == 100420.0


def test_journal_read_inexistente_vazio(tmp_path):
    assert read(tmp_path / "nada.csv").empty

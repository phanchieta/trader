"""Testa o alerta do Telegram. Preencha o .env primeiro, depois rode isto."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trader.core.dotenv import load_dotenv
from trader.live import alerts

loaded = load_dotenv()
print(f".env carregado: {sorted(loaded)}")

if not alerts.enabled():
    sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID ausentes ou vazios no .env")

ok = alerts.notify(f"✅ teste do paper-trade — {datetime.now():%d/%m %H:%M}")
print("mensagem enviada!" if ok else
      "falhou — confira token/chat_id no .env e a conexão")

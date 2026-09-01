"""Alertas via Telegram. No-op se TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID não
estiverem no ambiente — assim o dry-run roda sem configurar nada.

Configurar (PowerShell):
    $env:TELEGRAM_BOT_TOKEN = "123:ABC..."
    $env:TELEGRAM_CHAT_ID   = "987654321"
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request


def _creds() -> tuple[str | None, str | None]:
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def enabled() -> bool:
    tok, chat = _creds()
    return bool(tok and chat)


def notify(text: str, *, timeout: float = 8.0) -> bool:
    tok, chat = _creds()
    if not (tok and chat):
        return False
    url = f"https://api.telegram.org/bot{tok}/sendMessage"
    body = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=body, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 — alerta nunca derruba o executor
        return False

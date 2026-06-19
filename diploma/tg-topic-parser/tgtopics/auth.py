"""Telegram client construction and interactive login (session reuse)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from telethon import TelegramClient

from tgtopics.config import (
    SESSION_DIR,
    SESSION_NAME,
    ensure_dirs,
    get_telegram_creds,
)

log = logging.getLogger("tgtopics.auth")


def build_client() -> TelegramClient:
    ensure_dirs()
    creds = get_telegram_creds()
    session_path = str(SESSION_DIR / SESSION_NAME)
    client = TelegramClient(session_path, creds.api_id, creds.api_hash)
    # Auto-sleep on short flood waits; bigger ones are caught explicitly in parser.
    client.flood_sleep_threshold = 60
    return client


@asynccontextmanager
async def connected_client():
    """Yield a connected client. Prompts for code + 2FA on the first run only."""
    creds = get_telegram_creds()
    client = build_client()
    # client.start() handles the interactive login code and 2FA password prompts
    # via input()/getpass; subsequent runs reuse the saved session silently.
    await client.start(phone=creds.phone)
    log.info("Telegram client connected")
    try:
        yield client
    finally:
        await client.disconnect()

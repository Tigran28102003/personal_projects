"""Configuration: paths, .env credentials, YAML config, defaults, logging."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Project root = the directory that contains the `tgtopics` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DB_PATH = PROJECT_ROOT / "data.db"
SESSION_DIR = PROJECT_ROOT / "session"
MEDIA_DIR = PROJECT_ROOT / "media"
EXPORTS_DIR = PROJECT_ROOT / "exports"
CHANNELS_FILE = PROJECT_ROOT / "channels.yaml"
TOPICS_FILE = PROJECT_ROOT / "topics.yaml"
SESSION_NAME = "tg_topic_parser"

# Defaults
DEFAULT_SINCE = "2019-01-01"
DEFAULT_BATCH_SIZE = 20
DEFAULT_CONCURRENCY = 4
CLASSIFIER_MODEL = "claude-haiku-4-5"

# Haiku 4.5 pricing (USD per 1M tokens) — drives the cost estimator.
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00

# Message Batches API hard limit (requests per batch).
BATCH_MAX_REQUESTS = 100_000

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class TelegramCreds:
    api_id: int
    api_hash: str
    phone: str


@dataclass
class Topic:
    id: str
    description: str


def get_telegram_creds() -> TelegramCreds:
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    phone = os.getenv("TELEGRAM_PHONE")
    missing = [
        name
        for name, val in (
            ("TELEGRAM_API_ID", api_id),
            ("TELEGRAM_API_HASH", api_hash),
            ("TELEGRAM_PHONE", phone),
        )
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing Telegram credentials in .env: {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in (see README)."
        )
    return TelegramCreds(api_id=int(api_id), api_hash=api_hash, phone=phone)


def get_anthropic_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in .env (see README).")
    return key


def normalize_channel(name: str) -> str:
    """Normalize a channel reference to '@username' (or a numeric id string)."""
    name = str(name).strip()
    if name.startswith("https://t.me/") or name.startswith("t.me/"):
        name = name.rstrip("/").rsplit("/", 1)[-1]
    if name.startswith("@"):
        return name
    if name.lstrip("-").isdigit():
        return name
    return "@" + name


def load_channels(override: list[str] | None = None) -> list[str]:
    if override:
        return [normalize_channel(c) for c in override]
    if not CHANNELS_FILE.exists():
        raise RuntimeError(f"channels.yaml not found at {CHANNELS_FILE}")
    data = yaml.safe_load(CHANNELS_FILE.read_text(encoding="utf-8")) or {}
    channels = data.get("channels") or []
    if not channels:
        raise RuntimeError("No channels listed in channels.yaml")
    return [normalize_channel(c) for c in channels]


def load_topics() -> list[Topic]:
    if not TOPICS_FILE.exists():
        raise RuntimeError(f"topics.yaml not found at {TOPICS_FILE}")
    data = yaml.safe_load(TOPICS_FILE.read_text(encoding="utf-8")) or {}
    raw = data.get("topics") or []
    topics: list[Topic] = []
    for item in raw:
        if isinstance(item, str):
            topics.append(Topic(id=item, description=""))
        else:
            topics.append(Topic(id=item["id"], description=item.get("description", "")))
    if not topics:
        raise RuntimeError("No topics listed in topics.yaml")
    return topics


def ensure_dirs() -> None:
    for d in (SESSION_DIR, MEDIA_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not verbose:
        for noisy in ("telethon", "httpx", "anthropic"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

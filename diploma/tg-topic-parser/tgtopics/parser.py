"""Stage 1 — parsing: count (dry-run) + resumable backfill / incremental sync."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    UsernameNotOccupiedError,
)
from telethon.tl import types
from telethon.tl.functions.channels import GetFullChannelRequest

from tgtopics import storage
from tgtopics.config import MEDIA_DIR

log = logging.getLogger("tgtopics.parser")

COMMIT_EVERY = 200
THROTTLE_BETWEEN_CHANNELS = 3.0


def parse_since(value: str) -> datetime:
    """Parse 'YYYY-MM-DD' into a timezone-aware UTC datetime (never naive)."""
    dt = datetime.strptime(value, "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def _iso(dt) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


def extract_reactions(message) -> str | None:
    """Return {emoji: count} JSON. Custom emoji are stored as 'custom:<doc_id>'."""
    reactions = getattr(message, "reactions", None)
    if not reactions or not getattr(reactions, "results", None):
        return None
    out: dict[str, int] = {}
    for r in reactions.results:
        reaction = r.reaction
        emoticon = getattr(reaction, "emoticon", None)
        if emoticon is not None:
            key = emoticon
        else:
            doc_id = getattr(reaction, "document_id", None)
            key = f"custom:{doc_id}" if doc_id is not None else "custom:unknown"
        out[key] = out.get(key, 0) + r.count
    return json.dumps(out, ensure_ascii=False) if out else None


def extract_media(message) -> dict:
    media = message.media
    if media is None:
        return {"has_media": 0, "media_type": None, "media_size": None, "media_mime": None}
    media_type = type(media).__name__.replace("MessageMedia", "").lower()
    size = None
    mime = None
    if isinstance(media, types.MessageMediaDocument) and media.document:
        media_type = "document"
        size = getattr(media.document, "size", None)
        mime = getattr(media.document, "mime_type", None)
    elif isinstance(media, types.MessageMediaPhoto):
        media_type = "photo"
    return {"has_media": 1, "media_type": media_type, "media_size": size, "media_mime": mime}


def message_to_row(channel_id: int, message, media_path: str | None = None) -> dict:
    media = extract_media(message)
    reply_to = None
    if message.reply_to is not None:
        reply_to = getattr(message.reply_to, "reply_to_msg_id", None)
    return {
        "channel_id": channel_id,
        "message_id": message.id,
        "date": _iso(message.date),
        "text": message.raw_text or "",
        "views": getattr(message, "views", None),
        "forwards": getattr(message, "forwards", None),
        "reactions": extract_reactions(message),
        "reply_to": reply_to,
        "grouped_id": getattr(message, "grouped_id", None),
        "edit_date": _iso(getattr(message, "edit_date", None)),
        "has_media": media["has_media"],
        "media_type": media["media_type"],
        "media_size": media["media_size"],
        "media_mime": media["media_mime"],
        "media_path": media_path,
    }


async def _get_full(client: TelegramClient, entity):
    """Fetch participants_count + about; these are NOT on the basic entity."""
    participants = None
    description = None
    try:
        full = await client(GetFullChannelRequest(entity))
        participants = getattr(full.full_chat, "participants_count", None)
        description = getattr(full.full_chat, "about", None)
    except Exception as e:  # not a channel, or no access to full info
        log.debug("GetFullChannel failed for %s: %s", getattr(entity, "username", entity.id), e)
    return participants, description


async def save_channel_metadata(client: TelegramClient, conn, entity) -> None:
    participants, description = await _get_full(client, entity)
    storage.upsert_channel(
        conn,
        entity.id,
        getattr(entity, "username", None),
        getattr(entity, "title", None),
        description,
        participants,
    )


# --- count (dry-run) -------------------------------------------------------

async def count_channels(client: TelegramClient, channels: list[str]) -> list[dict]:
    results: list[dict] = []
    for username in channels:
        try:
            entity = await client.get_entity(username)
        except (ChannelPrivateError, UsernameNotOccupiedError, ValueError) as e:
            log.warning("Skipping %s: %s", username, e)
            results.append({"username": username, "error": str(e)})
            continue
        total = (await client.get_messages(entity, limit=0)).total
        newest = await client.get_messages(entity, limit=1)
        oldest = await client.get_messages(entity, limit=1, reverse=True)
        participants, _ = await _get_full(client, entity)
        results.append({
            "username": username,
            "title": getattr(entity, "title", username),
            "total": total,
            "participants": participants,
            "oldest": oldest[0].date if oldest else None,
            "newest": newest[0].date if newest else None,
        })
    return results


# --- backfill / incremental ------------------------------------------------

async def _ingest(client, conn, entity, username, channel_id, base_kwargs,
                  since, download_media, limit, mode, start_total):
    buffer: list[dict] = []
    count = 0
    max_id = 0
    cursor_id = base_kwargs.get("offset_id")
    saved_total = start_total or 0

    def flush():
        nonlocal saved_total
        if not buffer:
            return
        storage.upsert_messages(conn, buffer)
        saved_total += len(buffer)
        fields = {
            "since_date": since.isoformat(),
            "total_saved": saved_total,
            "last_message_id": max_id,
        }
        if mode == "backfill":
            fields["backfill_cursor_id"] = cursor_id
        storage.upsert_progress(conn, channel_id, **fields)
        conn.commit()
        buffer.clear()

    cur_kwargs = dict(base_kwargs)
    while True:
        hit_limit = False
        try:
            async for message in client.iter_messages(entity, **cur_kwargs):
                mid = message.id
                if message.date and message.date < since:
                    continue
                cursor_id = mid
                if mid > max_id:
                    max_id = mid
                if not isinstance(message, types.Message):
                    continue  # skip service messages (joins, pins, ...)
                media_path = None
                if download_media and message.media:
                    try:
                        dest = MEDIA_DIR / username.lstrip("@")
                        dest.mkdir(parents=True, exist_ok=True)
                        media_path = await message.download_media(file=str(dest) + "/")
                    except Exception as e:
                        log.debug("media download failed %s/%s: %s", username, mid, e)
                buffer.append(message_to_row(channel_id, message, media_path))
                count += 1
                if len(buffer) >= COMMIT_EVERY:
                    flush()
                    log.info("[%s] %s: %d saved (at %s)", username, mode, count,
                             message.date.date() if message.date else "?")
                if limit and count >= limit:
                    hit_limit = True
                    break
        except FloodWaitError as e:
            flush()
            log.warning("[%s] FloodWait %ss — sleeping then resuming", username, e.seconds)
            await asyncio.sleep(e.seconds + 5)
            # Restart the iterator from where we stopped (offset_date is finicky on resume).
            cur_kwargs = {k: v for k, v in base_kwargs.items() if k != "offset_date"}
            if cursor_id is not None:
                cur_kwargs["offset_id"] = cursor_id
            continue
        flush()
        return count, max_id, not hit_limit


async def backfill_channel(client, conn, username, since, download_media, limit) -> None:
    try:
        entity = await client.get_entity(username)
    except (ChannelPrivateError, UsernameNotOccupiedError, ValueError) as e:
        log.warning("Skipping %s: %s", username, e)
        return
    channel_id = entity.id
    await save_channel_metadata(client, conn, entity)

    progress = storage.get_progress(conn, channel_id)
    start_total = progress["total_saved"] if progress else 0
    existing_last = (progress["last_message_id"] if progress else 0) or 0
    complete = (progress["backfill_complete"] if progress else 0) or 0

    if complete:
        log.info("[%s] incremental sync (min_id=%s)", username, existing_last)
        base = {"reverse": True, "min_id": existing_last}
        count, max_id, _ = await _ingest(
            client, conn, entity, username, channel_id, base,
            since, download_media, limit, "incremental", start_total,
        )
        if max_id > existing_last:
            storage.upsert_progress(conn, channel_id, last_message_id=max_id)
        log.info("[%s] incremental done: +%d new", username, count)
        return

    cursor = (progress["backfill_cursor_id"] if progress else None)
    if cursor:
        log.info("[%s] resuming backfill from id %s", username, cursor)
        base = {"reverse": True, "offset_id": cursor}
    else:
        log.info("[%s] starting backfill from %s", username, since.date())
        base = {"reverse": True, "offset_date": since}

    count, max_id, exhausted = await _ingest(
        client, conn, entity, username, channel_id, base,
        since, download_media, limit, "backfill", start_total,
    )
    if exhausted:
        new_last = max(existing_last, max_id)
        storage.upsert_progress(conn, channel_id, backfill_complete=1, last_message_id=new_last)
        log.info("[%s] backfill complete: +%d (last_id=%s)", username, count, new_last)
    else:
        log.info("[%s] backfill paused at --limit: +%d (resume by re-running)", username, count)


async def run_parse(client, conn, channels, since, download_media, limit) -> None:
    for username in channels:
        await backfill_channel(client, conn, username, since, download_media, limit)
        await asyncio.sleep(THROTTLE_BETWEEN_CHANNELS)

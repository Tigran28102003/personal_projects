"""SQLite storage: schema, idempotent upserts, progress, batch-job tracking."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tgtopics.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    channel_id INTEGER PRIMARY KEY,
    username TEXT,
    title TEXT,
    description TEXT,
    participants_count INTEGER,
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    date TEXT,
    text TEXT,
    views INTEGER,
    forwards INTEGER,
    reactions TEXT,
    reply_to INTEGER,
    grouped_id INTEGER,
    edit_date TEXT,
    has_media INTEGER,
    media_type TEXT,
    media_size INTEGER,
    media_mime TEXT,
    media_path TEXT,
    PRIMARY KEY (channel_id, message_id)
);
CREATE TABLE IF NOT EXISTS message_topics (
    channel_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    primary_topic TEXT,
    topics TEXT,
    confidence REAL,
    tickers TEXT,
    summary TEXT,
    model TEXT,
    classified_at TEXT,
    PRIMARY KEY (channel_id, message_id)
);
CREATE TABLE IF NOT EXISTS parse_progress (
    channel_id INTEGER PRIMARY KEY,
    backfill_cursor_id INTEGER,
    backfill_complete INTEGER DEFAULT 0,
    last_message_id INTEGER,
    since_date TEXT,
    total_saved INTEGER DEFAULT 0,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS batch_jobs (
    batch_id TEXT PRIMARY KEY,
    created_at TEXT,
    status TEXT,
    n_requests INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_channel_date ON messages(channel_id, date);
"""

MESSAGE_COLUMNS = (
    "channel_id", "message_id", "date", "text", "views", "forwards", "reactions",
    "reply_to", "grouped_id", "edit_date", "has_media", "media_type",
    "media_size", "media_mime", "media_path",
)

PROGRESS_COLUMNS = (
    "channel_id", "backfill_cursor_id", "backfill_complete",
    "last_message_id", "since_date", "total_saved", "updated_at",
)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- channels --------------------------------------------------------------

def upsert_channel(conn, channel_id, username, title, description, participants_count) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO channels
           (channel_id, username, title, description, participants_count, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (channel_id, username, title, description, participants_count, _now()),
    )
    conn.commit()


def list_channels(conn, channel_ids: list[int] | None = None):
    sql = "SELECT * FROM channels"
    params: list = []
    if channel_ids:
        sql += f" WHERE channel_id IN ({', '.join('?' for _ in channel_ids)})"
        params = list(channel_ids)
    sql += " ORDER BY username"
    return conn.execute(sql, params).fetchall()


# --- messages --------------------------------------------------------------

def upsert_messages(conn, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    cols = ", ".join(MESSAGE_COLUMNS)
    placeholders = ", ".join("?" for _ in MESSAGE_COLUMNS)
    conn.executemany(
        f"INSERT OR REPLACE INTO messages ({cols}) VALUES ({placeholders})",
        [tuple(r.get(c) for c in MESSAGE_COLUMNS) for r in rows],
    )


def message_exists(conn, channel_id, message_id) -> bool:
    return conn.execute(
        "SELECT 1 FROM messages WHERE channel_id=? AND message_id=?",
        (channel_id, message_id),
    ).fetchone() is not None


# --- parse progress --------------------------------------------------------

def get_progress(conn, channel_id):
    return conn.execute(
        "SELECT * FROM parse_progress WHERE channel_id = ?", (channel_id,)
    ).fetchone()


def upsert_progress(conn, channel_id, **fields) -> None:
    existing = get_progress(conn, channel_id)
    data = dict(existing) if existing else {"channel_id": channel_id, "total_saved": 0}
    data.update(fields)
    data["channel_id"] = channel_id
    data["updated_at"] = _now()
    placeholders = ", ".join("?" for _ in PROGRESS_COLUMNS)
    conn.execute(
        f"INSERT OR REPLACE INTO parse_progress ({', '.join(PROGRESS_COLUMNS)}) "
        f"VALUES ({placeholders})",
        tuple(data.get(c) for c in PROGRESS_COLUMNS),
    )
    conn.commit()


# --- topics ----------------------------------------------------------------

def upsert_topic(conn, channel_id, message_id, primary_topic, topics,
                 confidence, tickers, summary, model) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO message_topics
           (channel_id, message_id, primary_topic, topics, confidence,
            tickers, summary, model, classified_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            channel_id, message_id, primary_topic,
            json.dumps(topics, ensure_ascii=False), confidence,
            json.dumps(tickers, ensure_ascii=False), summary, model, _now(),
        ),
    )


def is_classified(conn, channel_id, message_id) -> bool:
    return conn.execute(
        "SELECT 1 FROM message_topics WHERE channel_id=? AND message_id=?",
        (channel_id, message_id),
    ).fetchone() is not None


def _unclassified_where(channel_ids, reclassify):
    """Build the WHERE clause + params shared by iter/count of unclassified."""
    clauses, params = [], []
    if not reclassify:
        clauses.append("t.message_id IS NULL")
    if channel_ids:
        clauses.append(f"m.channel_id IN ({', '.join('?' for _ in channel_ids)})")
        params.extend(channel_ids)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def iter_unclassified(conn, channel_ids=None, limit=None, reclassify=False):
    join = "" if reclassify else (
        " LEFT JOIN message_topics t "
        "ON t.channel_id = m.channel_id AND t.message_id = m.message_id"
    )
    where, params = _unclassified_where(channel_ids, reclassify)
    sql = f"SELECT m.channel_id, m.message_id, m.text FROM messages m{join}{where} " \
          f"ORDER BY m.channel_id, m.message_id"
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


# --- batch jobs ------------------------------------------------------------

def add_batch_job(conn, batch_id, n_requests) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO batch_jobs (batch_id, created_at, status, n_requests) "
        "VALUES (?, ?, ?, ?)",
        (batch_id, _now(), "in_progress", n_requests),
    )
    conn.commit()


def update_batch_job(conn, batch_id, status) -> None:
    conn.execute("UPDATE batch_jobs SET status=? WHERE batch_id=?", (status, batch_id))
    conn.commit()


def open_batch_jobs(conn):
    return conn.execute("SELECT * FROM batch_jobs WHERE status != 'done'").fetchall()


# --- export ----------------------------------------------------------------

def export_rows(conn, channel_ids=None):
    sql = """
        SELECT m.channel_id, c.username, m.message_id, m.date, m.text,
               m.views, m.forwards, m.reactions,
               t.primary_topic, t.topics, t.tickers, t.confidence, t.summary, t.model
        FROM messages m
        LEFT JOIN channels c ON c.channel_id = m.channel_id
        LEFT JOIN message_topics t
            ON t.channel_id = m.channel_id AND t.message_id = m.message_id
    """
    params: list = []
    if channel_ids:
        sql += f" WHERE m.channel_id IN ({', '.join('?' for _ in channel_ids)})"
        params = list(channel_ids)
    sql += " ORDER BY m.channel_id, m.message_id"
    return conn.execute(sql, params).fetchall()

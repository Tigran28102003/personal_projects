"""Output & analytics: export (json/csv) and per-channel report."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from tgtopics import storage
from tgtopics.config import EXPORTS_DIR, ensure_dirs

log = logging.getLogger("tgtopics.report")
console = Console()


def resolve_channel_ids(conn, channels) -> list[int] | None:
    """Map CLI channel handles to stored channel_ids; None means all channels."""
    if not channels:
        return None
    ids: list[int] = []
    for c in channels:
        handle = str(c).lstrip("@").lower()
        row = conn.execute(
            "SELECT channel_id FROM channels WHERE lower(username) = ?", (handle,)
        ).fetchone()
        if row:
            ids.append(row["channel_id"])
        else:
            console.print(f"[yellow]Channel {c} not found in DB (not parsed yet).[/]")
    return ids or None


# --- export ----------------------------------------------------------------

def export(conn, fmt: str, channels, out: str | None) -> None:
    ensure_dirs()
    channel_ids = resolve_channel_ids(conn, channels)
    rows = storage.export_rows(conn, channel_ids)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out) if out else EXPORTS_DIR / f"export_{ts}.{fmt}"

    if fmt == "json":
        data = []
        for r in rows:
            d = dict(r)
            d["topics"] = json.loads(d["topics"]) if d.get("topics") else []
            d["tickers"] = json.loads(d["tickers"]) if d.get("tickers") else []
            data.append(d)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:  # csv — arrays kept as JSON strings
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if rows:
                keys = list(rows[0].keys())
                writer.writerow(keys)
                for r in rows:
                    writer.writerow([r[k] for k in keys])
    console.print(f"[green]Exported {len(rows)} rows -> {path}[/]")


# --- report ----------------------------------------------------------------

def _channel_stats(conn, channel_id):
    total = conn.execute(
        "SELECT COUNT(*) n FROM messages WHERE channel_id=?", (channel_id,)
    ).fetchone()["n"]
    classified = conn.execute(
        "SELECT COUNT(*) n FROM message_topics WHERE channel_id=?", (channel_id,)
    ).fetchone()["n"]
    span = conn.execute(
        "SELECT MIN(date) a, MAX(date) b FROM messages WHERE channel_id=?", (channel_id,)
    ).fetchone()
    return total, classified, span["a"], span["b"]


def _topic_distribution(conn, channel_id):
    return conn.execute(
        "SELECT primary_topic, COUNT(*) n FROM message_topics "
        "WHERE channel_id=? GROUP BY primary_topic ORDER BY n DESC",
        (channel_id,),
    ).fetchall()


def _top_tickers(conn, channel_id, limit=15):
    counter: Counter = Counter()
    for row in conn.execute(
        "SELECT tickers FROM message_topics "
        "WHERE channel_id=? AND tickers IS NOT NULL AND tickers != '[]'",
        (channel_id,),
    ):
        try:
            for t in json.loads(row["tickers"]):
                counter[str(t).upper()] += 1
        except (ValueError, TypeError):
            continue
    return counter.most_common(limit)


def _activity_by_day(conn, channel_id, tail=30):
    rows = conn.execute(
        "SELECT substr(date,1,10) d, COUNT(*) n FROM messages "
        "WHERE channel_id=? AND date IS NOT NULL GROUP BY d ORDER BY d",
        (channel_id,),
    ).fetchall()
    return rows[-tail:], len(rows)


def report(conn, channels) -> None:
    channel_ids = resolve_channel_ids(conn, channels)
    chans = storage.list_channels(conn, channel_ids)
    if not chans:
        console.print("[yellow]No parsed channels found. Run `parse` first.[/]")
        return

    for ch in chans:
        title = ch["title"] or ch["username"] or str(ch["channel_id"])
        console.rule(f"{title}  (@{ch['username']})")
        total, classified, first, last = _channel_stats(conn, ch["channel_id"])
        console.print(
            f"messages: [bold]{total}[/]   classified: [bold]{classified}[/]   "
            f"subscribers: {ch['participants_count']}\n"
            f"date span: {first} → {last}"
        )

        dist = _topic_distribution(conn, ch["channel_id"])
        if dist:
            t = Table(title="Topic distribution (primary_topic)")
            t.add_column("topic")
            t.add_column("messages", justify="right")
            t.add_column("%", justify="right")
            for r in dist:
                pct = (100.0 * r["n"] / classified) if classified else 0.0
                t.add_row(r["primary_topic"] or "—", str(r["n"]), f"{pct:.1f}")
            console.print(t)

        tickers = _top_tickers(conn, ch["channel_id"])
        if tickers:
            t = Table(title="Top tickers")
            t.add_column("ticker")
            t.add_column("mentions", justify="right")
            for sym, cnt in tickers:
                t.add_row(sym, str(cnt))
            console.print(t)

        days, n_days = _activity_by_day(conn, ch["channel_id"])
        if days:
            t = Table(title=f"Activity by day (last {len(days)} of {n_days} active days)")
            t.add_column("day")
            t.add_column("messages", justify="right")
            for r in days:
                t.add_row(r["d"], str(r["n"]))
            console.print(t)
        console.print()

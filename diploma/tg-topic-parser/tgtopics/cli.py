"""Typer CLI — `python -m tgtopics <command>`."""

from __future__ import annotations

import asyncio
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from tgtopics import classifier, config
from tgtopics import parser as parser_mod
from tgtopics import report as report_mod
from tgtopics import storage

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local Telegram channel parser + Claude topic classifier.",
)
console = Console()


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging")):
    config.setup_logging(verbose)


def _open_db():
    conn = storage.connect()
    storage.init_db(conn)
    return conn


@app.command()
def count(
    channels: Optional[List[str]] = typer.Option(None, "--channels", "-c",
                                                 help="Override channels.yaml"),
    batch_api: bool = typer.Option(False, "--batch-api",
                                   help="Project cost at the 50%-off Batches rate"),
):
    """Dry-run: message volume + date range + rough cost, no download, no classify."""
    chans = config.load_channels(channels)

    async def _run():
        from tgtopics.auth import connected_client
        async with connected_client() as client:
            return await parser_mod.count_channels(client, chans)

    results = asyncio.run(_run())
    table = Table(title="Channel count (dry-run)")
    table.add_column("channel")
    table.add_column("messages", justify="right")
    table.add_column("subscribers", justify="right")
    table.add_column("oldest")
    table.add_column("newest")
    table.add_column("est. classify $", justify="right")

    grand_total = 0
    for r in results:
        if "error" in r:
            table.add_row(r["username"], "[red]error[/]", "-", "-", "-", "-")
            continue
        grand_total += r["total"]
        cost = classifier.rough_cost_projection(r["total"], batch_api=batch_api)
        table.add_row(
            r["username"], str(r["total"]),
            str(r["participants"]) if r["participants"] is not None else "-",
            str(r["oldest"].date()) if r["oldest"] else "-",
            str(r["newest"].date()) if r["newest"] else "-",
            f"${cost:.2f}",
        )
    console.print(table)
    total_cost = classifier.rough_cost_projection(grand_total, batch_api=batch_api)
    mode = "Batches API (50% off)" if batch_api else "live"
    console.print(
        f"[bold]Total ~{grand_total} messages[/]; rough classify cost ({mode}): "
        f"[bold]${total_cost:.2f}[/] (very approximate — Haiku $1/$5 per 1M tok)."
    )


@app.command()
def parse(
    since: str = typer.Option(config.DEFAULT_SINCE, "--since", help="Backfill floor YYYY-MM-DD"),
    channels: Optional[List[str]] = typer.Option(None, "--channels", "-c"),
    download_media: bool = typer.Option(False, "--download-media",
                                        help="Also download media files (off by default)"),
    limit: Optional[int] = typer.Option(None, "--limit",
                                        help="Cap messages per channel (for testing)"),
):
    """Backfill history (resumable) and incrementally sync new messages."""
    chans = config.load_channels(channels)
    since_dt = parser_mod.parse_since(since)
    conn = _open_db()

    async def _run():
        from tgtopics.auth import connected_client
        async with connected_client() as client:
            await parser_mod.run_parse(client, conn, chans, since_dt, download_media, limit)

    try:
        asyncio.run(_run())
    finally:
        conn.close()


@app.command()
def classify(
    batch_api: bool = typer.Option(False, "--batch-api",
                                   help="Use Message Batches API (50% cheaper) for bulk runs"),
    batch_size: int = typer.Option(config.DEFAULT_BATCH_SIZE, "--batch-size"),
    reclassify: bool = typer.Option(False, "--reclassify",
                                    help="Re-classify everything, ignoring existing topics"),
    concurrency: int = typer.Option(config.DEFAULT_CONCURRENCY, "--concurrency",
                                    help="Live-path concurrent batches"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip cost confirmation"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    model: str = typer.Option(config.CLASSIFIER_MODEL, "--model"),
    channels: Optional[List[str]] = typer.Option(None, "--channels", "-c"),
):
    """Classify message topics via Claude (only unclassified, unless --reclassify)."""
    conn = _open_db()
    topics = config.load_topics()
    channel_ids = report_mod.resolve_channel_ids(conn, channels) if channels else None
    rows = storage.iter_unclassified(conn, channel_ids, limit, reclassify)
    if not rows:
        console.print("[green]Nothing to classify.[/]")
        conn.close()
        raise typer.Exit()

    non_empty = [r for r in rows if (r["text"] or "").strip()]
    if non_empty:
        est = classifier.estimate_cost(non_empty, topics, model, batch_size, batch_api)
        table = Table(title="Classification estimate")
        table.add_column("metric")
        table.add_column("value", justify="right")
        table.add_row("messages to classify", str(est["messages"]))
        table.add_row("batches", str(est["batches"]))
        table.add_row("~input tokens", f"{est['input_tokens']:,}")
        table.add_row("~output tokens", f"{est['output_tokens']:,}")
        table.add_row("mode", "Batches API (50% off)" if batch_api else "live")
        table.add_row("estimated cost", f"${est['cost']:.2f}")
        console.print(table)
        if not yes:
            typer.confirm(f"Proceed and spend ~${est['cost']:.2f}?", abort=True)

    classifier.run_classify(conn, rows, topics, model, batch_size, concurrency, batch_api)
    conn.close()


@app.command()
def export(
    format: str = typer.Option("json", "--format", "-f", help="json or csv"),
    channels: Optional[List[str]] = typer.Option(None, "--channels", "-c"),
    out: Optional[str] = typer.Option(None, "--out", help="Output path (default exports/)"),
):
    """Export messages joined with their topics to JSON or CSV."""
    fmt = format.lower()
    if fmt not in ("json", "csv"):
        raise typer.BadParameter("format must be 'json' or 'csv'")
    conn = _open_db()
    report_mod.export(conn, fmt, channels, out)
    conn.close()


@app.command()
def report(
    channels: Optional[List[str]] = typer.Option(None, "--channels", "-c"),
):
    """Per-channel topic distribution, top tickers, and activity by day."""
    conn = _open_db()
    report_mod.report(conn, channels)
    conn.close()


if __name__ == "__main__":
    app()

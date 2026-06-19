"""Stage 2 — classification via Claude Haiku (live + Message Batches paths)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time

from anthropic import Anthropic, AsyncAnthropic
from rich.console import Console

from tgtopics import storage
from tgtopics.config import (
    BATCH_MAX_REQUESTS,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
    Topic,
    get_anthropic_key,
)

log = logging.getLogger("tgtopics.classifier")
console = Console()

MAX_TEXT_CHARS = 2000          # truncate long messages before sending
MAX_RETRIES = 2               # per-batch JSON/API retries (live path)
TOKENS_PER_MSG = 200          # max_tokens sizing per message (live path)
OUTPUT_TOKENS_PER_MSG = 60    # expected output tokens per message (estimates)
AVG_INPUT_TOKENS_PER_MSG = 60  # rough input tokens/message for `count` projection
POLL_INTERVAL = 30            # seconds between Batches API polls


def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --- prompts ---------------------------------------------------------------

def build_system_prompt(topics: list[Topic]) -> str:
    lines = [
        "You are a financial-news topic classifier for Telegram messages.",
        "",
        "TAXONOMY (use these exact English ids as labels):",
    ]
    for t in topics:
        lines.append(f"- {t.id}: {t.description}" if t.description else f"- {t.id}")
    lines += [
        "",
        "Rules:",
        "- Message text may be in Russian or English.",
        "- Use labels ONLY from the taxonomy ids above (English ids), even for Russian text.",
        "- A message may belong to several topics (multilabel); pick the single best as primary_topic.",
        "- tickers: stock/crypto/FX symbols mentioned (e.g. AAPL, BTC, EURUSD); use [] if none.",
        "- summary: <= 12 words, written in the SAME language as the message.",
        "- confidence: your confidence in primary_topic, from 0.0 to 1.0.",
        "",
        "Output: respond with ONLY a JSON array (no prose, no markdown code fences).",
        "One object per input message, each shaped exactly:",
        '{"message_id": <int>, "topics": ["..."], "primary_topic": "...", '
        '"confidence": 0.0, "tickers": ["..."], "summary": "..."}',
    ]
    return "\n".join(lines)


def build_user_prompt(batch) -> str:
    parts = ["Classify these messages and return a JSON array of objects:\n"]
    for row in batch:
        text = (row["text"] or "").strip().replace("\n", " ")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "…"
        parts.append(f"message_id={row['message_id']}: {text}")
    return "\n".join(parts)


def system_blocks(topics: list[Topic]) -> list[dict]:
    # cache_control marks the (identical) taxonomy prompt as cacheable. Note: Haiku
    # 4.5's minimum cacheable prefix is 4096 tokens, so a short taxonomy may not
    # actually cache — the real savings come from the Batches discount + big batches.
    return [{
        "type": "text",
        "text": build_system_prompt(topics),
        "cache_control": {"type": "ephemeral"},
    }]


# --- JSON parsing / validation --------------------------------------------

def extract_json_array(text: str):
    """Strip optional ``` fences and parse the first JSON array found."""
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        text = text[nl + 1:] if nl != -1 else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON array found in model output")
    return json.loads(text[start:end + 1])


def normalize_item(item, valid_ids, batch_ids):
    if not isinstance(item, dict):
        return None
    mid = item.get("message_id")
    if not isinstance(mid, int):
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            return None
    if batch_ids is not None and mid not in batch_ids:
        return None
    primary = item.get("primary_topic")
    if primary not in valid_ids:
        primary = "other"
    topics = [t for t in (item.get("topics") or []) if t in valid_ids]
    if primary not in topics:
        topics = [primary] + topics
    try:
        conf = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
    except (TypeError, ValueError):
        conf = 0.0
    tickers = [str(x) for x in (item.get("tickers") or [])]
    summary = str(item.get("summary") or "")[:300]
    return {
        "message_id": mid, "primary_topic": primary, "topics": topics,
        "confidence": conf, "tickers": tickers, "summary": summary,
    }


# --- empty-text shortcut ---------------------------------------------------

def mark_empty(conn, rows) -> list:
    """Record media-only/empty messages as 'other' (no API call); return the rest."""
    non_empty = []
    for row in rows:
        if (row["text"] or "").strip():
            non_empty.append(row)
        else:
            storage.upsert_topic(conn, row["channel_id"], row["message_id"],
                                 "other", ["other"], 0.0, [], "", "skipped")
    conn.commit()
    return non_empty


# --- cost estimation -------------------------------------------------------

def rough_cost_projection(n_messages: int, batch_api: bool = False) -> float:
    cost = (n_messages * AVG_INPUT_TOKENS_PER_MSG * PRICE_INPUT_PER_MTOK / 1e6
            + n_messages * OUTPUT_TOKENS_PER_MSG * PRICE_OUTPUT_PER_MTOK / 1e6)
    return cost * 0.5 if batch_api else cost


def estimate_cost(rows, topics, model, batch_size, batch_api) -> dict:
    n = len(rows)
    n_batches = max(1, math.ceil(n / batch_size))
    sample = rows[:batch_size]
    per_batch_input = None
    try:
        client = Anthropic(api_key=get_anthropic_key())
        per_batch_input = client.messages.count_tokens(
            model=model,
            system=build_system_prompt(topics),
            messages=[{"role": "user", "content": build_user_prompt(sample)}],
        ).input_tokens
    except Exception as e:
        log.debug("count_tokens failed, using rough estimate: %s", e)

    if per_batch_input is not None:
        input_tokens = per_batch_input * n_batches
    else:
        input_tokens = n * AVG_INPUT_TOKENS_PER_MSG
    output_tokens = n * OUTPUT_TOKENS_PER_MSG
    cost = (input_tokens * PRICE_INPUT_PER_MTOK / 1e6
            + output_tokens * PRICE_OUTPUT_PER_MTOK / 1e6)
    if batch_api:
        cost *= 0.5
    return {
        "messages": n, "batches": n_batches,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "cost": cost,
    }


# --- live path (AsyncAnthropic + semaphore) --------------------------------

async def _classify_one_batch(client, conn, batch, system, valid_ids, model):
    batch_ids = {r["message_id"] for r in batch}
    max_tokens = max(512, len(batch) * TOKENS_PER_MSG)
    user = build_user_prompt(batch)
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await client.messages.create(
                model=model, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            log.warning("API error (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, e)
            await asyncio.sleep(2 ** attempt)
            continue
        if resp.stop_reason == "max_tokens" and len(batch) > 1:
            mid = len(batch) // 2
            await _classify_one_batch(client, conn, batch[:mid], system, valid_ids, model)
            await _classify_one_batch(client, conn, batch[mid:], system, valid_ids, model)
            return
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            data = extract_json_array(text)
        except (ValueError, json.JSONDecodeError) as e:
            log.warning("JSON parse failed (attempt %d/%d): %s", attempt + 1, MAX_RETRIES + 1, e)
            continue
        _persist_batch(conn, batch, data, valid_ids, batch_ids, model)
        return
    log.error("Giving up on a batch of %d after retries; left unclassified", len(batch))


def _persist_batch(conn, batch, data, valid_ids, batch_ids, model):
    by_id = {}
    for item in data:
        norm = normalize_item(item, valid_ids, batch_ids)
        if norm:
            by_id[norm["message_id"]] = norm
    for row in batch:
        norm = by_id.get(row["message_id"])
        if norm is None:
            continue  # missing -> stays unclassified, retried on next run
        storage.upsert_topic(conn, row["channel_id"], row["message_id"],
                             norm["primary_topic"], norm["topics"], norm["confidence"],
                             norm["tickers"], norm["summary"], model)
    conn.commit()


async def classify_live(conn, rows, topics, model, batch_size, concurrency):
    client = AsyncAnthropic(api_key=get_anthropic_key())
    system = system_blocks(topics)
    valid_ids = {t.id for t in topics}
    batches = list(chunk(rows, batch_size))
    sem = asyncio.Semaphore(concurrency)
    done = 0
    lock = asyncio.Lock()

    async def worker(batch):
        nonlocal done
        async with sem:
            await _classify_one_batch(client, conn, batch, system, valid_ids, model)
        async with lock:
            done += len(batch)
            if done % (batch_size * 10) < len(batch):
                console.print(f"  classified ~{done}/{len(rows)}")

    await asyncio.gather(*(worker(b) for b in batches))
    console.print(f"[green]Live classification done (~{len(rows)} messages).[/]")


# --- Message Batches API path ----------------------------------------------

def _channel_from_custom_id(custom_id: str) -> int:
    # format: "g-<channel_id>-<first_message_id>"
    return int(custom_id.split("-")[1])


def _persist_results(conn, channel_id, items, valid_ids, model) -> int:
    n = 0
    for item in items:
        norm = normalize_item(item, valid_ids, batch_ids=None)
        if not norm:
            continue
        if not storage.message_exists(conn, channel_id, norm["message_id"]):
            continue  # ignore hallucinated ids
        storage.upsert_topic(conn, channel_id, norm["message_id"], norm["primary_topic"],
                             norm["topics"], norm["confidence"], norm["tickers"],
                             norm["summary"], model)
        n += 1
    return n


def _poll_and_persist(client, conn, valid_ids, model):
    for job in storage.open_batch_jobs(conn):
        bid = job["batch_id"]
        while True:
            batch = client.messages.batches.retrieve(bid)
            if batch.processing_status == "ended":
                break
            console.print(f"  batch {bid}: {batch.processing_status} "
                          f"(succeeded={batch.request_counts.succeeded}) …")
            time.sleep(POLL_INTERVAL)
        persisted = 0
        for result in client.messages.batches.results(bid):
            if result.result.type != "succeeded":
                continue
            channel_id = _channel_from_custom_id(result.custom_id)
            text = "".join(b.text for b in result.result.message.content if b.type == "text")
            try:
                items = extract_json_array(text)
            except (ValueError, json.JSONDecodeError):
                continue
            persisted += _persist_results(conn, channel_id, items, valid_ids, model)
        conn.commit()
        storage.update_batch_job(conn, bid, "done")
        console.print(f"[green]  batch {bid}: persisted {persisted} classifications[/]")


def _submit_batches(client, conn, requests):
    for i in range(0, len(requests), BATCH_MAX_REQUESTS):
        part = requests[i:i + BATCH_MAX_REQUESTS]
        batch = client.messages.batches.create(requests=part)
        storage.add_batch_job(conn, batch.id, len(part))
        console.print(f"  submitted batch {batch.id} ({len(part)} requests)")


def classify_batch_api(conn, rows, topics, model, batch_size):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = Anthropic(api_key=get_anthropic_key())
    valid_ids = {t.id for t in topics}
    system = system_blocks(topics)

    # 1. Finish any in-flight jobs from a previous (interrupted) run.
    _poll_and_persist(client, conn, valid_ids, model)

    # 2. Re-filter: some rows may now be classified by the resumed jobs.
    pending = [r for r in rows if not storage.is_classified(conn, r["channel_id"], r["message_id"])]
    if not pending:
        console.print("[green]Nothing left to classify.[/]")
        return

    # 3. Build one request per group of N messages; chunk into batch jobs under the cap.
    requests = []
    for batch in chunk(pending, batch_size):
        cid = f"g-{batch[0]['channel_id']}-{batch[0]['message_id']}"
        max_tokens = max(512, len(batch) * OUTPUT_TOKENS_PER_MSG * 4)
        requests.append(Request(
            custom_id=cid,
            params=MessageCreateParamsNonStreaming(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": build_user_prompt(batch)}],
            ),
        ))
    _submit_batches(client, conn, requests)

    # 4. Poll the newly-submitted jobs to completion and persist.
    _poll_and_persist(client, conn, valid_ids, model)


# --- orchestration ---------------------------------------------------------

def run_classify(conn, rows, topics, model, batch_size, concurrency, batch_api):
    non_empty = mark_empty(conn, rows)
    if not non_empty:
        console.print("[green]No non-empty messages to classify (empties marked 'other').[/]")
        return
    if batch_api:
        classify_batch_api(conn, non_empty, topics, model, batch_size)
    else:
        asyncio.run(classify_live(conn, non_empty, topics, model, batch_size, concurrency))

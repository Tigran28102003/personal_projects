# tg-topic-parser

Parse public Telegram channels into a **local** SQLite store, then label each message's
topic with the Claude API. The two stages are independent over one shared database, so you
can re-derive topics without re-downloading anything.

**Everything stays on your machine** except the **message text** sent to the Claude API
during the classification stage.

## How it works

```
parse  (Telethon, your user session)  ──►  data.db  ◄──  classify  (Claude Haiku)
   stage 1: download channel history            stage 2: topic labels (separate table)
```

- **Stage 1 — parse**: full history backfill from 2019 (resumable), then incremental sync.
  Stores message text, dates, views/forwards, reactions, reply/album links and media
  metadata. Files are **not** downloaded unless you pass `--download-media`.
- **Stage 2 — classify**: sends message text to `claude-haiku-4-5` in batches and writes
  multilabel topics, a primary topic, confidence, tickers, and a short summary to a
  separate `message_topics` table. Already-labelled messages are skipped (cheap re-runs).

## Requirements

- Python 3.11+
- A Telegram account (user session, MTProto via Telethon)
- An Anthropic API key

## Getting API keys

**Telegram** (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, `TELEGRAM_PHONE`):
1. Go to <https://my.telegram.org> and log in with your phone number.
2. Open **API development tools**, create an app (any title/short name).
3. Copy **api_id** and **api_hash**. `TELEGRAM_PHONE` is your number in international
   format, e.g. `+15551234567`.

On the first `parse`/`count` run you'll be prompted in the terminal for the login code
(and your 2FA password if enabled). The session is then saved under `session/` and reused.

**Anthropic** (`ANTHROPIC_API_KEY`):
1. Go to <https://console.anthropic.com/settings/keys> and create a key.

## Install

```bash
cd tg-topic-parser
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the four values
```

Edit `channels.yaml` to add/remove channels and `topics.yaml` to tune the taxonomy.

## Usage

All commands run as `python -m tgtopics <command>` (add `-v` for debug logging).

```bash
# 1. Dry-run: how many messages, what date range, rough classification cost — no download.
python -m tgtopics count
python -m tgtopics count --batch-api          # project cost at the 50%-off Batches rate

# 2. Backfill full history from 2019 (resumable: Ctrl-C and re-run to continue).
python -m tgtopics parse --since 2019-01-01
python -m tgtopics parse --channels @buyside --limit 200   # quick test on one channel
# Re-running after a completed backfill fetches only new messages (incremental).

# 3. Classify topics. Shows a cost estimate and asks to confirm.
python -m tgtopics classify --batch-api       # bulk first run, 50% cheaper (async)
python -m tgtopics classify                    # live path, good for catching up new msgs
python -m tgtopics classify --reclassify -y    # redo everything, skip confirmation

# 4. Output.
python -m tgtopics report
python -m tgtopics export --format csv
python -m tgtopics export --format json --channels @buyside
```

### Bulk vs live classification

- `--batch-api` uses the Claude **Message Batches API**: ~50% cheaper, processes
  asynchronously, and is resumable (a job id is stored; re-running resumes it). Best for
  the first large backfill.
- Without `--batch-api`, classification runs **live** with bounded concurrency — lower
  latency, ideal for labelling the handful of new messages after an incremental `parse`.

## Storage

Everything is local to this folder:

- `data.db` — SQLite (`channels`, `messages`, `message_topics`, `parse_progress`,
  `batch_jobs`).
- `session/` — Telethon session (secret; gitignored).
- `media/` — downloaded files, only if `--download-media` is used.
- `exports/` — JSON/CSV exports.

`.env`, `session/`, `data.db`, `media/`, and `exports/` are all gitignored — don't commit
secrets or session files.

## Privacy, ToS, and legal

- Use this only on **public channels**, or channels you are lawfully a member of. Respect
  [Telegram's Terms of Service](https://telegram.org/tos).
- Nothing leaves your machine except the **message text** sent to the Claude API at
  classification time. Telegram credentials, the session, and all data stay local.
- This tool collects **channel content only**. It does not scrape channel members or
  personal user data, and implements no ban-evasion. Don't use it to do those things.

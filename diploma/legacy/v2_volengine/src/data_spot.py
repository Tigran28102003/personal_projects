"""Long spot-OHLC loader for the Vol-Engine (A-M0, refinement R1).

RV needs only intraday prices (not perp/funding), so we use the LONGEST consistent series:
Binance **spot** BTCUSDT 1h OHLC from ≈2017-08 — adding the 2017 parabola and 2018 bear, two major
vol regimes the perp inception (2019-09) would exclude. Unlike the v2_microstructure snapshot (which
keeps only close), this loader keeps intraday **high/low** so RV can use range estimators
(Garman-Klass / Rogers-Satchell, R2). Network is touched only in ``build_spot_snapshot``; everything
else reads the frozen parquet.

R4 (bar integrity): we audit the bars-per-day distribution (not just NaN) and flag short days; the
cross-day return convention is 24/7 (the first hourly return of a UTC day is 23:00→00:00 of the prior
day — no overnight gap), handled downstream in rv.py.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from . import config

# Reuse pure utilities from the legacy / v2 libraries (config put them on sys.path).
from free_features import _session, _SPOT_BASE, _to_ms  # noqa: E402
from v2_microstructure.src.data_snapshot import content_hash  # noqa: E402

logger = logging.getLogger(__name__)

OHLC_COLUMNS = ("open", "high", "low", "close", "volume")
_INTERVAL_MIN = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def fetch_spot_ohlc(symbol: str = config.SYMBOL, interval: str = config.SPOT_INTERVAL,
                    start=config.SPOT_START, end=None,
                    session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Binance spot klines with OHLC kept. Index = open_time (UTC, tz-naive). Paginated by 1000."""
    end = end or pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    sess = session or _session()
    step = _INTERVAL_MIN.get(interval, 60) * 60_000
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    rows, cursor = [], start_ms
    for _ in range(100_000):
        if cursor >= end_ms:
            break
        r = sess.get(_SPOT_BASE + "/api/v3/klines",
                     params={"symbol": symbol, "interval": interval,
                             "startTime": cursor, "endTime": end_ms, "limit": 1000}, timeout=20)
        batch = r.json()
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= cursor:
            break
        cursor = nxt
        if len(batch) < 1000:
            break
    if not rows:
        raise RuntimeError("spot klines came back empty — cannot build the RV series")
    arr = pd.DataFrame(rows)
    out = pd.DataFrame({
        "open": arr[1].astype(float), "high": arr[2].astype(float), "low": arr[3].astype(float),
        "close": arr[4].astype(float), "volume": arr[5].astype(float),
    })
    out.index = pd.to_datetime(arr[0].astype("int64"), unit="ms")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "timestamp"
    return out


def audit_bars_per_day(df: pd.DataFrame, expected: int = config.BARS_PER_DAY) -> dict:
    """Distribution of bars per UTC day; flags short days (missing hours bias RV downward)."""
    per_day = df.groupby(df.index.normalize()).size()
    short = per_day[per_day < expected]
    return {
        "n_days": int(len(per_day)),
        "full_days": int((per_day == expected).sum()),
        "short_days": int((per_day < expected).sum()),
        "over_days": int((per_day > expected).sum()),
        "min_bars": int(per_day.min()), "max_bars": int(per_day.max()),
        "median_bars": float(per_day.median()),
        "first_short_days": [str(d.date()) for d in short.index[:10]],
        "frac_full": float((per_day == expected).mean()),
    }


def build_spot_snapshot(start=config.SPOT_START, end=None, force: bool = False) -> pd.DataFrame:
    """Fetch the long spot-OHLC series, audit bars/day, and freeze to parquet + manifest."""
    config.ensure_dirs()
    if config.SPOT_SNAPSHOT_PATH.exists() and not force:
        logger.info("spot snapshot exists at %s (force=True to rebuild)", config.SPOT_SNAPSHOT_PATH)
        return load_spot_snapshot()
    df = fetch_spot_ohlc(start=start, end=end)
    audit = audit_bars_per_day(df)
    chash = content_hash(df)
    df.to_parquet(config.SPOT_SNAPSHOT_PATH)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": config.SYMBOL, "interval": config.SPOT_INTERVAL, "market": "spot",
        "range": {"start": str(df.index.min()), "end": str(df.index.max()), "n_rows": int(len(df))},
        "columns": list(df.columns), "content_sha256": chash,
        "bars_per_day_audit": audit,
        "cross_day_return_convention": "24/7: first hourly return of a UTC day = 23:00->00:00 prior day (no overnight gap)",
    }
    config.SPOT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info("spot snapshot frozen: %d rows %s→%s sha=%s", len(df), df.index.min(),
                df.index.max(), chash[:12])
    logger.info("bars/day: full=%d short=%d (frac_full=%.4f) min=%d max=%d",
                audit["full_days"], audit["short_days"], audit["frac_full"],
                audit["min_bars"], audit["max_bars"])
    return df


def load_spot_snapshot(verify_hash: bool = True) -> pd.DataFrame:
    if not config.SPOT_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"no frozen spot snapshot at {config.SPOT_SNAPSHOT_PATH} — run build_spot_snapshot() first")
    df = pd.read_parquet(config.SPOT_SNAPSHOT_PATH)
    if verify_hash and config.SPOT_MANIFEST.exists():
        expected = json.loads(config.SPOT_MANIFEST.read_text()).get("content_sha256")
        if expected and expected != content_hash(df):
            raise RuntimeError("spot snapshot hash mismatch — data drifted; rebuild or investigate")
    return df


def _main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Vol-Engine spot-OHLC snapshot (audit / build).")
    p.add_argument("cmd", choices=["audit", "build"], nargs="?", default="build")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "audit" and config.SPOT_SNAPSHOT_PATH.exists():
        print(json.dumps(audit_bars_per_day(load_spot_snapshot()), indent=2))
    else:
        build_spot_snapshot(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

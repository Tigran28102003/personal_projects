"""Frozen, hashed feature snapshot for the direction-C redesign (M0).

Two responsibilities, in order:

1. ``audit_sources()`` — the *first* thing M0 does. Query each upstream source for its real
   earliest timestamp and granularity (perp klines, spot klines, funding, Coin Metrics) and
   compute the EFFECTIVE INTERSECTION RANGE. This is logged before any feature is built; if
   it narrows below expectation the caller is expected to stop and raise it with a human.

2. ``build_snapshot()`` — fetch the MAXIMAL hourly history once (perp inception → now),
   assemble flow/derivatives + klines-proxy microstructure (order-flow from the PERP, the
   instrument we trade) + on-chain, and freeze it to a parquet with a content hash recorded
   in ``snapshot_manifest.json``. ``load_snapshot()`` reads ONLY the frozen file and verifies
   the hash. Network is touched here and nowhere else in the package.

Anti-leakage note: the snapshot stores CONTEMPORANEOUS feature values (no shift) plus the raw
price path needed for triple-barrier labeling. The causal t-1 shift is applied downstream in
``features.py`` (M2), so labeling can see the full path while models only ever see t-1 state.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

from . import config

# Sibling legacy library in ACTUAL_VERSION (config.LEGACY_LIB put it on sys.path).
from free_features import (  # noqa: E402
    _session, _SPOT_BASE, _FUT_BASE, _CM_BASE,
    fetch_klines, fetch_funding, fetch_coinmetrics,
    orderflow_features, basis_features, funding_on_grid, onchain_features,
    DEFAULT_CM_METRICS, ORDERFLOW_FEATURES, DERIV_FEATURES, ONCHAIN_FEATURES,
)

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("perp_close", "spot_close", "perp_volume")
FEATURE_COLUMNS = tuple(ORDERFLOW_FEATURES) + tuple(DERIV_FEATURES) + tuple(ONCHAIN_FEATURES)


# --------------------------------------------------------------------------- earliest-ts probes
def _earliest_kline(symbol: str, interval: str, market: str,
                    session: Optional[requests.Session] = None) -> Optional[pd.Timestamp]:
    """Earliest available kline open_time (UTC, tz-naive) via a 1-row probe from epoch."""
    sess = session or _session()
    base = _FUT_BASE if market == "futures" else _SPOT_BASE
    path = "/fapi/v1/klines" if market == "futures" else "/api/v3/klines"
    try:
        r = sess.get(base + path, params={"symbol": symbol, "interval": interval,
                     "startTime": 0, "limit": 1}, timeout=15)
        batch = r.json()
        if isinstance(batch, list) and batch:
            return pd.to_datetime(int(batch[0][0]), unit="ms")
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        logger.warning("earliest_kline(%s/%s) failed: %s", market, symbol, e)
    return None


def _earliest_funding(symbol: str, session: Optional[requests.Session] = None) -> Optional[pd.Timestamp]:
    """Earliest funding settlement. NOTE: the endpoint returns the *most recent* rows when the
    window is unbounded, so we must pass an endTime to anchor to the earliest ascending row."""
    sess = session or _session()
    now_ms = int(pd.Timestamp.now(tz="UTC").tz_localize(None).value // 1_000_000)
    try:
        r = sess.get(_FUT_BASE + "/fapi/v1/fundingRate",
                     params={"symbol": symbol, "startTime": 0, "endTime": now_ms, "limit": 1}, timeout=15)
        batch = r.json()
        if isinstance(batch, list) and batch:
            return pd.to_datetime(int(batch[0]["fundingTime"]), unit="ms")
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        logger.warning("earliest_funding(%s) failed: %s", symbol, e)
    return None


def _earliest_coinmetrics(metrics, asset: str = "btc",
                          session: Optional[requests.Session] = None) -> Optional[pd.Timestamp]:
    sess = session or _session()
    # NOTE: like funding, an unbounded query returns the most recent page; pass end_time to
    # anchor to the earliest ascending row.
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).strftime("%Y-%m-%d")
    try:
        r = sess.get(_CM_BASE + "/timeseries/asset-metrics", params={
            "assets": asset, "metrics": ",".join(metrics), "frequency": "1d",
            "start_time": "2009-01-01", "end_time": today, "page_size": 1,
            "paging_from": "start",  # v4 default paginates from the END; we want the earliest row
        }, timeout=30)
        data = r.json().get("data", [])
        if data:
            return pd.to_datetime(data[0]["time"]).tz_localize(None).normalize()
    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        logger.warning("earliest_coinmetrics failed: %s", e)
    return None


def audit_sources(symbol: str = config.SYMBOL, interval: str = config.INTERVAL) -> dict:
    """Probe every source's earliest timestamp + granularity and the effective intersection.

    Returns a JSON-serialisable dict. The effective start is the LATEST of the per-source
    earliest timestamps (the binding constraint, expected to be the perp ≈2019-09-08).
    """
    sess = _session()
    sources = {
        "perp_klines":  {"earliest": _earliest_kline(symbol, interval, "futures", sess),
                         "granularity": interval, "market": "futures"},
        "spot_klines":  {"earliest": _earliest_kline(symbol, interval, "spot", sess),
                         "granularity": interval, "market": "spot"},
        "funding":      {"earliest": _earliest_funding(symbol, sess),
                         "granularity": "8h"},
        "coinmetrics":  {"earliest": _earliest_coinmetrics(DEFAULT_CM_METRICS, "btc", sess),
                         "granularity": "1d", "metrics": list(DEFAULT_CM_METRICS)},
    }
    # Effective range start = max of the sources that bound the *tradable* hourly history.
    # On-chain (daily) is deprioritised on hourly (ffilled), so it does NOT bind the start.
    binding = ["perp_klines", "spot_klines", "funding"]
    earliest_vals = [sources[k]["earliest"] for k in binding if sources[k]["earliest"] is not None]
    eff_start = max(earliest_vals) if earliest_vals else None
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")

    audit = {
        "symbol": symbol,
        "interval": interval,
        "sources": {k: {**v, "earliest": (None if v["earliest"] is None else str(v["earliest"]))}
                    for k, v in sources.items()},
        "effective_start": None if eff_start is None else str(eff_start),
        "effective_end": str(now),
        "binding_sources": binding,
        "audited_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    logger.info("source audit: %s", json.dumps(audit, indent=2))
    for k, v in sources.items():
        logger.info("  %-12s earliest=%s  granularity=%s", k, v["earliest"], v["granularity"])
    logger.info("  EFFECTIVE RANGE: %s -> %s", eff_start, now)
    return audit


# --------------------------------------------------------------------------- content hashing
def content_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 over the dataframe content (values + index), format-agnostic."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update("|".join(map(str, df.columns)).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- snapshot build
def build_snapshot(start: Optional[str] = None, end: Optional[str] = None,
                   force: bool = False, use_cache: bool = True) -> pd.DataFrame:
    """Fetch the maximal hourly history, assemble features, and freeze to parquet + manifest.

    Order-flow proxies (tfi/cvd/vpin) are computed from the PERP klines (the traded
    instrument); basis = perp/spot - 1; funding aligned to the hourly grid without leakage;
    on-chain (daily) ffilled onto the grid. Values are CONTEMPORANEOUS (no shift) — the t-1
    causal lag is applied later in features.py.
    """
    config.ensure_dirs()
    if config.SNAPSHOT_PATH.exists() and not force:
        logger.info("snapshot exists at %s (use force=True to rebuild)", config.SNAPSHOT_PATH)
        return load_snapshot()

    sess = _session()
    audit = audit_sources(config.SYMBOL, config.INTERVAL)
    start = start or audit["effective_start"]
    if start is None:
        raise RuntimeError("source audit could not determine an effective start (no network?)")
    end = end or audit["effective_end"]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    logger.info("building snapshot %s %s  [%s -> %s]", config.SYMBOL, config.INTERVAL, start_ts, end_ts)

    perp = fetch_klines(config.SYMBOL, config.INTERVAL, start_ts, end_ts, "futures", sess, use_cache)
    spot = fetch_klines(config.SYMBOL, config.INTERVAL, start_ts, end_ts, "spot", sess, use_cache)
    if perp.empty:
        raise RuntimeError("perp klines came back empty — cannot build snapshot")

    grid = perp.index  # the tradable hourly grid
    out = pd.DataFrame(index=grid)
    out["perp_close"] = perp["close"]
    out["spot_close"] = spot["close"].reindex(grid) if not spot.empty else np.nan
    out["perp_volume"] = perp["volume"]

    # 1) klines-proxy microstructure from the PERP (not spot — we trade the perp)
    out = out.join(orderflow_features(perp).reindex(grid))
    # 2) derivatives: basis (perp vs spot) + funding on the grid
    if not spot.empty:
        out = out.join(basis_features(perp["close"], spot["close"]).reindex(grid))
    funding = fetch_funding(config.SYMBOL, start_ts, end_ts, sess, use_cache)
    out = out.join(funding_on_grid(funding, grid))
    # 3) on-chain (daily) ffilled onto the hourly grid (deprioritised; lag applied downstream)
    cm = fetch_coinmetrics(DEFAULT_CM_METRICS, start_ts, end_ts, session=sess, use_cache=use_cache)
    if not cm.empty:
        onchain = onchain_features(cm)
        onchain_h = onchain.reindex(onchain.index.union(grid)).sort_index().ffill().reindex(grid)
        out = out.join(onchain_h)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"  # free_features klines leave the index unnamed (`0`)

    chash = content_hash(out)
    out.to_parquet(config.SNAPSHOT_PATH)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": config.SYMBOL,
        "interval": config.INTERVAL,
        "freq": config.FREQ,
        "orderflow_market": config.ORDERFLOW_MARKET,
        "source_audit": audit,
        "effective_range": {"start": str(out.index.min()), "end": str(out.index.max()),
                            "n_rows": int(len(out))},
        "price_columns": list(PRICE_COLUMNS),
        "feature_columns": [c for c in FEATURE_COLUMNS if c in out.columns],
        "all_columns": list(out.columns),
        "lag_applied": False,
        "content_sha256": chash,
        "notes": "Contemporaneous values (no shift); orderflow from PERP; t-1 lag applied in features.py.",
    }
    config.SNAPSHOT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info("snapshot frozen: %d rows, %d cols, sha256=%s",
                len(out), out.shape[1], chash[:12])
    logger.info("  columns: %s", list(out.columns))
    return out


def load_snapshot(verify_hash: bool = True) -> pd.DataFrame:
    """Read the frozen snapshot (parquet) and verify its content hash against the manifest."""
    if not config.SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"no frozen snapshot at {config.SNAPSHOT_PATH} — run build_snapshot() first")
    df = pd.read_parquet(config.SNAPSHOT_PATH)
    if verify_hash and config.SNAPSHOT_MANIFEST.exists():
        manifest = json.loads(config.SNAPSHOT_MANIFEST.read_text())
        expected = manifest.get("content_sha256")
        actual = content_hash(df)
        if expected and expected != actual:
            raise RuntimeError(
                f"snapshot hash mismatch: manifest={expected[:12]} actual={actual[:12]} "
                "— data drifted; rebuild or investigate")
    return df


# --------------------------------------------------------------------------- CLI
def _main(argv=None) -> int:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Direction-C data snapshot (audit / build).")
    p.add_argument("cmd", choices=["audit", "build"], nargs="?", default="audit")
    p.add_argument("--force", action="store_true", help="rebuild even if a snapshot exists")
    p.add_argument("--no-cache", action="store_true", help="bypass the raw fetch CSV cache")
    args = p.parse_args(argv)
    if args.cmd == "audit":
        audit_sources()
    else:
        build_snapshot(force=args.force, use_cache=not args.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

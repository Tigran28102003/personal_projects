# v2_microstructure — Direction C redesign (Microstructure/Flow + Triple-Barrier + Meta-Label)

A self-contained, **standalone package at the git repo root** that **imports** the existing
pipeline — the sibling `ACTUAL_VERSION/` legacy library, attached via an explicit `sys.path`
entry in `src/config.py` (`LEGACY_LIB`) — and replaces only the *paradigm*: target, features,
and trading idea. The package is intentionally **NOT** inside `ACTUAL_VERSION/`. See
`../ACTUAL_VERSION/REDESIGN_BRIEF.md` (single source of truth) and the approved plan.

## What this is
- **Target:** vol-scaled **triple barrier** `{+1, 0, −1}` (path-dependent, vol-normalised) —
  replaces one-step `r_t` / directional-accuracy.
- **Features:** flow/derivatives (funding/basis/OI/on-chain) as the hourly core; klines-proxy
  microstructure (taker-buy imbalance, CVD, bar-VPIN) — **not** tick-level — as proxies.
- **Models:** baseline → linear → LightGBM 3-class ladder; Stage-2 meta-label sizing.
- **Instrument & PnL:** perp `BTCUSDT` (funding/basis + native short; taker 0.05%). Strategy
  PnL is decomposed into `price_pnl` vs `funding_pnl` so carry cannot masquerade as alpha.
- **Judge:** economics after costs (`net_cost_sharpe`, Sortino/Calmar/MDD, breakeven-cost) +
  `deflated_sharpe`/`PBO` on **de-overlapped trade returns**. DA is report-only.

## Layout
- `src/` — all Python (modules, scripts, orchestration).
- `figures/` — all plots (written only via `config.FIGURES_DIR`).
- `data/` — frozen feature snapshot (`features_snapshot.parquet`) + `snapshot_manifest.json`.
- `reports/<run_id>/` — the per-run artifact contract (manifest, fold/CPCV metrics, OOF,
  de-overlapped trade returns, PnL decomposition, equity, turnover, baselines, ablation, delta).

## Reproducibility
Fixed `SEED`, a frozen content-hashed data snapshot, and a `pip freeze` env lockfile hash
recorded in every run manifest — so "the result changed" can never be confused with seed noise.

## How to run
```bash
# from REPO_ROOT (.../ACTUAL_VERSION), using its .venv
.venv/bin/python -m v2_microstructure.src.data_snapshot audit     # M0: source-availability audit
.venv/bin/python -m v2_microstructure.src.data_snapshot build     # freeze the maximal snapshot
# later milestones:
# .venv/bin/python -m v2_microstructure.src.experiment            # full run -> reports/<run_id>/
```

## Hard-bar expectation (read before judging results)
DSR ≥ 0.95 is computed on **de-overlapped trade returns** (hundreds, not ~21k bars) — a
deliberately strict bar with a wide CI. **Most configs will not pass; that is the design,
not a bug.** Deployment requires `DSR ≥ 0.95 AND PBO ≤ 0.5`.

## Status
- **M0** (source audit + scaffolding + frozen snapshot) — in progress.
- M1 labeling · M2 features+tripwires · M3 Stage-1+ablation gate · M4 Stage-2+strategy ·
  M5 economics+artifacts · M6 fix-cycle · M7 sealed-holdout unseal.

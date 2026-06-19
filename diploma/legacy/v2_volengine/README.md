# v2_volengine — Direction A (Vol-Engine: RV-forecast + vol-timing; VRP backlog)

Standalone package at the repo root, sibling of `v2_microstructure/` and `ACTUAL_VERSION/`. It
**imports** the legacy library (`ACTUAL_VERSION/`, via `src/config.py` `LEGACY_LIB`) and reuses the
perp-era exo snapshot from `v2_microstructure/`. Direction C was retired after its M3 economic gate
failed (see `../v2_microstructure/FINDINGS_C.md`); A pivots to the **predictable** target.

## Thesis continuity
The original diploma reviewed the **GARCH family** for volatility. A returns to that volatility
target via **HAR-RV** — the realized-volatility successor to GARCH (Corsi 2009) — correcting the
price-level misstep. The econometric deliverable (Layer 1) stands at any sign of the trading result.

## Three layers (deliberately decoupled)
- **Layer 1 — forecasting (guaranteed):** 1-day-ahead **log realized volatility** from the long
  spot-OHLC series (Binance spot BTCUSDT 2017-08→, R1). Beat the **HAR family** (HAR / HAR-J / **HARQ
  central**) with ML+exo. Judged by proper vol-scoring (**QLIKE**, R²-logRV).
- **Layer 2 — trading (may fail its gate):** **vol-timing** overlay — scale perp exposure inversely
  to forecast RV. Judged by net-cost Sharpe vs buy&hold.
- **Layer 3 — backlog (never claimed validated):** DVOL / IV−RV / option-VRP.

## Two pre-registered kill-switches (distributional over CPCV; margins in `config` before any run)
- **Gate 1 (forecasting):** ML+exo beats HAR on QLIKE — **Diebold-Mariano (HLN)** primary +
  paired-bootstrap Δ-CI complement, on the perp-era 2019→ subsample. Fail → ML null, **HAR stands**.
- **Gate 2 (economics):** vol-timing net-cost Sharpe > buy&hold (hard bar: Moreira–Muir vs Cederburg).

**Principle carried from C/B:** a predictable target is *not* a trading edge; every economic claim
must survive its own pre-registered kill-switch. Honest measurement note: hourly RV is ~12× noisier
than 5-min RV → temper R² expectations; range estimators (Garman-Klass / Rogers-Satchell) are
preferred (R2). Holdout = last ~12 months, sealed, unsealed once at the end.

## Layout
`src/` (all code), `figures/`, `data/` (frozen spot-OHLC snapshot + manifest), `reports/<run_id>/`,
`notebooks/`. Run from the repo root: `ACTUAL_VERSION/.venv/bin/python -m v2_volengine.src.<module>`.

## Status
A-M0 (scaffolding + spot loader + RV) in progress · M1 HAR · M2 features+tripwires · M3 ML+Gate 1 ·
M4 vol-timing · M5 Gate 2 + artifacts · M6 DVOL backlog · M7 holdout unseal.

"""BTC 3-class direction forecasting (hourly, 24h horizon).

Reproducible diploma project. Public modules:

* ``config``        — seeds, paths, horizon/threshold, QUICK_MODE toggle, set_determinism().
* ``get_data``      — DataMaker (reused) + fetch_binance_ohlcv (long-history hourly OHLCV).
* ``free_features`` — Binance klines/funding/basis/orderflow + CoinMetrics on-chain (reused).
* ``features``      — trailing feature engineering + exogenous merge (leakage-safe).
* ``labeling``      — make_target: {-1,0,+1} label + matched forward return / exit info.
* ``validation``    — purged walk-forward + PurgedKFold/CPCV + recency/uniqueness/drift.
* ``models``        — Setup A/B/C with a unified fit/predict_proba3 interface.
* ``optimize``      — Optuna (TPE+MedianPruner) on purged walk-forward, MCC objective.
* ``backtest``      — next-open horizon strategy + buy&hold (reused helpers).
* ``metrics``       — classification + IC + economics + PSR/DSR/PBO/DM + evaluate().
* ``plotting``      — save_fig (writes ONLY to pictures/) + EDA/diagnostic plots.
* ``data_io``       — load_or_build_features + build_dataset (the data switch).
"""

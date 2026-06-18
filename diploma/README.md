# Forecasting Bitcoin Returns with Heterogeneous Financial Indicators

*Course project — Modern Methods of Data Analysis.*

## Overview

This project studies one-step-ahead forecasting of Bitcoin (BTC) logarithmic returns
from a heterogeneous set of financial, macroeconomic, crypto-market, and technical
indicators, and it compares gradient-boosted decision trees (LightGBM, XGBoost, CatBoost)
with neural sequence models (LSTM, GRU) under an identical, leakage-controlled walk-forward
protocol. Models are selected by an economically meaningful composite criterion and
assessed with a cost-aware backtest under multiple-testing controls (Deflated Sharpe Ratio
and Probability of Backtest Overfitting).

The prediction target is the logarithmic return `r_t = ln(P_t) - ln(P_{t-1})`; the price
path is reconstructed as `P̂_t = P_{t-1} · exp(r̂_t)`.

## Repository

The complete analysis is contained in the notebook
[`ACTUAL_VERSION/mmda_btc_report.ipynb`](ACTUAL_VERSION/mmda_btc_report.ipynb), which
reproduces every result from frozen CSV snapshots without network access. The supporting
logic is organised into Python modules for data acquisition, walk-forward orchestration,
models, validation, metrics, the backtest, and meta-labeling. A detailed component map and
schematic of the training procedures is given in
[`ACTUAL_VERSION/RAZBOR.md`](ACTUAL_VERSION/RAZBOR.md).

## Reproducibility

Install the dependencies from `ACTUAL_VERSION/requirements.txt` and run the notebook from
top to bottom. All preprocessing, feature selection, and hyperparameter tuning are fitted
on training data only within each walk-forward fold, so the reported out-of-sample metrics
are free of look-ahead leakage.

## Main findings

A small, momentum-like signal is detectable at all sampling frequencies, but it is
economically exploitable only at the daily horizon and only for the neural models: a daily
GRU with cost-aware position sizing beats Buy-and-Hold out of sample (Sharpe ≈ 0.89 versus
≈ 0.41) with a substantially smaller drawdown. At the hourly and 5-minute horizons every
strategy is net-negative after transaction costs, and the 5-minute results are flagged as
overfit by the Probability of Backtest Overfitting. The principal lesson is that
statistical predictability at high frequency does not translate into economic tradability.

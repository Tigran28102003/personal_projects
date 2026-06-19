"""Target construction for the 3-class direction problem.

Three pre-registered label variants (each a SEPARATE end-to-end experiment, never
mixed inside one Optuna study — the threshold is the label *definition*, not a tuned
hyperparameter):

* ``pointwise``     — ret = close[t+H]/close[t]-1; ±THRESHOLD cuts; exit = hold H bars.
* ``vol_scaled``    — threshold thr_t = k·σ_t with σ_t a CAUSAL EWMA vol (shift(1)),
                      scaled by √H; k ∈ {1.0,1.5,2.0} (each k a separate experiment).
* ``triple_barrier``— path-dependent: first of upper(+thr)/lower(-thr)/vertical(H);
                      fwd_ret + exit offset reflect the touch bar (label ↔ backtest).

All thresholds apply to the **simple** return ``close[t+H]/close[t]-1`` (R10). The
label is computed ON THE FLY and never cached, so switching ``TARGET_TYPE`` is instant.
``make_target`` returns ``(y3, fwd_ret, exit_info)`` where ``exit_info`` is the per-bar
**bars-to-exit offset** (int; ``None`` for fixed-horizon variants → backtest holds H).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # standalone
    import config


def _simple_forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """ret_t = close[t+H]/close[t] - 1 (простая forward-доходность; tail H -> NaN)."""
    c = close.astype(float)
    return c.shift(-horizon) / c - 1.0


def causal_ewma_sigma(close: pd.Series, span: int, horizon: int) -> pd.Series:
    """Каузальная EWMA-волатильность баровых лог-доходностей, сдвинутая на 1 бар и
    масштабированная на √H. shift(1) => σ_t использует только информацию ДО t (нет
    look-ahead в пороге метки). НЕ rolling-std (у него ghosting-артефакт)."""
    r = np.log(close.astype(float)).diff()
    sig = r.ewm(span=span, min_periods=max(2, span // 4)).std().shift(1)
    return sig * np.sqrt(horizon)


def _label_from_threshold(fwd_ret: pd.Series, thr) -> pd.Series:
    """+1 если ret>=thr; -1 если ret<=-thr; иначе 0 (thr скаляр или Series)."""
    y = pd.Series(0, index=fwd_ret.index, dtype="float")
    up = fwd_ret >= thr
    dn = fwd_ret <= -thr if np.isscalar(thr) else fwd_ret <= -thr
    y[up] = 1.0
    y[dn] = -1.0
    y[fwd_ret.isna()] = np.nan
    return y


def _triple_barrier(ohlc: pd.DataFrame, horizon: int, thr: float,
                    close_col: str, high_col: str, low_col: str
                    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Path-dependent triple-barrier. Возвращает (y3, fwd_ret, exit_off).

    Барьеры на цене относительно close[t]: верх +thr, низ −thr, вертикаль через H.
    Первое касание определяет метку. Внутрибаровое двойное касание (и верх, и низ в
    одном баре) трактуется как неопределённость -> метка 0 (консервативно, с оговоркой).
    `fwd_ret` = close[exit]/close[t]−1, `exit_off` = число баров от t до выхода (1..H)."""
    c = ohlc[close_col].to_numpy(dtype=float)
    hi = ohlc[high_col].to_numpy(dtype=float) if high_col in ohlc else c
    lo = ohlc[low_col].to_numpy(dtype=float) if low_col in ohlc else c
    n = len(c)
    y = np.full(n, np.nan)
    fr = np.full(n, np.nan)
    off = np.full(n, np.nan)
    for t in range(n):
        if t + horizon >= n:
            break  # tail: нет полного окна
        base = c[t]
        if base <= 0:
            continue
        up_b, dn_b = base * (1.0 + thr), base * (1.0 - thr)
        label, exit_m = 0, horizon
        for m in range(1, horizon + 1):
            s = t + m
            hit_up = hi[s] >= up_b
            hit_dn = lo[s] <= dn_b
            if hit_up and hit_dn:
                label, exit_m = 0, m       # двойное касание -> неопределённость
                break
            if hit_up:
                label, exit_m = 1, m
                break
            if hit_dn:
                label, exit_m = -1, m
                break
        s = t + exit_m
        y[t] = label
        fr[t] = c[s] / base - 1.0
        off[t] = exit_m
    idx = ohlc.index
    return (pd.Series(y, index=idx), pd.Series(fr, index=idx), pd.Series(off, index=idx))


def make_target(ohlc: pd.DataFrame, target_type: str = "pointwise",
                k: Optional[float] = None, horizon: Optional[int] = None,
                threshold: Optional[float] = None, close_col: str = "BTC_Close",
                high_col: str = "BTC_High", low_col: str = "BTC_Low"
                ) -> Tuple[pd.Series, pd.Series, Optional[pd.Series]]:
    """Построить 3-классовую метку на лету.

    Возвращает ``(y3, fwd_ret, exit_info)``:
      * ``y3``       — {-1,0,+1} с NaN в хвосте (нет полного окна) — caller дропает;
      * ``fwd_ret``  — реализованная доходность, СОГЛАСОВАННАЯ с правилом выхода;
      * ``exit_info``— bars-to-exit offset (для triple_barrier); ``None`` -> держать H.
    """
    H = horizon if horizon is not None else config.HORIZON
    thr = threshold if threshold is not None else config.THRESHOLD
    close = ohlc[close_col]

    if target_type == "pointwise":
        fwd = _simple_forward_return(close, H)
        return _label_from_threshold(fwd, thr), fwd, None

    if target_type == "vol_scaled":
        kk = k if k is not None else 1.5
        sigma = causal_ewma_sigma(close, config.EWMA_VOL_SPAN, H)
        thr_t = kk * sigma
        fwd = _simple_forward_return(close, H)
        y = _label_from_threshold(fwd, thr_t)
        # где порог не определён (warmup σ) -> метка NaN
        y[thr_t.isna()] = np.nan
        return y, fwd, None

    if target_type == "triple_barrier":
        return _triple_barrier(ohlc, H, thr, close_col, high_col, low_col)

    raise ValueError(f"unknown target_type={target_type!r}; expected one of {config.TARGET_TYPES}")

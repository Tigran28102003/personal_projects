"""Three problem formulations (A/B/C) behind one interface.

We compare *formulations*, not learners, so the base learner is fixed: LightGBM
(deterministic). All three expose:

* ``fit(X, y3, sample_weight=None, y_ret=None)`` — y3 ∈ {-1,0,+1}; ``y_ret`` (forward
  return) only used by cascade c2 to label the small-move direction.
* ``predict_proba3(X) -> (n,3)`` columns ``[P(-1), P(0), P(+1)]``.
* ``predict(X) -> {-1,0,+1}`` (argmax).
* ``signal_score(X) = P(+1) - P(-1)``.

Setups:
* **A** ``SetupSoftmax``  — one LightGBM multiclass.
* **B** ``SetupTwoBinary``— model_up=1[y3==+1], model_down=1[y3==-1] (both on all data);
  combine: P(+1)∝p_up(1-p_down), P(-1)∝(1-p_up)p_down, P(0)∝(1-p_up)(1-p_down)+p_up·p_down.
* **C** ``SetupCascade`` — gate=1[y3≠0], direction=1[y3==+1] vs 0[y3==-1];
  P(+1)=p_big·p_up, P(-1)=p_big·(1-p_up), P(0)=1-p_big. Variant **c1** trains direction
  on the real |ret|≥thr subset; **c2** trains on the full population weighted by an
  **out-of-fold** gate p_big (selection-bias demo, R2).

Sample weights (R5) compose class balance × label uniqueness × recency.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold

try:
    from . import config, validation
except ImportError:  # standalone
    import config, validation

CLASSES = (-1, 0, 1)


# ───────────────────────────── helpers ─────────────────────────────────────
def _lgbm(params: Optional[dict] = None, objective: str = "binary", num_class: int = 1):
    """LGBMClassifier с детерминированными настройками + переданными гиперпараметрами.
    num_threads/n_jobs берутся из COMPUTE_CFG (W1: один-поточные фиты, параллель — снаружи)."""
    p = dict(config.LGBM_DETERMINISTIC)
    lt = int(config.COMPUTE_CFG.get("lgbm_threads", 1))
    # Регуляризованный ДЕФОЛТ: мелкие деревья, высокий min_child, сабсэмплинг
    # строк/колонок, L1/L2 — чтобы default-путь (params={}) сам по себе не
    # переобучался при низком SNR. Optuna при желании переопределяет любой ключ.
    p.update({"n_estimators": 600, "learning_rate": 0.05,
              "num_leaves": 31, "max_depth": 6, "min_child_samples": 100,
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "reg_alpha": 0.1, "reg_lambda": 1.0,
              "n_jobs": lt, "num_threads": lt})
    if params:
        p.update(params)
    if objective == "multiclass":
        p["objective"] = "multiclass"
        p["num_class"] = num_class
    else:
        p["objective"] = "binary"
    return LGBMClassifier(**p)


def _prep_X(X):
    """float32-матрица фич (W1: вдвое меньше RAM, быстрее на малых данных; имена сохранены).
    NaN сохраняются (деревья кодируют нативно)."""
    return pd.DataFrame(X).astype("float32")


def _es_split(n: int, val_frac: float = 0.15, gap: Optional[int] = None):
    """Хронологический хвост под early stopping: последние ``val_frac`` строк как eval,
    с purge-зазором ``gap`` (по умолчанию HORIZON) между обучающей частью и eval —
    чтобы перекрывающиеся 24h-метки не утекали в критерий ранней остановки.
    Возвращает ``(fit_idx, val_idx)`` либо ``(arange(n), None)``, если данных мало."""
    gap = config.HORIZON if gap is None else int(gap)
    n_val = int(n * val_frac)
    if n_val < 50 or (n - n_val - gap) < 100:
        return np.arange(n), None
    val_idx = np.arange(n - n_val, n)
    fit_idx = np.arange(0, n - n_val - gap)
    return fit_idx, val_idx


def _fit_es(model, X, y, sample_weight, eval_metric: str):
    """Фит LightGBM c early stopping на purged-хвосте трейна (число деревьев выбирается
    по фолду, а не задаётся жёстко n_estimators). Вырожденный хвост (один класс) или
    слишком малая выборка -> обычный фит без ранней остановки."""
    Xp = _prep_X(X)
    y = np.asarray(y)
    fit_idx, val_idx = _es_split(len(Xp))
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if val_idx is None or np.unique(y[val_idx]).size < 2:
        model.fit(Xp, y, sample_weight=sw)
        return model
    model.fit(
        Xp.iloc[fit_idx], y[fit_idx],
        sample_weight=None if sw is None else sw[fit_idx],
        eval_set=[(Xp.iloc[val_idx], y[val_idx])],
        eval_sample_weight=None if sw is None else [sw[val_idx]],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model


class _ConstantProba:
    """Заглушка для вырожденного бинарного фолда (один класс в таргете)."""

    def __init__(self, p1: float):
        self.p1 = float(p1)

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p1), np.full(n, self.p1)])


def _fit_binary(X, y, sample_weight=None, params=None):
    """Бинарный LGBM с защитой от одно-классового фолда."""
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    if classes.size < 2:
        return _ConstantProba(float(y.mean()) if y.size else 0.0)
    m = _lgbm(params, objective="binary")
    return _fit_es(m, X, y, sample_weight, eval_metric="binary_logloss")


def _p1(model, X) -> np.ndarray:
    """P(class==1) из бинарной модели (или заглушки)."""
    proba = model.predict_proba(X)
    if isinstance(model, _ConstantProba):
        return proba[:, 1]
    # порядок колонок = model.classes_ (отсортированы); ищем класс 1
    classes = list(model.classes_)
    return proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def compose_sample_weight(y, horizon: int, halflife: Optional[int],
                          class_balance: bool = True) -> np.ndarray:
    """class_weight × average_uniqueness(overlap) × recency (R5).

    `y` определяет классовый компонент (мульти- или бинарный). uniqueness/recency —
    позиционные (предполагается хронологический порядок строк)."""
    y = np.asarray(y)
    n = y.size
    w = validation.average_uniqueness(n, horizon) * validation.recency_weights(n, halflife)
    if class_balance:
        vals, counts = np.unique(y[~pd.isna(y)], return_counts=True)
        freq = {v: c / counts.sum() for v, c in zip(vals, counts)}
        cw = np.array([1.0 / (len(vals) * freq[v]) if v in freq else 1.0 for v in y], dtype=float)
        w = w * cw
    return w


# ─────────────────────────────── base ──────────────────────────────────────
class _BaseSetup:
    name = "base"

    def __init__(self, params: Optional[dict] = None, horizon: Optional[int] = None,
                 halflife: Optional[int] = None):
        self.params = params or {}
        self.horizon = horizon if horizon is not None else config.HORIZON
        self.halflife = halflife if halflife is not None else config.RECENCY_HALFLIFE

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba3(X)
        return np.asarray(CLASSES)[proba.argmax(axis=1)]

    def signal_score(self, X) -> np.ndarray:
        proba = self.predict_proba3(X)
        return proba[:, 2] - proba[:, 0]


# ─────────────────────────────── A: softmax ────────────────────────────────
class SetupSoftmax(_BaseSetup):
    """A — одна LightGBM multiclass на 3 класса."""
    name = "A_softmax"

    def fit(self, X, y3, sample_weight=None, y_ret=None):
        y = np.asarray(y3).astype(int)
        if sample_weight is None:
            sample_weight = compose_sample_weight(y, self.horizon, self.halflife)
        ymap = y + 1  # {-1,0,1} -> {0,1,2}
        self.model_ = _lgbm(self.params, objective="multiclass", num_class=3)
        _fit_es(self.model_, X, ymap, sample_weight, eval_metric="multi_logloss")
        self.classes_map_ = list(self.model_.classes_)  # ожидаем [0,1,2]
        return self

    def predict_proba3(self, X) -> np.ndarray:
        proba = self.model_.predict_proba(X)
        # переупорядочить колонки в [P(-1),P(0),P(+1)] по classes_map_ (= 0,1,2)
        order = [self.classes_map_.index(c) for c in (0, 1, 2)]
        return proba[:, order]


# ─────────────────────────────── B: two binary ─────────────────────────────
class SetupTwoBinary(_BaseSetup):
    """B — две независимые бинарные (up / down), затем согласование в 3 класса."""
    name = "B_two_binary"

    def fit(self, X, y3, sample_weight=None, y_ret=None):
        y = np.asarray(y3).astype(int)
        y_up, y_dn = (y == 1).astype(int), (y == -1).astype(int)
        w_up = compose_sample_weight(y_up, self.horizon, self.halflife)
        w_dn = compose_sample_weight(y_dn, self.horizon, self.halflife)
        self.model_up_ = _fit_binary(X, y_up, w_up, self.params)
        self.model_down_ = _fit_binary(X, y_dn, w_dn, self.params)
        return self

    def predict_proba3(self, X) -> np.ndarray:
        p_up = _p1(self.model_up_, X)
        p_dn = _p1(self.model_down_, X)
        p_plus = p_up * (1 - p_dn)
        p_minus = (1 - p_up) * p_dn
        p_zero = (1 - p_up) * (1 - p_dn) + p_up * p_dn
        P = np.column_stack([p_minus, p_zero, p_plus])
        return P / P.sum(axis=1, keepdims=True).clip(min=1e-12)


# ─────────────────────────────── C: cascade ────────────────────────────────
class SetupCascade(_BaseSetup):
    """C — каскад gate -> direction. variant ∈ {'c1','c2'} (R2: c2 gate p_big OOF)."""
    name = "C_cascade"

    def __init__(self, variant: str = "c1", oof_splits: int = 3, **kw):
        super().__init__(**kw)
        if variant not in ("c1", "c2"):
            raise ValueError("variant must be 'c1' or 'c2'")
        self.variant = variant
        self.oof_splits = oof_splits
        self.name = f"C_cascade_{variant}"

    def _oof_pbig(self, X, y_big, w_big) -> np.ndarray:
        """Out-of-fold p_big: gate учится на под-фолдах, предсказывает на отложенных
        (R2 — без in-sample утечки в direction-веса)."""
        X = pd.DataFrame(X).reset_index(drop=True)
        y_big = np.asarray(y_big).astype(int)
        oof = np.full(len(X), np.nan)
        kf = KFold(n_splits=min(self.oof_splits, max(2, len(X) // 50)), shuffle=False)
        for tr, te in kf.split(X):
            m = _fit_binary(X.iloc[tr], y_big[tr], w_big[tr], self.params)
            oof[te] = _p1(m, X.iloc[te])
        oof[np.isnan(oof)] = np.nanmean(oof) if np.isfinite(oof).any() else 0.5
        return oof

    def fit(self, X, y3, sample_weight=None, y_ret=None):
        X = pd.DataFrame(X)
        y = np.asarray(y3).astype(int)
        y_big = (y != 0).astype(int)
        w_big = compose_sample_weight(y_big, self.horizon, self.halflife)
        self.gate_ = _fit_binary(X, y_big, w_big, self.params)

        if self.variant == "c1":
            # direction обучается на РЕАЛЬНОЙ big-популяции (|ret|>=thr)
            mask = y != 0
            Xd, yd = X[mask], (y[mask] == 1).astype(int)
            wd = compose_sample_weight(yd, self.horizon, self.halflife)
            self.direction_ = _fit_binary(Xd, yd, wd, self.params)
        else:
            # c2: вся популяция, вес = OOF p_big; направление по знаку доходности
            if y_ret is None:
                raise ValueError("cascade c2 requires y_ret (forward return) to label direction")
            pbig_oof = self._oof_pbig(X, y_big, w_big)
            yd = (np.asarray(y_ret, dtype=float) > 0).astype(int)
            base = validation.average_uniqueness(len(X), self.horizon) * \
                validation.recency_weights(len(X), self.halflife)
            self.direction_ = _fit_binary(X, yd, base * pbig_oof, self.params)
        return self

    def predict_proba3(self, X) -> np.ndarray:
        p_big = _p1(self.gate_, X)
        p_up = _p1(self.direction_, X)
        p_plus = p_big * p_up
        p_minus = p_big * (1 - p_up)
        p_zero = 1 - p_big
        P = np.column_stack([p_minus, p_zero, p_plus])
        return P / P.sum(axis=1, keepdims=True).clip(min=1e-12)


SETUPS = {
    "A": SetupSoftmax,
    "B": SetupTwoBinary,
    "C1": lambda **kw: SetupCascade(variant="c1", **kw),
    "C2": lambda **kw: SetupCascade(variant="c2", **kw),
}

"""Two-stage meta-labeling (EPIC 4) — López de Prado, *AFML* гл. 3.

Stage 1 (первичная модель, например `CryptoNetRegressor`/GBM) задаёт **сторону**
сделки `side = sign(r̂)`. Stage 2 (GBM-классификатор здесь) предсказывает
**вероятность, что ставка первичной модели верна** — это уверенность для сайзинга и
фильтрации сделок. Аддитивная коррекция величины (`r̂_A + r̂_B`) здесь НЕ выбрана —
только мета-лейблинг (см. оценку пайплайна §7).

Три обязательных условия, иначе двухэтапность = усиленная утечка:
  1. **OOF / cross-fit**: прогнозы Stage 1 для обучения Stage 2 берутся только из
     фолдов, где наблюдение было в test (вход `oof` = `wf_oof_all[freq][model]`).
  2. **Return-space**: Stage 1 предсказывает `r_t`; всё считается в доходностях.
  3. **Собственный КАУЗАЛЬНЫЙ walk-forward Stage 2**: meta обучается на
     расширяющемся окне строго из прошлого с purge/embargo = горизонт, НЕ на полном
     OOF с оценкой там же. Поэтому здесь НЕ используется `validation.PurgedKFold`
     (он некаузален — берёт будущие блоки в train): сгенерированные `p_ok` идут в
     каузальный бэктест, и любое подглядывание в будущее стало бы look-ahead.

Stage 2 — только GBM (LightGBM/CatBoost), не нейросеть: остаток первичной модели —
почти шум, и NN его переобучит (см. §4 «Чего НЕ делать»).
"""

from __future__ import annotations

from typing import Callable, Iterator, NamedTuple, Optional

import numpy as np
import pandas as pd

from metrics import rank_ic


# ---------------------------------------------------------------------------
# Каузальные purged-сплиты для OOF Stage 2
# ---------------------------------------------------------------------------

def expanding_purged_splits(
    n: int,
    n_splits: int = 5,
    min_train_frac: float = 0.3,
    embargo: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Каузальные расширяющиеся фолды с embargo (purge перекрытия меток).

    Train — всегда строго ИЗ ПРОШЛОГО относительно test (`max(train) < min(test)`),
    что обязательно для OOF-сигнала, уходящего в каузальный бэктест. Между train и
    test вырезается `embargo` баров (для горизонта h задавайте `embargo >= h`, чтобы
    убрать перекрытие многобаровых меток и автокорреляционную утечку).

    Первые `min_train_frac * n` баров служат начальным train и никогда не попадают в
    test (их `p_ok` останется неопределён -> в бэктесте size=0).
    """
    if not 0.0 < min_train_frac < 1.0:
        raise ValueError(f'min_train_frac must be in (0,1), got {min_train_frac}')
    if n_splits < 1:
        raise ValueError(f'n_splits must be >= 1, got {n_splits}')
    min_train = int(np.floor(min_train_frac * n))
    remaining = n - min_train
    test_size = remaining // n_splits
    if test_size < 1:
        raise ValueError(
            f'not enough rows for {n_splits} folds: n={n}, min_train_frac={min_train_frac}'
        )
    train_end = min_train
    for _ in range(n_splits):
        test_start = train_end
        test_end = min(test_start + test_size, n)
        if test_start >= n:
            break
        train_idx = np.arange(0, max(0, train_end - embargo))
        test_idx = np.arange(test_start, test_end)
        if train_idx.size and test_idx.size:
            yield train_idx, test_idx
        train_end = test_end
        if train_end >= n:
            break


# ---------------------------------------------------------------------------
# Дефолтные Stage-2 модели (lazy import — модуль остаётся лёгким без torch/catboost)
# ---------------------------------------------------------------------------

def _default_meta_clf(seed: int = 42):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )


def _default_meta_reg(seed: int = 42):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )


# ---------------------------------------------------------------------------
# Построение мета-датасета
# ---------------------------------------------------------------------------

class MetaDataset(NamedTuple):
    """Выровненный по `timestamp` мета-датасет Stage 2."""
    X: pd.DataFrame          # фичи (лагированные exog [+ r̂, |r̂|]), index = timestamp
    y: np.ndarray            # таргет meta: correct = 1[sign(r̂) == sign(r)]
    weight: np.ndarray       # sample_weight = |r| (магнитуда движения)
    side: np.ndarray         # ставка первичной модели sign(r̂) ∈ {-1,0,1}
    timestamp: np.ndarray    # временные метки (отсортированы)


def build_meta_dataset(
    oof: pd.DataFrame,
    features: pd.DataFrame,
    include_pred: bool = True,
) -> MetaDataset:
    """Построить мета-датасет из OOF первичной модели (return-space).

    Parameters
    ----------
    oof          : OOF одной первичной модели — колонки 'timestamp','actual'(r_t),
                   'predicted'(r̂_t). Берётся из `wf_oof_all[freq][model]`.
    features     : лагированные экзогенные фичи, индексированные по timestamp
                   (тот же индекс, что у `oof['timestamp']`).
    include_pred : добавить ли в X сам прогноз `r̂` и его модуль `|r̂|` (фичи
                   уверенности первичной модели).

    Таргет — `correct = 1[sign(r̂) == sign(r)]`, вес — `|r|`. Признаки выравниваются
    `features.reindex(timestamp)` (отсутствующие -> NaN; GBM обрабатывает нативно).
    """
    required = {'timestamp', 'actual', 'predicted'}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f'oof missing columns: {sorted(missing)}')

    o = oof.sort_values('timestamp')
    ts = o['timestamp'].to_numpy()
    actual = o['actual'].to_numpy(dtype=float)
    predicted = o['predicted'].to_numpy(dtype=float)

    side = np.sign(predicted)
    correct = (np.sign(actual) == side).astype(int)
    weight = np.abs(actual)

    X = features.reindex(o['timestamp']).copy()
    X.index = o['timestamp'].to_numpy()
    if include_pred:
        X['_pred'] = predicted
        X['_abs_pred'] = np.abs(predicted)

    return MetaDataset(X=X, y=correct, weight=weight, side=side, timestamp=ts)


# ---------------------------------------------------------------------------
# Каузальный walk-forward Stage 2 -> OOF p_ok
# ---------------------------------------------------------------------------

def train_meta_walkforward(
    meta: MetaDataset,
    horizon: int = 1,
    n_splits: int = 5,
    min_train_frac: float = 0.3,
    embargo: Optional[int] = None,
    model_factory: Optional[Callable[[int], object]] = None,
    seed: int = 42,
) -> np.ndarray:
    """Каузальный walk-forward Stage 2 поверх OOF -> вероятности `p_ok` (тоже OOF).

    На каждом расширяющемся фолде Stage-2-классификатор обучается на ПРОШЛЫХ
    наблюдениях (с embargo перед test) и предсказывает `P(correct)` на test-блоке.
    Никакого фита на полном OOF с оценкой там же. Возвращает массив `p_ok` длины
    `len(meta.y)`; для начального train-периода (и любых не покрытых баров) — NaN
    (в бэктесте -> size=0).

    `embargo` по умолчанию = `max(horizon, 1)`; задавайте `>= horizon` при h>1.
    `model_factory(seed) -> estimator` с интерфейсом `.fit(X, y, sample_weight)` и
    `.predict_proba(X)`; по умолчанию LightGBM.
    """
    if embargo is None:
        embargo = max(horizon, 1)
    factory = model_factory or _default_meta_clf

    n = len(meta.y)
    p_ok = np.full(n, np.nan, dtype=float)

    for train_idx, test_idx in expanding_purged_splits(
        n, n_splits=n_splits, min_train_frac=min_train_frac, embargo=embargo
    ):
        Xtr, ytr, wtr = meta.X.iloc[train_idx], meta.y[train_idx], meta.weight[train_idx]
        Xte = meta.X.iloc[test_idx]

        # Нужны оба класса в train; иначе откатываемся на base-rate train-доли.
        if np.unique(ytr).size < 2:
            p_ok[test_idx] = float(ytr.mean()) if ytr.size else 0.5
            continue

        model = factory(seed)
        model.fit(Xtr, ytr, sample_weight=wtr)
        proba = model.predict_proba(Xte)
        # столбец класса «1» (correct)
        p1 = proba[:, 1] if getattr(proba, 'ndim', 1) == 2 else np.asarray(proba)
        p_ok[test_idx] = p1

    return p_ok


def meta_size(side: np.ndarray, p_ok: np.ndarray) -> np.ndarray:
    """Непрерывный размер позиции из стороны и уверенности Stage 2.

    ``size = side · clip(2·p_ok − 1, 0, 1)`` ∈ [−1, 1]: 0 = не торговать
    (`p_ok <= 0.5` или `p_ok` неопределён), (0,1] = доля позиции в сторону `side`.
    """
    side = np.asarray(side, dtype=float)
    p = np.asarray(p_ok, dtype=float)
    conf = np.clip(2.0 * p - 1.0, 0.0, 1.0)
    size = side * conf
    return np.where(np.isfinite(p), size, 0.0)


# ---------------------------------------------------------------------------
# Sanity-gate: есть ли вообще структурный остаток (T4.5)
# ---------------------------------------------------------------------------

def residual_sanity_gate(
    oof: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int = 1,
    n_splits: int = 5,
    min_train_frac: float = 0.3,
    embargo: Optional[int] = None,
    model_factory: Optional[Callable[[int], object]] = None,
    seed: int = 42,
) -> float:
    """OOS Rank IC лёгкого GBM-регрессора остатка `e = r − r̂` против фич.

    Проверка осмысленности ДО внедрения мета-этапа: если OOS Rank IC ≈ 0 —
    структурного остатка нет, и Stage 2 не даст сигнала (логировать предупреждение).
    Считается тем же каузальным walk-forward, что и `train_meta_walkforward`.
    """
    if embargo is None:
        embargo = max(horizon, 1)
    factory = model_factory or _default_meta_reg

    o = oof.sort_values('timestamp')
    e = (o['actual'].to_numpy(dtype=float) - o['predicted'].to_numpy(dtype=float))
    X = features.reindex(o['timestamp']).copy()
    X.index = o['timestamp'].to_numpy()

    n = len(e)
    pred = np.full(n, np.nan, dtype=float)
    for train_idx, test_idx in expanding_purged_splits(
        n, n_splits=n_splits, min_train_frac=min_train_frac, embargo=embargo
    ):
        model = factory(seed)
        model.fit(X.iloc[train_idx], e[train_idx])
        pred[test_idx] = np.asarray(model.predict(X.iloc[test_idx]), dtype=float)

    mask = np.isfinite(pred)
    if mask.sum() < 2:
        return float('nan')
    return rank_ic(e[mask], pred[mask])

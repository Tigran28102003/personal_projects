"""Тесты валидаторов с purge+embargo (EPIC 2, T2.5).

Проверяем: нет пересечения train/test; тестовые блоки покрывают все наблюдения;
embargo соблюдён; для horizon>1 purge перекрытий корректен; cpcv даёт C(N,k)
путей без утечки. Запуск: `pytest -q`.
"""

import numpy as np
import pandas as pd
import pytest

from validation import PurgedKFold, cpcv_splits, n_cpcv_paths


# ==================== PurgedKFold ====================

def test_purgedkfold_no_train_test_overlap():
    pkf = PurgedKFold(n_splits=5, embargo=3, horizon=1)
    X = np.zeros((200, 4))
    for tr, te in pkf.split(X):
        assert set(tr).isdisjoint(set(te))


def test_purgedkfold_test_blocks_partition_all():
    pkf = PurgedKFold(n_splits=5, embargo=2, horizon=1)
    n = 203
    X = np.zeros((n, 2))
    test_union = np.concatenate([te for _, te in pkf.split(X)])
    # тестовые блоки непрерывны, непересекающиеся и покрывают все наблюдения
    np.testing.assert_array_equal(np.sort(test_union), np.arange(n))
    assert len(test_union) == len(set(test_union.tolist()))


def test_purgedkfold_embargo_respected():
    embargo = 5
    pkf = PurgedKFold(n_splits=4, embargo=embargo, horizon=1)
    n = 200
    X = np.zeros((n, 2))
    for tr, te in pkf.split(X):
        b = int(te.max()) + 1
        forbidden = set(range(b, min(n, b + embargo)))
        assert forbidden.isdisjoint(set(tr.tolist()))


def test_purgedkfold_purge_for_horizon():
    h = 4
    pkf = PurgedKFold(n_splits=4, embargo=0, horizon=h)
    n = 200
    X = np.zeros((n, 2))
    for tr, te in pkf.split(X):
        a = int(te.min())
        b = int(te.max()) + 1
        # purge слева [a-h+1, a) и справа [b, b+h-1) — не должно быть train-индексов
        left = set(range(max(0, a - h + 1), a))
        right = set(range(b, min(n, b + h - 1)))
        assert left.isdisjoint(set(tr.tolist()))
        assert right.isdisjoint(set(tr.tolist()))


def test_purgedkfold_accepts_dataframe_and_get_n_splits():
    pkf = PurgedKFold(n_splits=3, embargo=1, horizon=1)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 5)))
    folds = list(pkf.split(df))
    assert len(folds) == 3 == pkf.get_n_splits()
    for tr, te in folds:
        assert set(tr).isdisjoint(set(te))


def test_purgedkfold_validates_args():
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=3, horizon=0)
    with pytest.raises(ValueError):
        list(PurgedKFold(n_splits=10).split(np.zeros((5, 2))))


# ==================== cpcv_splits ====================

def test_cpcv_path_count():
    n = 300
    paths = list(cpcv_splits(n, N=6, k=2, embargo_frac=0.01, horizon=1))
    assert len(paths) == n_cpcv_paths(6, 2) == 15


def test_cpcv_no_overlap_and_purge():
    n = 300
    h = 3
    for tr, te in cpcv_splits(n, N=6, k=2, embargo_frac=0.01, horizon=h):
        tr_set, te_set = set(tr.tolist()), set(te.tolist())
        assert tr_set.isdisjoint(te_set)
        # для каждой непрерывной тест-подгруппы соседние бары (purge+embargo) не в train
        emb = max(h, int(np.ceil(0.01 * n)))
        # граница справа от максимального тестового индекса
        b = int(te.max()) + 1
        forbidden = set(range(b, min(n, b + (h - 1) + emb)))
        # часть forbidden может быть другой тест-группой; train в неё попадать не должен
        assert forbidden.isdisjoint(tr_set)


def test_cpcv_all_groups_used_in_test():
    n = 300
    seen = set()
    for _, te in cpcv_splits(n, N=6, k=2):
        seen.update(te.tolist())
    # объединение всех тест-путей покрывает все наблюдения
    assert seen == set(range(n))


def test_cpcv_validates_args():
    with pytest.raises(ValueError):
        list(cpcv_splits(100, N=6, k=6))   # k must be < N
    with pytest.raises(ValueError):
        list(cpcv_splits(3, N=6, k=2))     # n < N

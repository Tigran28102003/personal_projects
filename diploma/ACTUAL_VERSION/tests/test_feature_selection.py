"""T7.5 — стабильный кросс-фолдовый отбор фич (медиана MDA по dev-фолдам) + CFI."""

import numpy as np
import pandas as pd
import pytest

from walk_forward import (
    _feature_importance, cluster_collinear, select_stable_features,
    rolling_window_splits,
)


# --------------------------------------------------------------------------
# _feature_importance
# --------------------------------------------------------------------------

@pytest.mark.parametrize('method', ['pearson', 'mutual_info', 'model_gain', 'mda'])
def test_feature_importance_returns_series(method):
    rng = np.random.default_rng(0)
    n = 200
    f0 = rng.normal(size=n)
    X = pd.DataFrame({'f0': f0, 'f1': rng.normal(size=n), 'f2': rng.normal(size=n)})
    y = pd.Series(f0 ** 2 + 0.05 * rng.normal(size=n))
    s = _feature_importance(X, y, method=method)
    assert isinstance(s, pd.Series)
    assert set(s.index) <= set(X.columns)


# --------------------------------------------------------------------------
# cluster_collinear (CFI)
# --------------------------------------------------------------------------

def test_cluster_collinear_groups_correlated():
    corr = pd.DataFrame(
        [[1.0, 0.99, 0.10], [0.99, 1.0, 0.05], [0.10, 0.05, 1.0]],
        index=['a', 'b', 'c'], columns=['a', 'b', 'c'],
    ).abs()
    clusters = [sorted(cl) for cl in cluster_collinear(corr, threshold=0.9)]
    assert ['a', 'b'] in clusters      # коллинеарные вместе
    assert ['c'] in clusters           # независимая — отдельно


def test_cluster_collinear_all_independent():
    corr = pd.DataFrame(np.eye(3), index=['a', 'b', 'c'], columns=['a', 'b', 'c'])
    clusters = cluster_collinear(corr, threshold=0.9)
    assert len(clusters) == 3          # каждая фича — свой кластер


# --------------------------------------------------------------------------
# select_stable_features
# --------------------------------------------------------------------------

@pytest.fixture
def redundant_frame():
    rng = np.random.default_rng(1)
    n = 600
    idx = pd.date_range('2020-01-01', periods=n, freq='D')
    f_good = rng.normal(size=n)
    f_dup = f_good + 0.5 * rng.normal(size=n)          # коллинеарна f_good, но шумнее
    frame = pd.DataFrame({
        'y': 2.0 * f_good + 0.1 * rng.normal(size=n),  # сигнал только в f_good
        'f_good': f_good,
        'f_dup': f_dup,
        'f_noise1': rng.normal(size=n),
        'f_noise2': rng.normal(size=n),
    }, index=idx)
    return frame


def test_stable_selection_collapses_collinear_and_drops_noise(redundant_frame):
    frame = redundant_frame
    splits = rolling_window_splits(len(frame), train_size=150, test_size=50)
    sel = select_stable_features(frame, ['f_good', 'f_dup', 'f_noise1', 'f_noise2'],
                                 'y', splits, k=3, method='mda', corr_threshold=0.8)
    # CFI схлопнул коллинеарную пару -> ровно одна из {f_good, f_dup}
    assert len({'f_good', 'f_dup'} & set(sel)) == 1
    # из пары осталась более важная (f_good)
    assert 'f_good' in sel and 'f_dup' not in sel
    # ведущая фича — f_good
    assert sel[0] == 'f_good'


def test_stable_selection_respects_k(redundant_frame):
    splits = rolling_window_splits(len(redundant_frame), train_size=150, test_size=50)
    sel = select_stable_features(redundant_frame, ['f_good', 'f_dup', 'f_noise1', 'f_noise2'],
                                 'y', splits, k=1, method='mda', corr_threshold=0.8)
    assert len(sel) == 1 and sel[0] == 'f_good'


def test_stable_selection_small_pool_returns_all():
    rng = np.random.default_rng(2)
    n = 200
    frame = pd.DataFrame({'y': rng.normal(size=n), 'a': rng.normal(size=n), 'b': rng.normal(size=n)},
                         index=pd.date_range('2020-01-01', periods=n, freq='D'))
    splits = rolling_window_splits(n, train_size=80, test_size=40)
    sel = select_stable_features(frame, ['a', 'b'], 'y', splits, k=5)
    assert set(sel) == {'a', 'b'}      # фич меньше k -> возвращаем все


def test_stable_selection_uses_only_dev_prefix(redundant_frame, monkeypatch):
    # leakage-safe: важность считается только на dev-префиксе фолдов
    splits = rolling_window_splits(len(redundant_frame), train_size=150, test_size=50)
    seen_starts = []
    import walk_forward as wf
    orig = wf._feature_importance

    def spy(X, y, method='pearson', random_state=0):
        seen_starts.append(int(X.index.min().value))   # запоминаем, какие фолды видели
        return orig(X, y, method=method, random_state=random_state)

    monkeypatch.setattr(wf, '_feature_importance', spy)
    wf.select_stable_features(redundant_frame, ['f_good', 'f_dup', 'f_noise1', 'f_noise2'],
                              'y', splits, k=2, dev_frac=0.5, method='pearson')
    # последний dev-фолд (dev_frac=0.5) начинается раньше, чем последние тест-фолды
    last_dev_start = redundant_frame.index[splits[len(splits) // 2 - 1][0][0]].value
    assert max(seen_starts) <= last_dev_start

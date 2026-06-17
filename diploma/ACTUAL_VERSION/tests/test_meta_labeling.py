"""EPIC 4 — мета-лейблинг: OOF-дисциплина, сайзинг, построение датасета."""

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from meta_labeling import (
    build_meta_dataset,
    expanding_purged_splits,
    train_meta_walkforward,
    meta_size,
    residual_sanity_gate,
)


# --------------------------------------------------------------------------
# build_meta_dataset
# --------------------------------------------------------------------------

def test_build_meta_dataset_labels_weights_side():
    oof = pd.DataFrame({
        'timestamp': [0, 1, 2, 3],
        'actual':    [0.02, -0.01, 0.0, 0.03],
        'predicted': [0.01, 0.02, -0.01, 0.04],
    })
    features = pd.DataFrame({'f0': [1.0, 2.0, 3.0, 4.0]}, index=[0, 1, 2, 3])
    md = build_meta_dataset(oof, features, include_pred=True)
    # side = sign(pred); correct = 1[sign(actual)==side]; weight = |actual|
    assert list(md.side) == [1, 1, -1, 1]
    assert list(md.y) == [1, 0, 0, 1]
    assert np.allclose(md.weight, [0.02, 0.01, 0.0, 0.03])
    assert {'_pred', '_abs_pred'} <= set(md.X.columns)
    assert md.X.shape[0] == 4


def test_build_meta_dataset_sorts_by_timestamp():
    oof = pd.DataFrame({
        'timestamp': [2, 0, 1],
        'actual':    [0.03, 0.01, -0.02],
        'predicted': [0.01, 0.01, 0.01],
    })
    features = pd.DataFrame({'f0': [10.0, 20.0, 30.0]}, index=[0, 1, 2])
    md = build_meta_dataset(oof, features, include_pred=False)
    assert list(md.timestamp) == [0, 1, 2]
    # фичи выровнены по отсортированному timestamp
    assert list(md.X['f0']) == [10.0, 20.0, 30.0]


def test_build_meta_dataset_requires_columns():
    with pytest.raises(ValueError):
        build_meta_dataset(pd.DataFrame({'timestamp': [0], 'actual': [0.0]}),
                           pd.DataFrame({'f0': [1.0]}, index=[0]))


# --------------------------------------------------------------------------
# expanding_purged_splits — каузальность + embargo
# --------------------------------------------------------------------------

def test_expanding_splits_causal_and_embargo():
    embargo = 2
    splits = list(expanding_purged_splits(100, n_splits=4, min_train_frac=0.4, embargo=embargo))
    assert splits
    for tr, te in splits:
        assert tr.max() < te.min()                  # train строго в прошлом
        assert te.min() - 1 - tr.max() == embargo    # ровно `embargo` баров вырезано
        assert np.intersect1d(tr, te).size == 0      # нет пересечения
    assert splits[0][1].min() == 40                  # test начинается после min_train


def test_expanding_splits_validates_args():
    with pytest.raises(ValueError):
        list(expanding_purged_splits(100, min_train_frac=1.5))


# --------------------------------------------------------------------------
# train_meta_walkforward — OOF-дисциплина (никакого будущего в train)
# --------------------------------------------------------------------------

def test_meta_walkforward_oof_discipline():
    n = 120
    rng = np.random.default_rng(0)
    ts = np.arange(n)
    actual = rng.normal(scale=0.01, size=n)
    predicted = 0.5 * actual + rng.normal(scale=0.005, size=n)
    oof = pd.DataFrame({'timestamp': ts, 'actual': actual, 'predicted': predicted})
    feat = pd.DataFrame({'f0': rng.normal(size=n), 'f1': rng.normal(size=n)}, index=ts)
    md = build_meta_dataset(oof, feat)

    captured = []   # (train_range, test_range) для каждого фолда

    def factory(seed):
        class _M:
            def fit(self, X, y, sample_weight=None):
                self._tr = (int(X.index.min()), int(X.index.max()))
                return self
            def predict_proba(self, X):
                captured.append((self._tr, (int(X.index.min()), int(X.index.max()))))
                return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])
        return _M()

    embargo = 2
    p_ok = train_meta_walkforward(md, n_splits=4, min_train_frac=0.3,
                                  embargo=embargo, model_factory=factory)
    assert p_ok.shape == (n,)
    # начальный train-период не покрыт OOF-прогнозом
    min_train = int(np.floor(0.3 * n))
    assert np.isnan(p_ok[:min_train]).all()
    assert np.isfinite(p_ok[min_train:]).any()
    # каждый train строго предшествует своему test с зазором embargo
    assert captured
    for (tr_min, tr_max), (te_min, te_max) in captured:
        assert tr_max < te_min
        assert te_min - 1 - tr_max == embargo


# --------------------------------------------------------------------------
# meta_size
# --------------------------------------------------------------------------

def test_meta_size_zero_below_half_and_nan():
    side = np.array([1, 1, -1, 1, 1])
    p_ok = np.array([0.3, 0.5, 0.8, 0.7, np.nan])
    size = meta_size(side, p_ok)
    # conf = clip(2p-1,0,1) = [0,0,0.6,0.4,(nan->0)]; size = side*conf
    assert np.allclose(size, [0.0, 0.0, -0.6, 0.4, 0.0])


def test_meta_size_in_unit_interval():
    rng = np.random.default_rng(1)
    side = np.sign(rng.normal(size=200))
    p_ok = rng.uniform(0, 1, size=200)
    size = meta_size(side, p_ok)
    assert np.all(np.abs(size) <= 1.0 + 1e-9)


# --------------------------------------------------------------------------
# residual_sanity_gate
# --------------------------------------------------------------------------

def _reg_factory(seed):
    return LinearRegression()


def test_sanity_gate_detects_no_structure():
    n = 200
    rng = np.random.default_rng(2)
    ts = np.arange(n)
    feat = pd.DataFrame({'f0': rng.normal(size=n)}, index=ts)
    actual = rng.normal(scale=0.01, size=n)
    predicted = rng.normal(scale=0.01, size=n)   # остаток = шум, не связан с f0
    oof = pd.DataFrame({'timestamp': ts, 'actual': actual, 'predicted': predicted})
    ic = residual_sanity_gate(oof, feat, model_factory=_reg_factory)
    assert np.isfinite(ic) and abs(ic) < 0.4   # ~0 -> структурного остатка нет


def test_sanity_gate_detects_structure():
    n = 200
    rng = np.random.default_rng(3)
    ts = np.arange(n)
    f0 = rng.normal(size=n)
    feat = pd.DataFrame({'f0': f0}, index=ts)
    actual = rng.normal(scale=0.01, size=n)
    # остаток e = actual - predicted = 3*f0 + tiny -> восстановим линейно
    predicted = actual - (3.0 * f0 + rng.normal(scale=0.01, size=n))
    oof = pd.DataFrame({'timestamp': ts, 'actual': actual, 'predicted': predicted})
    ic = residual_sanity_gate(oof, feat, model_factory=_reg_factory)
    assert ic > 0.5

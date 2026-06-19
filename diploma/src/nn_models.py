"""Neural-net learners for the A (softmax) formulation — run in a SEPARATE process.

torch and LightGBM must NOT share a process (macOS duplicate-libomp segfault), so this
module is imported only by ``run_nn_comparison.py`` (a subprocess); **torch is imported
lazily** inside functions. The aim is to compare NN as a *learner* inside formulation A
on the SAME purged-WF folds and the SAME ``metrics.evaluate`` — isolating the learner
effect, exactly like §10 for the trees.

Unified interface (identical to ``models.SetupSoftmax``):
``fit(X, y3, sample_weight=None, y_ret=None)`` · ``predict_proba3(X) -> (n,3)=[P(-1),
P(0),P(+1)]`` · ``predict`` (argmax→{-1,0,+1}) · ``signal_score = P(+1)-P(-1)``.

Two data views: a **tabular MLP** on the same X (apples-to-apples with LightGBM) and
**sequence GRU/LSTM/TCN** on strictly-trailing causal windows of raw channels (TCN uses
causal padding). Aggressive regularisation (small nets, dropout, weight decay, early
stopping on a purged inner-val) — few independent 24h labels ⇒ NN overfit easily, GBMs
usually win at this scale.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

CLASSES = (-1, 0, 1)
SEQ_CHANNELS = ("ret", "log_volume", "rsi", "macd_hist", "atr_close", "vol24")


# ─────────────────────────────── device / loss ─────────────────────────────
def pick_device(force_cpu: bool = False):
    """cuda > mps > cpu, with ``NN_DEVICE`` env override. Local macOS smoke forces CPU
    (V6: MPS has op-coverage/determinism gaps for RNN/TCN)."""
    import torch
    env = os.environ.get("NN_DEVICE", "")
    if force_cpu or env == "cpu":
        return torch.device("cpu")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_determinism(seed: int = 42, fast: bool = False):
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not fast:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    elif hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True


def _focal_ce(logits, target, class_weight, gamma: float = 1.5):
    """Focal cross-entropy с классовыми весами — ПО-САМПЛОВО (без усреднения), чтобы
    взвесить ещё и sample_weight (uniqueness×recency) снаружи."""
    import torch.nn.functional as F
    logp = F.log_softmax(logits, dim=1)
    pt = logp.gather(1, target[:, None]).squeeze(1).exp()
    nll = -logp.gather(1, target[:, None]).squeeze(1)
    cw = class_weight[target] if class_weight is not None else 1.0
    return cw * (1 - pt) ** gamma * nll                      # per-sample loss


# ─────────────────────────────── architectures ─────────────────────────────
def _mlp(in_dim, hidden, dropout):
    import torch.nn as nn
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, 3)]
    return nn.Sequential(*layers)


def _make_rnn(rnn_type, n_ch, hidden, dropout):
    import torch.nn as nn

    class RNNNet(nn.Module):
        def __init__(self):
            super().__init__()
            rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
            self.rnn = rnn(n_ch, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 3))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])      # last hidden -> dense -> softmax
    return RNNNet()


def _make_tcn(n_ch, channels, dropout, levels=3, k=3):
    import torch
    import torch.nn as nn

    class CausalConv1d(nn.Module):
        def __init__(self, ci, co, dilation):
            super().__init__()
            self.pad = (k - 1) * dilation                # left-pad only -> causal
            self.conv = nn.Conv1d(ci, co, k, dilation=dilation)

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (self.pad, 0)))

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks, ci = [], n_ch
            for i in range(levels):
                blocks += [CausalConv1d(ci, channels, 2 ** i), nn.ReLU(), nn.Dropout(dropout)]
                ci = channels
            self.net = nn.Sequential(*blocks)
            self.head = nn.Linear(channels, 3)

        def forward(self, x):                            # x: (B, L, C) -> (B, C, L)
            h = self.net(x.transpose(1, 2))
            return self.head(h[:, :, -1])                # last (most recent) timestep
    return TCN()


# ─────────────────────────────── sequence builder ──────────────────────────
def make_sequences(X: pd.DataFrame, lookback: int, channels: Sequence[str]):
    """(n, L, C) строго-трейлинговые окна [t-L+1 .. t]. Возвращает (tensor_array,
    valid_mask) — первые L-1 строк не имеют полного окна (mask=False)."""
    chans = [c for c in channels if c in X.columns] or list(X.columns[:4])
    arr = X[chans].to_numpy(dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0)
    n, c = arr.shape
    seq = np.zeros((n, lookback, c), dtype=np.float32)
    for L in range(lookback):
        seq[L:, lookback - 1 - L, :] = arr[: n - L]      # shift past bars into the window
    valid = np.zeros(n, dtype=bool); valid[lookback - 1:] = True
    return seq, valid, chans


# ─────────────────────────────── training core ─────────────────────────────
def _train_loop(model, Xtr, ytr, wtr, Xval, yval, device, *, epochs, lr, weight_decay,
                batch, fast, patience=8, seed=42):
    import torch
    from sklearn.metrics import matthews_corrcoef
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    cls_w = torch.tensor([1.0 / max(1, (ytr == c).sum()) for c in range(3)], dtype=torch.float32)
    cls_w = (cls_w / cls_w.sum() * 3).to(device)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    wtr_t = torch.tensor(wtr, dtype=torch.float32)
    Xval_t = torch.tensor(Xval, dtype=torch.float32).to(device)
    use_amp = fast and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    best_mcc, best_state, bad = -2.0, None, 0
    n = len(Xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xtr_t[idx].to(device); yb = ytr_t[idx].to(device); wb = wtr_t[idx].to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                ps = _focal_ce(model(xb), yb, cls_w)          # per-sample focal loss
                loss = (ps * wb).sum() / (wb.sum() + 1e-8)     # weight by uniqueness×recency
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            vp = model(Xval_t).argmax(1).cpu().numpy()
        mcc = matthews_corrcoef(yval, vp) if len(np.unique(yval)) > 1 else 0.0
        if mcc > best_mcc:
            best_mcc, best_state, bad = mcc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mcc


# ─────────────────────────────── learner setups ────────────────────────────
class _NNBase:
    name = "nn"

    def predict(self, X):
        return np.asarray(CLASSES)[self.predict_proba3(X).argmax(1)]

    def signal_score(self, X):
        p = self.predict_proba3(X)
        return p[:, 2] - p[:, 0]


class NNMlp(_NNBase):
    """Табличный MLP на тех же фичах X (apples-to-apples с LightGBM)."""
    name = "MLP"

    def __init__(self, hidden=(96, 48), dropout=0.4, lr=1e-3, weight_decay=1e-4,
                 epochs=60, batch=512, force_cpu=True, fast=False, seed=42, val_frac=0.15):
        self.__dict__.update(locals()); del self.self

    def _prep_fit(self, X):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        self.imp_ = SimpleImputer(strategy="median").fit(X)
        self.sc_ = StandardScaler().fit(self.imp_.transform(X))
        return self.sc_.transform(self.imp_.transform(X)).astype(np.float32)

    def _prep(self, X):
        return self.sc_.transform(self.imp_.transform(X)).astype(np.float32)

    def fit(self, X, y3, sample_weight=None, y_ret=None):
        set_determinism(self.seed, self.fast)
        dev = pick_device(self.force_cpu)
        X = pd.DataFrame(X); y = np.asarray(y3).astype(int) + 1
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        Xs = self._prep_fit(X)
        nval = max(32, int(self.val_frac * len(Xs)))
        tr = slice(0, len(Xs) - nval); va = slice(len(Xs) - nval, len(Xs))   # chronological inner-val
        self.model_ = _mlp(Xs.shape[1], self.hidden, self.dropout)
        self.model_, self.val_mcc_ = _train_loop(
            self.model_, Xs[tr], y[tr], w[tr], Xs[va], y[va], dev,
            epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
            batch=self.batch, fast=self.fast, seed=self.seed)
        self.device_ = dev
        return self

    def predict_proba3(self, X):
        import torch
        Xs = torch.tensor(self._prep(pd.DataFrame(X)), dtype=torch.float32).to(self.device_)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(Xs), dim=1).cpu().numpy()


class NNSeq(_NNBase):
    """Sequence GRU/LSTM/TCN на causal-окнах сырых каналов. Первые L-1 строк фолда без
    полного окна -> для них P = классовый prior (uniform-ish)."""
    def __init__(self, kind="tcn", lookback=48, hidden=48, dropout=0.3, lr=1e-3,
                 weight_decay=1e-4, epochs=50, batch=256, channels=SEQ_CHANNELS,
                 force_cpu=True, fast=False, seed=42, val_frac=0.15):
        self.__dict__.update(locals()); del self.self
        self.name = {"tcn": "TCN", "gru": "GRU", "lstm": "LSTM"}[kind]

    def _build(self, n_ch):
        if self.kind == "tcn":
            return _make_tcn(n_ch, self.hidden, self.dropout)
        return _make_rnn(self.kind, n_ch, self.hidden, self.dropout)

    def fit(self, X, y3, sample_weight=None, y_ret=None):
        from sklearn.preprocessing import StandardScaler
        set_determinism(self.seed, self.fast)
        dev = pick_device(self.force_cpu)
        X = pd.DataFrame(X); y = np.asarray(y3).astype(int) + 1
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        seq, valid, self.chans_ = make_sequences(X, self.lookback, self.channels)
        # standardise per-channel on the (valid) training rows
        flat = seq[valid].reshape(-1, seq.shape[2])
        self.sc_ = StandardScaler().fit(flat)
        seq = self.sc_.transform(seq.reshape(-1, seq.shape[2])).reshape(seq.shape).astype(np.float32)
        vi = np.where(valid)[0]
        nval = max(32, int(self.val_frac * len(vi)))
        tr_i, va_i = vi[:-nval], vi[-nval:]
        self.prior_ = np.bincount(y, minlength=3) / len(y)
        self.model_ = self._build(seq.shape[2])
        self.model_, self.val_mcc_ = _train_loop(
            self.model_, seq[tr_i], y[tr_i], w[tr_i], seq[va_i], y[va_i], dev,
            epochs=self.epochs, lr=self.lr, weight_decay=self.weight_decay,
            batch=self.batch, fast=self.fast, seed=self.seed)
        self.device_ = dev
        return self

    def predict_proba3(self, X):
        import torch
        X = pd.DataFrame(X)
        seq, valid, _ = make_sequences(X, self.lookback, self.chans_)
        seq = self.sc_.transform(seq.reshape(-1, seq.shape[2])).reshape(seq.shape).astype(np.float32)
        out = np.tile(self.prior_, (len(X), 1)).astype(np.float32)   # prior for warmup rows
        vi = np.where(valid)[0]
        if len(vi):
            self.model_.eval()
            with torch.no_grad():
                xb = torch.tensor(seq[vi], dtype=torch.float32).to(self.device_)
                out[vi] = torch.softmax(self.model_(xb), dim=1).cpu().numpy()
        return out


def build_learners(force_cpu=True, fast=False, quick=False):
    """NN-обучатели для learner-comparison (постановка A). quick -> меньше эпох."""
    ep_mlp, ep_seq = (12, 10) if quick else (60, 50)
    return {
        "MLP": NNMlp(force_cpu=force_cpu, fast=fast, epochs=ep_mlp),
        "GRU": NNSeq(kind="gru", force_cpu=force_cpu, fast=fast, epochs=ep_seq),
        "TCN": NNSeq(kind="tcn", force_cpu=force_cpu, fast=fast, epochs=ep_seq),
    }

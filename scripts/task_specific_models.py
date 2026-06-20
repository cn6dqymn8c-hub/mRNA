#!/usr/bin/env python3
"""
Task-specific (purpose-built) mRNA-localization architectures, trained FROM
SCRATCH on the same data/split/labels as the foundation-model runs.

These are compact re-implementations of two well-known localization model
families — they are NOT the published weights (which target different
compartment schemes / species) and are meant as controlled benchmark baselines:

  rnatracker : CNN + BiLSTM + attention pooling   (Yan et al. 2019 style)
  dm3loc     : conv embedding + multi-head self-attention + attention pooling
               (Wang et al. 2021 style, multi-label)

Both consume one-hot RNA sequence (A,C,G,U), are trained with the SAME masked
BCE + pos_weight + per-sample weights + per-label mask as finetune_model, do
early stopping on validation macro-AUC, and return per-transcript probabilities
for val/test. This keeps the comparison "same data/split/head-objective, only the
encoder differs".

Interface mirrors finetune_model:
    train_task_specific(genes, Y, classes, tr_idx, va_idx, te_idx, args, arch,
                        sample_weight=None, label_mask=None) -> (prob_va, prob_te)
"""
from __future__ import annotations

import numpy as np

_BASE2IDX = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}


def _one_hot(seq: str, max_len: int) -> np.ndarray:
    """(4, max_len) one-hot. Sequence is taken 5'->3' and padded/truncated to
    max_len. N / unknown -> all-zero column."""
    x = np.zeros((4, max_len), dtype=np.float32)
    s = str(seq)[:max_len]
    for i, ch in enumerate(s):
        j = _BASE2IDX.get(ch)
        if j is not None:
            x[j, i] = 1.0
    return x


def _build_models(arch, n_classes, max_len):
    import torch.nn as nn

    class AttnPool(nn.Module):
        """Additive attention pooling over the length axis. Input (B, T, H)."""
        def __init__(self, h):
            super().__init__()
            self.w = nn.Linear(h, 1)

        def forward(self, x, mask=None):
            scores = self.w(x).squeeze(-1)            # (B, T)
            if mask is not None:
                scores = scores.masked_fill(~mask, -1e9)
            attn = scores.softmax(dim=1).unsqueeze(-1)  # (B, T, 1)
            return (x * attn).sum(dim=1)               # (B, H)

    class RNATracker(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(4, 32, 10, padding=5), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(3),
                nn.Conv1d(32, 32, 10, padding=5), nn.ReLU(), nn.Dropout(0.2), nn.MaxPool1d(3),
            )
            self.lstm = nn.LSTM(32, 32, batch_first=True, bidirectional=True)
            self.pool = AttnPool(64)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(64, n_classes))

        def forward(self, x):                 # x: (B, 4, L)
            h = self.conv(x).transpose(1, 2)  # (B, L', 32)
            h, _ = self.lstm(h)               # (B, L', 64)
            return self.head(self.pool(h))

    class DM3Loc(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(
                nn.Conv1d(4, 64, 7, padding=3), nn.ReLU(),
                nn.MaxPool1d(4),                       # shrink length before attention
            )
            self.attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(64)
            self.pool = AttnPool(64)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
                                      nn.Linear(64, n_classes))

        def forward(self, x):                 # x: (B, 4, L)
            h = self.embed(x).transpose(1, 2)  # (B, L'', 64)
            a, _ = self.attn(h, h, h)
            h = self.norm(h + a)
            return self.head(self.pool(h))

    return RNATracker() if arch == "rnatracker" else DM3Loc()


def train_task_specific(
    genes, Y, classes, tr_idx, va_idx, te_idx, args, arch,
    sample_weight=None, label_mask=None,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score

    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    max_len = int(getattr(args, "ts_max_len", 4000))
    nC = len(classes)
    seqs = genes["sequence"].tolist()

    Y = np.asarray(Y, dtype=np.float32)
    if label_mask is None:
        label_mask = np.ones_like(Y, dtype=np.float32)
    else:
        label_mask = np.asarray(label_mask, dtype=np.float32)
    if sample_weight is None:
        sample_weight = np.ones_like(Y, dtype=np.float32)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.ndim == 1:
            sample_weight = np.repeat(sample_weight[:, None], nC, axis=1)

    over = sum(1 for j in np.concatenate([tr_idx, va_idx, te_idx]) if len(str(seqs[j])) > max_len)
    if over:
        print(f"[{arch}][WARN] {over} sequences exceed --ts-max-len={max_len} and are "
              "truncated (5'->3'). Raise --ts-max-len for full-length runs.", flush=True)

    class DS(Dataset):
        def __init__(self, idx):
            self.idx = list(idx)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return _one_hot(seqs[j], max_len), Y[j], sample_weight[j], label_mask[j], j

    def collate(b):
        xs = torch.tensor(np.stack([x[0] for x in b]))
        ys = torch.tensor(np.stack([x[1] for x in b]))
        ws = torch.tensor(np.stack([x[2] for x in b]))
        ms = torch.tensor(np.stack([x[3] for x in b]))
        owners = [x[4] for x in b]
        return xs, ys, ws, ms, owners

    torch.manual_seed(int(getattr(args, "seed", 0)))
    model = _build_models(arch, nC, max_len).to(device)

    known = label_mask[tr_idx].sum(axis=0)
    pos = (Y[tr_idx] * label_mask[tr_idx]).sum(axis=0)
    neg = known - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 0.2, 5.0),
                              dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, "ts_lr", 1e-3)),
                            weight_decay=1e-4)
    batch = int(getattr(args, "ts_batch", 32))

    def run(idx, train):
        dl = DataLoader(DS(idx), batch_size=batch, shuffle=train, collate_fn=collate)
        model.train(train)
        pos_of = {int(j): p for p, j in enumerate(idx)}
        out = np.zeros((len(idx), nC), dtype=np.float32)
        for xs, ys, ws, ms, owners in dl:
            xs, ys, ws, ms = xs.to(device), ys.to(device), ws.to(device), ms.to(device)
            with torch.set_grad_enabled(train):
                logits = model(xs)
                raw = F.binary_cross_entropy_with_logits(logits, ys, pos_weight=pos_weight,
                                                          reduction="none")
                eff = ms * ws
                loss = (raw * eff).sum() / eff.sum().clamp(min=1.0)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            lg = logits.detach().float().cpu().numpy()
            for k, j in enumerate(owners):
                out[pos_of[int(j)]] = lg[k]
        return out

    def macro_auc(idx, prob):
        vals = []
        for j in range(nC):
            v = label_mask[idx, j].astype(bool)
            yt = Y[idx, j][v]
            if 0 < yt.sum() < len(yt):
                vals.append(roc_auc_score(yt, prob[v, j]))
        return float(np.mean(vals)) if vals else 0.0

    best_auc, best_state, stale = -1.0, None, 0
    epochs = int(getattr(args, "ts_epochs", 30))
    patience = int(getattr(args, "ts_patience", 5))
    for ep in range(epochs):
        run(tr_idx, True)
        va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
        auc = macro_auc(va_idx, va_prob)
        print(f"[{arch}] epoch {ep + 1}/{epochs} val_macro_auc={auc:.4f}", flush=True)
        if auc > best_auc + 1e-6:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"[{arch}] early stop after {stale} non-improving epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
    te_prob = 1.0 / (1.0 + np.exp(-run(te_idx, False)))
    print(f"[{arch}] best val_macro_auc={best_auc:.4f}")
    return va_prob, te_prob

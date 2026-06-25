#!/usr/bin/env python3
"""
Fusion model (the "proposed" architecture): an attention-gated late fusion of
heterogeneous feature views — e.g. a foundation-model embedding (RNA-FM) and
hand-crafted localization features (length / GC / motifs) — for neuronal mRNA
localization.

Each view is projected to a common space; an input-dependent gate computes a
softmax weight per view, and the views are combined into one fused vector that
feeds an MLP head.

Training improvements (default ON; controllable via args with safe getattr defaults
so the calling script needs no new flags):
  * label smoothing (noise-robust BCE)        -> args.fusion_label_smooth (0.05)
  * validation-only hyperparameter search      -> args.fusion_hp_search   (True)
  * multi-seed ensemble (averaged probabilities)-> args.fusion_ensemble    (3)
HP search uses ONLY the validation set (no test leakage); the ensemble averages
per-seed probabilities. The deployable artifact stores all ensemble members.

Interface (unchanged):
    train_fusion(blocks, Y, classes, tr_idx, va_idx, te_idx, args,
                 sample_weight=None, label_mask=None) -> (prob_va, prob_te)
"""
from __future__ import annotations

import os

import numpy as np


def _build_fusion(block_dims, n_classes, hidden=128, dropout=0.3):
    import torch
    import torch.nn as nn

    class FusionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.ModuleList([nn.Linear(d, hidden) for d in block_dims])
            self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in block_dims])
            self.gate = nn.Linear(hidden, 1)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, blocks, return_gate=False):
            h = [self.norm[i](torch.relu(self.proj[i](x))) for i, x in enumerate(blocks)]
            H = torch.stack(h, dim=1)                 # (B, n_views, hidden)
            scores = self.gate(H).squeeze(-1)         # (B, n_views)
            w = torch.softmax(scores, dim=1)          # input-dependent view weights
            fused = (H * w.unsqueeze(-1)).sum(dim=1)  # (B, hidden)
            logits = self.head(fused)
            return (logits, w) if return_gate else logits

    return FusionNet()


def train_fusion(blocks, Y, classes, tr_idx, va_idx, te_idx, args,
                 sample_weight=None, label_mask=None):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    device = args.device if (torch.cuda.is_available() or not str(args.device).startswith("cuda")) else "cpu"
    nC = len(classes)
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

    # Standardize each view on TRAIN rows only (no leakage).
    scalers = [StandardScaler().fit(b[tr_idx]) for b in blocks]
    blocks_s = [sc.transform(b).astype(np.float32) for sc, b in zip(scalers, blocks)]
    block_dims = [b.shape[1] for b in blocks_s]
    print(f"[fusion] views={len(blocks_s)} dims={block_dims}")

    # ---- config (improvements; safe defaults) ----
    label_smooth = float(getattr(args, "fusion_label_smooth", 0.05))
    n_ensemble = max(1, int(getattr(args, "fusion_ensemble", 3)))
    do_hp = bool(getattr(args, "fusion_hp_search", True))
    base_seed = int(getattr(args, "seed", 0))
    batch = int(getattr(args, "ts_batch", 64))
    epochs = int(getattr(args, "ts_epochs", 60))
    patience = int(getattr(args, "ts_patience", 8))

    known = label_mask[tr_idx].sum(axis=0)
    pos = (Y[tr_idx] * label_mask[tr_idx]).sum(axis=0)
    neg = known - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 0.2, 5.0),
                              dtype=torch.float32, device=device)

    class DS(Dataset):
        def __init__(self, idx):
            self.idx = list(idx)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return [b[j] for b in blocks_s], Y[j], sample_weight[j], label_mask[j], j

    def collate(b):
        nv = len(blocks_s)
        xs = [torch.tensor(np.stack([x[0][v] for x in b])) for v in range(nv)]
        ys = torch.tensor(np.stack([x[1] for x in b]), dtype=torch.float32)
        ws = torch.tensor(np.stack([x[2] for x in b]), dtype=torch.float32)
        ms = torch.tensor(np.stack([x[3] for x in b]), dtype=torch.float32)
        return xs, ys, ws, ms, [x[4] for x in b]

    def macro_auc(idx, prob):
        vals = []
        for j in range(nC):
            v = label_mask[idx, j].astype(bool)
            yt = Y[idx, j][v]
            if 0 < yt.sum() < len(yt):
                vals.append(roc_auc_score(yt, prob[v, j]))
        return float(np.mean(vals)) if vals else 0.0

    def fit_one(hidden, dropout, lr, seed):
        """Train one fusion net with early stopping; return (state, va_prob, te_prob, best_auc)."""
        torch.manual_seed(seed)
        model = _build_fusion(block_dims, nC, hidden=hidden, dropout=dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        def run(idx, train):
            dl = DataLoader(DS(idx), batch_size=batch, shuffle=train, collate_fn=collate)
            model.train(train)
            pos_of = {int(j): p for p, j in enumerate(idx)}
            out = np.zeros((len(idx), nC), dtype=np.float32)
            for xs, ys, ws, ms, owners in dl:
                xs = [x.to(device) for x in xs]
                ys, ws, ms = ys.to(device), ws.to(device), ms.to(device)
                if train and label_smooth > 0:
                    ys = ys * (1.0 - label_smooth) + 0.5 * label_smooth   # soft targets
                with torch.set_grad_enabled(train):
                    logits = model(xs)
                    raw = F.binary_cross_entropy_with_logits(logits, ys, pos_weight=pos_weight, reduction="none")
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

        best_auc, best_state, stale = -1.0, None, 0
        for ep in range(epochs):
            run(tr_idx, True)
            va = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
            auc = macro_auc(va_idx, va)
            if auc > best_auc + 1e-6:
                best_auc, stale = auc, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        va = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
        te = 1.0 / (1.0 + np.exp(-run(te_idx, False)))
        return best_state, va, te, best_auc

    # ---- (1) hyperparameter search on VALIDATION only ----
    if do_hp:
        grid = [(h, d, lr) for h in (128, 256) for d in (0.3, 0.5) for lr in (1e-3,)]
    else:
        grid = [(128, 0.3, float(getattr(args, "ts_lr", 1e-3)))]
    best_cfg, best_cfg_auc = grid[0], -1.0
    for (h, d, lr) in grid:
        _, _, _, auc = fit_one(h, d, lr, base_seed)
        print(f"[fusion][hp] hidden={h} dropout={d} lr={lr} val_macro_auc={auc:.4f}", flush=True)
        if auc > best_cfg_auc:
            best_cfg_auc, best_cfg = auc, (h, d, lr)
    h, d, lr = best_cfg
    print(f"[fusion] best config: hidden={h} dropout={d} lr={lr} (val {best_cfg_auc:.4f})")

    # ---- (2) multi-seed ensemble with the best config ----
    states, va_probs, te_probs, aucs = [], [], [], []
    for k in range(n_ensemble):
        st, va, te, auc = fit_one(h, d, lr, base_seed + 100 * (k + 1))
        states.append(st); va_probs.append(va); te_probs.append(te); aucs.append(auc)
        print(f"[fusion][ensemble {k + 1}/{n_ensemble}] val_macro_auc={auc:.4f}", flush=True)
    va_prob = np.mean(va_probs, axis=0)
    te_prob = np.mean(te_probs, axis=0)
    print(f"[fusion] ensemble of {n_ensemble} | per-seed val mean {np.mean(aucs):.4f}±{np.std(aucs):.4f} "
          f"| ensemble val {macro_auc(va_idx, va_prob):.4f}")

    out_dir = getattr(args, "output_dir", None)

    # ---- gate weights (interpretability), from the first ensemble member ----
    import torch as _t
    view_tags = list(getattr(args, "features", []) or [f"view{i}" for i in range(len(blocks_s))])
    te_list = list(te_idx)
    gate_w = np.zeros((len(te_list), len(blocks_s)), dtype=np.float32)
    gm = _build_fusion(block_dims, nC, hidden=h, dropout=d).to(device)
    gm.load_state_dict(states[0]); gm.eval()
    with _t.no_grad():
        for b in range(0, len(te_list), batch):
            js = te_list[b:b + batch]
            xs = [_t.tensor(np.stack([bl[j] for j in js])).to(device) for bl in blocks_s]
            _, w = gm(xs, return_gate=True)
            gate_w[b:b + len(js)] = w.detach().cpu().numpy()
    if out_dir is not None:
        import pandas as pd
        pd.DataFrame({f"w_{t}": gate_w[:, i] for i, t in enumerate(view_tags)}).to_csv(
            os.path.join(str(out_dir), "gate_weights.csv"), index=False)
        print(f"[fusion] saved per-test gate weights -> {out_dir}/gate_weights.csv")

    # ---- deployable artifact (all ensemble members) ----
    if out_dir is not None:
        import joblib
        joblib.dump(
            {
                "ensemble_states": states,                 # list of state_dicts (averaged at inference)
                "state_dict": states[0],                   # back-compat single member
                "block_dims": block_dims, "hidden": h, "dropout": d,
                "n_classes": nC, "classes": list(classes), "scalers": scalers,
                "features": list(getattr(args, "features", []) or []),
                "label_smooth": label_smooth, "best_val_macro_auc": float(best_cfg_auc),
            },
            os.path.join(str(out_dir), "fusion_model.joblib"),
        )
        print(f"[fusion] saved deployable ensemble artifact ({len(states)} members) -> "
              f"{out_dir}/fusion_model.joblib")
    return va_prob, te_prob


def score_fusion(artifact_dir, blocks, device="cpu"):
    """Score new sequences with a saved fusion artifact (averages ensemble members)."""
    import joblib
    import torch

    bundle = joblib.load(os.path.join(str(artifact_dir), "fusion_model.joblib"))
    if len(blocks) != len(bundle["block_dims"]):
        raise SystemExit(
            f"[error] fusion expects {len(bundle['block_dims'])} feature views "
            f"({bundle['features']}) but got {len(blocks)}.")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    blocks_s = [sc.transform(b).astype(np.float32) for sc, b in zip(bundle["scalers"], blocks)]
    states = bundle.get("ensemble_states") or [bundle["state_dict"]]
    xs = [torch.tensor(b, device=device) for b in blocks_s]
    probs = []
    for st in states:
        model = _build_fusion(bundle["block_dims"], bundle["n_classes"],
                              hidden=bundle["hidden"], dropout=bundle["dropout"])
        model.load_state_dict(st)
        model.to(device).eval()
        with torch.no_grad():
            probs.append(1.0 / (1.0 + np.exp(-model(xs).cpu().numpy())))
    return np.mean(probs, axis=0)

#!/usr/bin/env python3
"""
Fusion model (the "proposed" architecture): an attention-gated late fusion of
heterogeneous feature views — e.g. a foundation-model embedding (RNA-FM) and
hand-crafted localization features (length / GC / motifs) — for neuronal mRNA
localization.

Each view is projected to a common space; an input-dependent gate computes a
softmax weight per view, and the views are combined into one fused vector that
feeds an MLP head. This lets the model decide, per transcript, how much to trust
the foundation-model representation vs the interpretable engineered features —
and is the mechanism by which fusion can beat either view (or k-mer) alone.

Trained with the SAME masked-BCE + pos_weight + per-sample weight + per-label
mask + leakage-safe split + early stopping as the other model families, so it
enters the benchmark on equal footing.

Interface mirrors finetune_model / train_task_specific:
    train_fusion(blocks, Y, classes, tr_idx, va_idx, te_idx, args,
                 sample_weight=None, label_mask=None) -> (prob_va, prob_te)
where `blocks` is a list of (N, d_i) feature matrices aligned to the gene rows.
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
            # blocks: list of (B, d_i)
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
    import torch.nn as nn
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

    class DS(Dataset):
        def __init__(self, idx):
            self.idx = list(idx)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            return [b[j] for b in blocks_s], Y[j], sample_weight[j], label_mask[j], j

    def collate(batch):
        nv = len(blocks_s)
        xs = [torch.tensor(np.stack([b[0][v] for b in batch])) for v in range(nv)]
        ys = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
        ws = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        ms = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        owners = [b[4] for b in batch]
        return xs, ys, ws, ms, owners

    torch.manual_seed(int(getattr(args, "seed", 0)))
    hidden, dropout = 128, 0.3
    model = _build_fusion(block_dims, nC, hidden=hidden, dropout=dropout).to(device)

    known = label_mask[tr_idx].sum(axis=0)
    pos = (Y[tr_idx] * label_mask[tr_idx]).sum(axis=0)
    neg = known - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 0.2, 5.0),
                              dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(getattr(args, "ts_lr", 1e-3)),
                            weight_decay=1e-4)
    batch = int(getattr(args, "ts_batch", 64))

    def run(idx, train):
        dl = DataLoader(DS(idx), batch_size=batch, shuffle=train, collate_fn=collate)
        model.train(train)
        pos_of = {int(j): p for p, j in enumerate(idx)}
        out = np.zeros((len(idx), nC), dtype=np.float32)
        for xs, ys, ws, ms, owners in dl:
            xs = [x.to(device) for x in xs]
            ys, ws, ms = ys.to(device), ws.to(device), ms.to(device)
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

    def macro_auc(idx, prob):
        vals = []
        for j in range(nC):
            v = label_mask[idx, j].astype(bool)
            yt = Y[idx, j][v]
            if 0 < yt.sum() < len(yt):
                vals.append(roc_auc_score(yt, prob[v, j]))
        return float(np.mean(vals)) if vals else 0.0

    best_auc, best_state, stale = -1.0, None, 0
    epochs = int(getattr(args, "ts_epochs", 60))
    patience = int(getattr(args, "ts_patience", 8))
    for ep in range(epochs):
        run(tr_idx, True)
        va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
        auc = macro_auc(va_idx, va_prob)
        print(f"[fusion] epoch {ep + 1}/{epochs} val_macro_auc={auc:.4f}", flush=True)
        if auc > best_auc + 1e-6:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"[fusion] early stop after {stale} non-improving epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    va_prob = 1.0 / (1.0 + np.exp(-run(va_idx, False)))
    te_prob = 1.0 / (1.0 + np.exp(-run(te_idx, False)))
    print(f"[fusion] best val_macro_auc={best_auc:.4f}")

    out_dir = getattr(args, "output_dir", None)

    # Per-test-sample gate weights (softmax over views) for interpretability:
    # how much the model relied on each view (e.g. foundation model vs engineered)
    # for each transcript. Saved in te_idx order so it aligns by position with
    # test_predictions.csv.
    view_tags = list(getattr(args, "features", []) or [f"view{i}" for i in range(len(blocks_s))])
    te_list = list(te_idx)
    gate_w = np.zeros((len(te_list), len(blocks_s)), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(te_list), batch):
            js = te_list[b:b + batch]
            xs = [torch.tensor(np.stack([bl[j] for j in js])).to(device) for bl in blocks_s]
            _, w = model(xs, return_gate=True)
            gate_w[b:b + len(js)] = w.detach().cpu().numpy()
    if out_dir is not None:
        import pandas as pd
        pd.DataFrame({f"w_{t}": gate_w[:, i] for i, t in enumerate(view_tags)}).to_csv(
            os.path.join(str(out_dir), "gate_weights.csv"), index=False)
        print(f"[fusion] saved per-test gate weights -> {out_dir}/gate_weights.csv")

    # Persist a deployable artifact so predict.py can score new sequences with the
    # SAME fused architecture, per-view standardization and class order.
    if out_dir is not None:
        import joblib
        joblib.dump(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "block_dims": block_dims,
                "hidden": hidden,
                "dropout": dropout,
                "n_classes": nC,
                "classes": list(classes),
                "scalers": scalers,
                "features": list(getattr(args, "features", []) or []),
            },
            os.path.join(str(out_dir), "fusion_model.joblib"),
        )
        print(f"[fusion] saved deployable artifact -> {out_dir}/fusion_model.joblib")
    return va_prob, te_prob


def score_fusion(artifact_dir, blocks, device="cpu"):
    """Score new sequences with a saved fusion artifact.

    `blocks` is a list of (N, d_i) raw feature matrices in the SAME order as the
    training `--features` views; per-view scalers and the fused net are reloaded
    from `fusion_model.joblib`. Returns an (N, n_classes) probability array.
    """
    import joblib
    import torch

    bundle = joblib.load(os.path.join(str(artifact_dir), "fusion_model.joblib"))
    if len(blocks) != len(bundle["block_dims"]):
        raise SystemExit(
            f"[error] fusion expects {len(bundle['block_dims'])} feature views "
            f"({bundle['features']}) but got {len(blocks)}."
        )
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    blocks_s = [sc.transform(b).astype(np.float32) for sc, b in zip(bundle["scalers"], blocks)]
    model = _build_fusion(bundle["block_dims"], bundle["n_classes"],
                          hidden=bundle["hidden"], dropout=bundle["dropout"])
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()
    xs = [torch.tensor(b, device=device) for b in blocks_s]
    with torch.no_grad():
        logits = model(xs).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))

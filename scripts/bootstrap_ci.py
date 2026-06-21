#!/usr/bin/env python3
"""
Group-level (cluster) bootstrap test for whether two runs differ on the test set.

Reads two runs' test_predictions.csv, joins on (species, gene_name) so both models
are scored on the SAME test samples, then resamples SPLIT GROUPS with replacement
(respecting the leakage-safe grouping — never individual correlated samples) to get
a 95% CI and bootstrap p-value for (metric_A - metric_B).

Usage:
    python scripts/bootstrap_ci.py \
        --a results/track3_full/rnafm --b results/track3_full/kmer \
        --label is_neurite --metric roc_auc --n-boot 2000

For fine multi-label runs pass --label Cell_body (etc.). Default label = is_neurite
(the binary soma-vs-neurite task).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _metric(y, s, name):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s) if name == "roc_auc" else average_precision_score(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="run dir A (has test_predictions.csv)")
    ap.add_argument("--b", type=Path, required=True, help="run dir B")
    ap.add_argument("--label", default="is_neurite")
    ap.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    def load(run):
        p = Path(run) / "test_predictions.csv"
        if not p.exists():
            raise SystemExit(f"[error] {p} not found. Re-run training (it now saves "
                             "test_predictions.csv).")
        return pd.read_csv(p)

    A, B = load(args.a), load(args.b)
    yk, mk, pk = f"y_{args.label}", f"mask_{args.label}", f"prob_{args.label}"
    for d, n in ((A, "A"), (B, "B")):
        if pk not in d.columns:
            raise SystemExit(f"[error] label '{args.label}' not in run {n}. "
                             f"Available: {[c[5:] for c in d.columns if c.startswith('prob_')]}")

    key = ["species", "gene_name"]
    m = A[key + ["split_group", yk, mk, pk]].merge(
        B[key + [pk]], on=key, suffixes=("_a", "_b"))
    # keep only samples where the label is evaluable
    m = m[m[mk] == 1].reset_index(drop=True)
    if len(m) == 0:
        raise SystemExit("[error] no overlapping evaluable test samples between A and B.")

    groups = m["split_group"].to_numpy()
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    y = m[yk].to_numpy()
    sa = m[f"{pk}_a"].to_numpy()
    sb = m[f"{pk}_b"].to_numpy()

    obs_a = _metric(y, sa, args.metric)
    obs_b = _metric(y, sb, args.metric)
    obs_d = obs_a - obs_b

    rng = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        da = _metric(y[idx], sa[idx], args.metric)
        db = _metric(y[idx], sb[idx], args.metric)
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: how often the difference crosses 0
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = min(p, 1.0)

    print(f"=== {args.metric}  A={args.a.name}  B={args.b.name}  label={args.label} ===")
    print(f"n_eval_samples={len(m)}  n_groups={len(uniq)}  n_boot={len(diffs)}")
    print(f"A {args.metric} = {obs_a:.4f}")
    print(f"B {args.metric} = {obs_b:.4f}")
    print(f"observed diff (A-B) = {obs_d:+.4f}")
    print(f"95% CI (group bootstrap) = [{lo:+.4f}, {hi:+.4f}]")
    print(f"bootstrap p (two-sided) = {p:.3f}")
    sig = (lo > 0) or (hi < 0)
    print(f">>> {'SIGNIFICANT' if sig else 'NOT significant'} at 95% "
          f"(CI {'excludes' if sig else 'includes'} 0)")


if __name__ == "__main__":
    main()

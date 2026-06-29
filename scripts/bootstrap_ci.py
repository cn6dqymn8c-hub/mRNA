#!/usr/bin/env python3
"""
Group-level (cluster) bootstrap test for whether two runs differ on the test set.

Reads two runs' test_predictions.csv, joins on (species, gene_name) so both models
are scored on the SAME test samples, then resamples SPLIT GROUPS with replacement
(respecting the leakage-safe grouping — never individual correlated samples) to get
a 95% CI and bootstrap p-value for (metric_A - metric_B).

CLI (single comparison):
    python scripts/bootstrap_ci.py \
        --a results/track3_full/rnafm --b results/track3_full/kmer \
        --label is_neurite --metric roc_auc --n-boot 2000

Also exposes bootstrap_compare(...) for batch use by summarize_results.py.
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


def _aligned(a_dir, b_dir):
    """Read both runs' test_predictions.csv and return a single frame whose rows are
    the SAME test samples for both models (split_group + per-label y_/mask_/prob_ for
    A and B). Returns None if predictions are missing. Raises if the two runs cannot be
    aligned (different row order at isoform level, where the key is not unique)."""
    pa, pb = Path(a_dir) / "test_predictions.csv", Path(b_dir) / "test_predictions.csv"
    if not pa.exists() or not pb.exists():
        return None
    A, B = pd.read_csv(pa).reset_index(drop=True), pd.read_csv(pb).reset_index(drop=True)
    key = ["species", "gene_name"]
    # Two runs in the same setting share data/split/seed, so their test_predictions
    # rows are in IDENTICAL order. Align by position — the only correct join at isoform
    # level, where (species, gene_name) is NOT unique and a key merge would explode.
    same_order = (
        len(A) == len(B)
        and A["species"].astype(str).equals(B["species"].astype(str))
        and A["gene_name"].astype(str).equals(B["gene_name"].astype(str))
    )
    ycols = [c for c in A.columns if c.startswith("y_")]
    mcols = [c for c in A.columns if c.startswith("mask_")]
    pcols = [c for c in A.columns if c.startswith("prob_")]
    if same_order:
        m = A[["species", "gene_name", "split_group"] + ycols + mcols + pcols].copy()
        for c in pcols:
            m[f"{c}_b"] = B[c].to_numpy()
        m = m.rename(columns={c: f"{c}_a" for c in pcols})
    else:
        if A.duplicated(key).any() or B.duplicated(key).any():
            raise SystemExit(
                "[error] runs are not in the same row order and (species, gene_name) is "
                "not unique (isoform-level). Re-run both with the same config/seed so "
                "their test_predictions rows align by position."
            )
        m = A[key + ["split_group"] + ycols + mcols + pcols].merge(
            B[key + pcols], on=key, suffixes=("_a", "_b"))
    return m


def _boot_diff(y, sa, sb, groups, gidx, uniq, metric, n_boot, rng):
    """Group-resampled distribution of (metric_A - metric_B) for one label."""
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([gidx[g] for g in pick])
        da, db = _metric(y[idx], sa[idx], metric), _metric(y[idx], sb[idx], metric)
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    return np.asarray(diffs)


def bootstrap_compare(a_dir, b_dir, label="is_neurite", metric="roc_auc",
                      n_boot=2000, seed=0):
    """Return a dict with observed metrics, diff, 95% CI, bootstrap p, or None if
    predictions are missing / the label is absent / there is no overlap."""
    m = _aligned(a_dir, b_dir)
    if m is None:
        return None
    yk, mk, pk = f"y_{label}", f"mask_{label}", f"prob_{label}"
    if f"{pk}_a" not in m.columns:
        return None
    m = m[m[mk] == 1].reset_index(drop=True)
    if len(m) == 0:
        return None

    groups = m["split_group"].to_numpy()
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    y = m[yk].to_numpy()
    sa, sb = m[f"{pk}_a"].to_numpy(), m[f"{pk}_b"].to_numpy()

    obs_a, obs_b = _metric(y, sa, metric), _metric(y, sb, metric)
    rng = np.random.default_rng(seed)
    diffs = _boot_diff(y, sa, sb, groups, gidx, uniq, metric, n_boot, rng)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan))
    p = min(2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0) if len(diffs) else np.nan
    return {
        "metric": metric, "label": label,
        "A_metric": round(float(obs_a), 4), "B_metric": round(float(obs_b), 4),
        "diff": round(float(obs_a - obs_b), 4),
        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
        "p": round(float(p), 4),
        "n_eval": int(len(m)), "n_groups": int(len(uniq)),
        "significant": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
    }


def bootstrap_compare_macro(a_dir, b_dir, labels, metric="roc_auc", n_boot=2000, seed=0):
    """Macro version: average `metric` over `labels` (the fine compartments) and test
    whether macro_A - macro_B differs. Each bootstrap resamples split_group ONCE and
    recomputes the macro on the same resample for both models, so the per-label
    correlations are preserved. A label contributes only where its mask==1 and both
    classes are present in the resample; the macro averages over labels valid that
    round. Returns a dict (label='macro<k>') or None if no labels are usable."""
    m = _aligned(a_dir, b_dir)
    if m is None:
        return None
    rng = np.random.default_rng(seed)
    groups = m["split_group"].to_numpy()
    uniq = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in uniq}

    # per-label arrays restricted to that label's evaluable rows, plus a
    # group->row-positions map (precomputed so each bootstrap round is just gathers).
    lab_arrays, used = {}, []
    for lab in labels:
        yk, mk, pk = f"y_{lab}", f"mask_{lab}", f"prob_{lab}"
        if f"{pk}_a" not in m.columns:
            continue
        valid = (m[mk] == 1).to_numpy()
        if valid.sum() == 0:
            continue
        g = groups[valid]
        used.append(lab)
        lab_arrays[lab] = (
            m[yk].to_numpy()[valid],
            m[f"{pk}_a"].to_numpy()[valid],
            m[f"{pk}_b"].to_numpy()[valid],
            {gg: np.where(g == gg)[0] for gg in np.unique(g)},
        )
    if not used:
        return None

    def macro(pick=None):
        va, vb = [], []
        for lab in used:
            y, sa, sb, lg = lab_arrays[lab]
            if pick is None:
                idx = np.arange(len(y))
            else:
                parts = [lg[gg] for gg in pick if gg in lg]
                if not parts:
                    continue
                idx = np.concatenate(parts)
            da, db = _metric(y[idx], sa[idx], metric), _metric(y[idx], sb[idx], metric)
            if np.isfinite(da) and np.isfinite(db):
                va.append(da); vb.append(db)
        if not va:
            return np.nan, np.nan
        return float(np.mean(va)), float(np.mean(vb))

    obs_a, obs_b = macro()
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        da, db = macro(pick)
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    lo, hi = (np.percentile(diffs, [2.5, 97.5]) if len(diffs) else (np.nan, np.nan))
    p = min(2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0) if len(diffs) else np.nan
    return {
        "metric": metric, "label": f"macro{len(used)}",
        "A_metric": round(float(obs_a), 4), "B_metric": round(float(obs_b), 4),
        "diff": round(float(obs_a - obs_b), 4),
        "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
        "p": round(float(p), 4),
        "n_eval": int(len(m)), "n_groups": int(len(uniq)),
        "significant": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="run dir A (has test_predictions.csv)")
    ap.add_argument("--b", type=Path, required=True, help="run dir B")
    ap.add_argument("--label", default="is_neurite")
    ap.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    r = bootstrap_compare(args.a, args.b, args.label, args.metric, args.n_boot, args.seed)
    if r is None:
        raise SystemExit("[error] missing test_predictions.csv, missing label, or no "
                         "overlapping evaluable test samples. Re-run training to save "
                         "test_predictions.csv.")
    print(f"=== {args.metric}  A={args.a.name}  B={args.b.name}  label={args.label} ===")
    print(f"n_eval={r['n_eval']}  n_groups={r['n_groups']}")
    print(f"A = {r['A_metric']:.4f}   B = {r['B_metric']:.4f}")
    print(f"observed diff (A-B) = {r['diff']:+.4f}")
    print(f"95% CI (group bootstrap) = [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
    print(f"bootstrap p (two-sided) = {r['p']:.3f}")
    print(f">>> {'SIGNIFICANT' if r['significant'] else 'NOT significant'} at 95% "
          f"(CI {'excludes' if r['significant'] else 'includes'} 0)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build leakage-safe 5-fold cross-validation splits from an existing run's
split_assignments.csv (which already carries the leakage-safe `split_group` for
every sample of a setting's universe).

We re-partition the GROUPS (never individual samples) into k folds. For fold j:
    test  = groups in chunk j
    val   = groups in chunk (j+1) mod k
    train = the remaining groups
so every sample is tested exactly once across folds and no leakage-safe group
ever straddles train/val/test within a fold. Each fold is written as a
split_assignments-style CSV (species, gene_name, split) that the training script
consumes via --split-assignments.

Usage:
    python scripts/make_cv_folds.py \
        --from-split results/track3_full/kmer/split_assignments.csv \
        --out-prefix results/_frozen_splits/cv_track3_full --k 5 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-split", type=Path, required=True,
                    help="an existing split_assignments.csv (has species, gene_name, split_group)")
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="output path prefix; writes <prefix>_fold{0..k-1}.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.from_split, dtype={"species": str, "gene_name": str})
    if not {"species", "gene_name"}.issubset(df.columns):
        raise SystemExit(f"[error] {args.from_split} needs species, gene_name; found {df.columns.tolist()}")
    if "split_group" in df.columns:
        groups = df["split_group"].astype(str).to_numpy()
    else:
        # Fallback: group by gene (species|GENE). Leakage-safe at the gene level
        # but WITHOUT ortholog / exact-sequence merging — slightly less strict.
        print("[cv][warn] no 'split_group' column; falling back to gene-level grouping "
              "(species|gene). For full leakage-safety supply a split with split_group.")
        groups = (df["species"].astype(str).str.lower().str.strip() + "|"
                  + df["gene_name"].astype(str).str.upper().str.strip()).to_numpy()
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, args.k)              # k disjoint group chunks
    gset = [set(c.tolist()) for c in chunks]
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cv] {len(df)} samples, {len(uniq)} leakage-safe groups -> {args.k} folds")
    for j in range(args.k):
        test_g = gset[j]
        val_g = gset[(j + 1) % args.k]
        split = np.where(np.isin(groups, list(test_g)), "test",
                         np.where(np.isin(groups, list(val_g)), "val", "train"))
        out = pd.DataFrame({"species": df["species"], "gene_name": df["gene_name"], "split": split})
        # leakage check
        g_by = {s: set(groups[split == s]) for s in ("train", "val", "test")}
        if (g_by["train"] & g_by["val"]) or (g_by["train"] & g_by["test"]) or (g_by["val"] & g_by["test"]):
            raise SystemExit(f"[error] fold {j}: group leakage across splits")
        path = f"{args.out_prefix}_fold{j}.csv"
        out.to_csv(path, index=False)
        n = {s: int((split == s).sum()) for s in ("train", "val", "test")}
        print(f"  fold{j}: train={n['train']} val={n['val']} test={n['test']} -> {path}")
    # sanity: every sample is test exactly once across folds
    print(f"[cv] each sample is in the test set exactly once across the {args.k} folds.")


if __name__ == "__main__":
    main()

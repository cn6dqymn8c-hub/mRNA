#!/usr/bin/env python3
"""
Run every benchmark significance comparison in one shot, for BOTH metrics
(roc_auc and pr_auc), and collect the results into a single tidy CSV.

Each comparison is a group-level (cluster) bootstrap of (metric_A - metric_B)
on the shared test set (see bootstrap_ci.bootstrap_compare). Output columns:
    group, track, A, B, label, metric, A_metric, B_metric, diff, ci_lo, ci_hi,
    p, n_eval, n_groups, significant

Usage:
    python scripts/run_all_bootstraps.py --results-dir results \
        --out results/bootstrap_all.csv --n-boot 2000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import bootstrap_compare

# fusion run-dir name per track (CDS uses the codon model mRNA-FM)
FUSION = {
    "track1a_gene": "fusion_rnafm_eng",
    "track1b_isoform": "fusion_rnafm_eng",
    "track2_gene": "fusion_mrnafm_eng",
    "track3_full": "fusion_rnafm_eng",
    "track3_full_isoform": "fusion_rnafm_eng",
}
# best single foundation model per track
BEST_FM = {
    "track1a_gene": "rnafm",
    "track1b_isoform": "rnafm",
    "track2_gene": "mrnafm",
    "track3_full": "rnafm",
    "track3_full_isoform": "rnafm",
}
BINARY_TRACKS = list(FUSION)
FINE_TRACKS = ["fine_full_gene", "fine_full_isoform"]
COMPARTMENTS = ["Cell_body", "Dendrite", "Neuropil", "Axon", "Neurite"]


def build_comparisons():
    """Return list of (group, track, A_subdir, B_subdir, label)."""
    comps = []
    for t in BINARY_TRACKS:
        fus = FUSION[t]
        comps.append(("fusion_vs_kmer", t, fus, "kmer", "is_neurite"))
        comps.append(("fusion_vs_net", t, fus, "rnatracker", "is_neurite"))
        comps.append(("fusion_vs_net", t, fus, "dm3loc", "is_neurite"))
        comps.append(("bestFM_vs_kmer", t, BEST_FM[t], "kmer", "is_neurite"))
    # component decomposition (full-length + 3'UTR isoform)
    for t in ("track3_full", "track1b_isoform"):
        fus = FUSION[t]
        comps.append(("decomp", t, "engineered", "length", "is_neurite"))
        comps.append(("decomp", t, fus, "engineered", "is_neurite"))
        comps.append(("decomp", t, fus, "rnafm", "is_neurite"))
    # region ablation (matched genes, RNA-FM, region the only variable)
    comps.append(("region_ablation", "ablation_region", "rnafm_full", "rnafm_utr3", "is_neurite"))
    comps.append(("region_ablation", "ablation_region", "rnafm_full", "rnafm_cds", "is_neurite"))
    comps.append(("region_ablation", "ablation_region", "rnafm_cds", "rnafm_utr3", "is_neurite"))
    # fine multi-label: fusion vs kmer per compartment
    for t in FINE_TRACKS:
        for lab in COMPARTMENTS:
            comps.append(("fine_fusion_vs_kmer", t, "fusion_rnafm_eng", "kmer", lab))
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/bootstrap_all.csv"))
    ap.add_argument("--metrics", nargs="+", default=["roc_auc", "pr_auc"],
                    choices=["roc_auc", "pr_auc"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows, skipped = [], []
    for group, track, a_sub, b_sub, label in build_comparisons():
        a_dir = args.results_dir / track / a_sub
        b_dir = args.results_dir / track / b_sub
        for metric in args.metrics:
            r = bootstrap_compare(a_dir, b_dir, label=label, metric=metric,
                                  n_boot=args.n_boot, seed=args.seed)
            if r is None:
                skipped.append((group, track, a_sub, b_sub, label, metric))
                continue
            rows.append({"group": group, "track": track, "A": a_sub, "B": b_sub, **r})
            print(f"[{metric}] {track:22s} {a_sub} vs {b_sub} ({label}): "
                  f"d={r['diff']:+.4f} p={r['p']:.3f} "
                  f"{'SIG' if r['significant'] else 'ns'}", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows -> {args.out}")
    if skipped:
        print(f"[skipped] {len(skipped)} comparisons (missing run dir / label):")
        for s in skipped:
            print("   ", s)


if __name__ == "__main__":
    main()

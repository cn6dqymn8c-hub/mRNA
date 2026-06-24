#!/usr/bin/env python3
"""
Aggregate multiple seed runs of the same config into mean ± sd of test metrics.

Point it at the per-seed run dirs (each has overall_metrics.csv with a row where
split=="test"); it reports mean, sd and min/max of test ROC-AUC / AUPRC across
seeds, so single-seed results can be replaced by seed-robust mean ± sd.

Usage:
    python scripts/aggregate_seeds.py \
        --runs results/track3_full/fusion_rnafm_eng \
               results/track3_full/fusion_rnafm_eng_s1 \
               results/track3_full/fusion_rnafm_eng_s2 \
               results/track3_full/fusion_rnafm_eng_s3 \
               results/track3_full/fusion_rnafm_eng_s4 \
        --label-glob "results/track3_full/fusion_rnafm_eng*"   # alternative: shell-glob the dirs
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _test_row(run_dir):
    f = Path(run_dir) / "overall_metrics.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    t = df[df.get("split", "test") == "test"]
    return t.iloc[0] if len(t) else df.iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="explicit per-seed run dirs")
    ap.add_argument("--glob", default=None, help="shell glob for run dirs (e.g. 'results/track3_full/fusion*')")
    ap.add_argument("--metrics", nargs="+", default=["macro_roc_auc", "macro_pr_auc"])
    ap.add_argument("--out", type=Path, default=None, help="optional CSV to append the summary row")
    args = ap.parse_args()

    runs = list(args.runs or [])
    if args.glob:
        runs += sorted(glob.glob(args.glob))
    runs = [r for r in dict.fromkeys(runs)]  # dedup, keep order
    if not runs:
        raise SystemExit("[error] provide --runs and/or --glob")

    rows = []
    for r in runs:
        tr = _test_row(r)
        if tr is None:
            print(f"[skip] no overall_metrics.csv in {r}")
            continue
        rows.append({"run": r, **{m: float(tr[m]) for m in args.metrics if m in tr}})
    if not rows:
        raise SystemExit("[error] no usable runs")
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} seed runs:")
    print(df.to_string(index=False))

    print("\n=== mean ± sd across seeds ===")
    summary = {"n_seeds": len(df)}
    for m in args.metrics:
        if m in df.columns:
            v = df[m].to_numpy()
            summary[f"{m}_mean"] = round(float(v.mean()), 4)
            summary[f"{m}_sd"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
            summary[f"{m}_min"] = round(float(v.min()), 4)
            summary[f"{m}_max"] = round(float(v.max()), 4)
            print(f"  {m}: {v.mean():.4f} ± {v.std(ddof=1) if len(v) > 1 else 0:.4f} "
                  f"(min {v.min():.4f}, max {v.max():.4f}, n={len(v)})")
    if args.out:
        pd.DataFrame([summary]).to_csv(args.out, index=False)
        print(f"\nwrote summary -> {args.out}")


if __name__ == "__main__":
    main()

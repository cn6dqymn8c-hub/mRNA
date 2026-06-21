#!/usr/bin/env python3
"""
Aggregate and visualize benchmark results.

Walks a results directory, reads every run's overall_metrics.csv (+ run_config.json),
builds a tidy summary table, and renders comparison figures. Comparisons are made
WITHIN a track (different tracks = different data universes / priors and are NOT
directly comparable); the region ablation is shown separately as the valid
3'UTR-vs-CDS-vs-full comparison.

Usage:
    python scripts/summarize_results.py --results-dir results --metric macro_pr_auc

Outputs (under --results-dir):
    summary_table.csv          one row per run (val + test metrics, config)
    fig_by_track.png           per-track model comparison (test metric) vs prior
    fig_val_vs_test.png        val vs test (overfitting / noise check)
    fig_region_ablation.png    utr3/cds/full on the same genes (if present)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINES = {"label_prior_probability", "all_zero"}


def _load(results_dir: Path):
    rows = []
    for ov_path in sorted(results_dir.rglob("overall_metrics.csv")):
        run_dir = ov_path.parent
        # skip the frozen-split scaffold dirs
        if "_frozen_splits" in run_dir.parts:
            continue
        try:
            ov = pd.read_csv(ov_path)
        except Exception:
            continue
        if "split" not in ov.columns:
            ov["split"] = "test"  # legacy runs: treat the single row as test
        cfg = {}
        cfg_path = run_dir / "run_config.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}

        def metric(split, model_filter, col):
            sub = ov[ov["split"] == split]
            if model_filter == "model":
                sub = sub[~sub["model"].isin(BASELINES)]
            else:
                sub = sub[sub["model"] == model_filter]
            return float(sub.iloc[0][col]) if len(sub) and col in sub.columns else np.nan

        track = run_dir.parent.name
        model = run_dir.name
        rows.append({
            "track": track, "model": model, "run_dir": str(run_dir),
            "region": cfg.get("region"), "sample_level": cfg.get("sample_level"),
            "label_scheme": cfg.get("label_scheme"),
            "val_pr_auc": metric("val", "model", "macro_pr_auc"),
            "test_pr_auc": metric("test", "model", "macro_pr_auc"),
            "val_roc_auc": metric("val", "model", "macro_roc_auc"),
            "test_roc_auc": metric("test", "model", "macro_roc_auc"),
            "prior_pr_auc": metric("test", "label_prior_probability", "macro_pr_auc"),
            "n_test": metric("test", "model", "n_test"),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", default="pr_auc", choices=["pr_auc", "roc_auc"])
    args = ap.parse_args()

    df = _load(args.results_dir)
    if df.empty:
        raise SystemExit(f"No overall_metrics.csv found under {args.results_dir}")
    df = df.sort_values(["track", f"val_{args.metric}"], ascending=[True, False])
    out_csv = args.results_dir / "summary_table.csv"
    df.to_csv(out_csv, index=False)

    print("=== summary (sorted by val within track) ===")
    cols = ["track", "model", "region", f"val_{args.metric}", f"test_{args.metric}", "prior_pr_auc", "n_test"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved {out_csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[viz] matplotlib not installed (`pip install matplotlib`); table only.")
        return

    vcol, tcol = f"val_{args.metric}", f"test_{args.metric}"

    # 1) per-track model comparison (test metric) + prior line, ablation excluded here
    main_tracks = [t for t in df["track"].unique() if "ablation" not in t]
    if main_tracks:
        ncol = min(2, len(main_tracks)); nrow = int(np.ceil(len(main_tracks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4 * nrow), squeeze=False)
        for i, tr in enumerate(sorted(main_tracks)):
            ax = axes[i // ncol][i % ncol]
            d = df[df["track"] == tr].sort_values(tcol, ascending=False)
            x = np.arange(len(d))
            ax.bar(x - 0.2, d[vcol], width=0.4, label="val", color="#88b")
            ax.bar(x + 0.2, d[tcol], width=0.4, label="test", color="#3a6")
            prior = d["prior_pr_auc"].dropna()
            if args.metric == "pr_auc" and len(prior):
                ax.axhline(prior.iloc[0], ls="--", color="grey", label="prior")
            ax.set_xticks(x); ax.set_xticklabels(d["model"], rotation=30, ha="right", fontsize=8)
            ax.set_title(tr); ax.set_ylabel(args.metric); ax.legend(fontsize=7)
        for j in range(len(main_tracks), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.tight_layout(); fig.savefig(args.results_dir / "fig_by_track.png", dpi=130)
        print(f"Saved {args.results_dir / 'fig_by_track.png'}")

    # 2) val vs test scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df[vcol], df[tcol], s=24)
    lim = [min(df[vcol].min(), df[tcol].min()) - 0.02, max(df[vcol].max(), df[tcol].max()) + 0.02]
    ax.plot(lim, lim, ls="--", color="grey")
    for _, r in df.iterrows():
        ax.annotate(f"{r['track']}/{r['model']}", (r[vcol], r[tcol]), fontsize=5, alpha=0.7)
    ax.set_xlabel(f"val {args.metric}"); ax.set_ylabel(f"test {args.metric}")
    ax.set_title("val vs test (on/near diagonal = no overfit; scatter = noise)")
    fig.tight_layout(); fig.savefig(args.results_dir / "fig_val_vs_test.png", dpi=130)
    print(f"Saved {args.results_dir / 'fig_val_vs_test.png'}")

    # 3) region ablation (same genes): utr3 vs cds vs full
    abl = df[df["track"].str.contains("ablation", na=False)].copy()
    if not abl.empty:
        abl["region_"] = abl["region"].fillna(abl["model"].str.extract(r"(utr3|cds|full)")[0])
        order = [r for r in ["utr3", "cds", "full"] if r in set(abl["region_"])]
        abl = abl.set_index("region_").reindex(order)
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(order))
        ax.bar(x - 0.2, abl[vcol], width=0.4, label="val", color="#88b")
        ax.bar(x + 0.2, abl[tcol], width=0.4, label="test", color="#3a6")
        ax.set_xticks(x); ax.set_xticklabels(order)
        ax.set_ylabel(args.metric); ax.set_title("Region ablation (same genes, RNA-FM): 3'UTR vs CDS vs full")
        ax.legend()
        fig.tight_layout(); fig.savefig(args.results_dir / "fig_region_ablation.png", dpi=130)
        print(f"Saved {args.results_dir / 'fig_region_ablation.png'}")
    else:
        print("[viz] no ablation_region runs found (run: bash scripts/run_all.sh ablation).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gate interpretability analysis for the fusion model.

Reads a fusion run dir's gate_weights.csv (per-test-sample softmax weights over
views, in te_idx order) and test_predictions.csv (same order: species, gene_name,
y_/prob_/mask_ per class), and optionally gene_table.csv (seq_len). Shows how much
the gate relied on the foundation-model view vs the engineered view, and whether
that reliance depends on transcript length or the label.

Writes results/figures/fig_gate_interpretation.(png|pdf).
Usage:
    python scripts/make_gate_analysis.py --run-dir results/track3_full/fusion_rnafm_eng \
        --gene-table results/track3_full/fusion_rnafm_eng/gene_table.csv --label is_neurite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_FM, C_ENG = "#2f6f95", "#4f8a5b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="fusion run dir with gate_weights.csv")
    ap.add_argument("--gene-table", type=Path, default=None, help="gene_table.csv for seq_len (optional)")
    ap.add_argument("--label", default="is_neurite", help="label column in test_predictions for the by-class panel")
    ap.add_argument("--out", type=Path, default=Path("results/figures/fig_gate_interpretation"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    gate = pd.read_csv(args.run_dir / "gate_weights.csv")
    preds = pd.read_csv(args.run_dir / "test_predictions.csv")
    if len(gate) != len(preds):
        raise SystemExit(f"[error] gate_weights ({len(gate)}) and test_predictions ({len(preds)}) "
                         "differ in length; they must be the same te_idx order.")
    # FM-view weight column (prefer 'w_fm'); engineered = remaining
    wcols = [c for c in gate.columns if c.startswith("w_")]
    fmcol = "w_fm" if "w_fm" in gate.columns else wcols[0]
    df = pd.concat([gate.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
    df["w_fm"] = df[fmcol]
    print(f"[gate] {len(df)} test transcripts; mean w_fm={df['w_fm'].mean():.3f} "
          f"(median {df['w_fm'].median():.3f}); views={wcols}")

    have_len = False
    if args.gene_table and Path(args.gene_table).exists():
        gt = pd.read_csv(args.gene_table)
        if {"species", "gene_name", "seq_len"}.issubset(gt.columns):
            gt = gt.drop_duplicates(["species", "gene_name"])[["species", "gene_name", "seq_len"]]
            df = df.merge(gt, on=["species", "gene_name"], how="left")
            have_len = df["seq_len"].notna().any()

    npanel = 2 + int(have_len)
    fig, axes = plt.subplots(1, npanel, figsize=(4.6 * npanel, 4.2))
    if npanel == 1:
        axes = [axes]
    ai = 0

    # (a) distribution of FM-view weight
    ax = axes[ai]; ai += 1
    ax.hist(df["w_fm"], bins=30, color=C_FM, alpha=0.85, edgecolor="black", linewidth=0.3)
    ax.axvline(df["w_fm"].mean(), color="black", ls="--", lw=1, label=f"mean {df['w_fm'].mean():.2f}")
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("gate weight on foundation-model view  (w_fm)")
    ax.set_ylabel("transcripts"); ax.set_title("a  Gate reliance on FM view"); ax.legend(fontsize=8)

    # (b) FM-view weight vs transcript length (if available)
    if have_len:
        ax = axes[ai]; ai += 1
        d = df.dropna(subset=["seq_len"]).copy()
        d["logL"] = np.log10(d["seq_len"].clip(lower=1))
        bins = np.linspace(d["logL"].min(), d["logL"].max(), 9)
        d["bin"] = pd.cut(d["logL"], bins)
        g = d.groupby("bin", observed=True)["w_fm"].agg(["mean", "std", "count"])
        centers = [iv.mid for iv in g.index]
        ax.errorbar(centers, g["mean"], yerr=g["std"], fmt="o-", color=C_FM, capsize=3)
        ax.set_xlabel("log10 transcript length (nt)")
        ax.set_ylabel("mean w_fm"); ax.set_title("b  Reliance vs length")

    # (c) FM-view weight by label (positive vs negative for the chosen label)
    yk, mk = f"y_{args.label}", f"mask_{args.label}"
    ax = axes[ai]; ai += 1
    if yk in df.columns:
        m = df[mk] == 1 if mk in df.columns else np.ones(len(df), bool)
        pos = df.loc[m & (df[yk] == 1), "w_fm"]
        neg = df.loc[m & (df[yk] == 0), "w_fm"]
        ax.boxplot([neg.dropna(), pos.dropna()], positions=[1, 2],
                   patch_artist=True, boxprops=dict(facecolor=C_FM, alpha=0.6))
        ax.set_xticks([1, 2]); ax.set_xticklabels(["soma\n(neg)", "neurite\n(pos)"])
        ax.set_ylabel("w_fm"); ax.set_title(f"c  Reliance by label ({args.label})")
    else:
        ax.text(0.5, 0.5, f"label {args.label}\nnot in predictions", ha="center", va="center")
        ax.axis("off")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()

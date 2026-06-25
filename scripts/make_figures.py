#!/usr/bin/env python3
"""
Generate the main figures from the benchmark outputs.

Inputs (produced earlier in the pipeline):
  results/summary_table.csv     (per run: test_roc_auc / test_pr_auc / prior / n_test)
  results/bootstrap_all.csv      (group-bootstrap diffs/CI/p for every comparison)

Figures written to results/figures/:
  fig1_benchmark_baseline.(png|pdf)  -- per-setting model bars + best-FM vs k-mer Δ
  fig2_fusion_model.(png|pdf)        -- fusion vs k-mer Δ + component decomposition
  fig3_region_ablation.(png|pdf)     -- matched-gene region ablation
  fig4_multilabel.(png|pdf)          -- fine per-compartment fusion vs k-mer Δ

Usage:
    python scripts/make_figures.py --results-dir results --metric roc_auc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- consistent ordering / labels / colors --------------------------------
BIN_TRACKS = ["track1a_gene", "track1b_isoform", "track2_gene", "track3_full", "track3_full_isoform"]
TRACK_LABEL = {
    "track1a_gene": "3'UTR\ngene", "track1b_isoform": "3'UTR\nisoform",
    "track2_gene": "CDS\ngene", "track3_full": "full\ngene",
    "track3_full_isoform": "full\nisoform",
}
# canonical model display order (subset present per track is plotted)
MODEL_ORDER = ["kmer", "length", "engineered", "utrbert", "dnabert2", "rnafm", "mrnafm",
               "rnatracker", "dm3loc", "fusion_rnafm_eng", "fusion_mrnafm_eng"]
MODEL_LABEL = {
    "kmer": "k-mer", "length": "length", "engineered": "engineered",
    "utrbert": "UTR-BERT", "dnabert2": "DNABERT-2", "rnafm": "RNA-FM", "mrnafm": "mRNA-FM",
    "rnatracker": "RNATracker", "dm3loc": "DM3Loc",
    "fusion_rnafm_eng": "fusion", "fusion_mrnafm_eng": "fusion",
}
COMPARTMENTS = ["Cell_body", "Dendrite", "Neuropil", "Axon", "Neurite"]
C_FUS, C_KMER, C_FM, C_ENG, C_NET = "#d1495b", "#8d99ae", "#3a7ca5", "#5c946e", "#9b6a9e"


def _save(fig, out_base):
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_base}.png / .pdf")


def _color(model):
    if model.startswith("fusion"):
        return C_FUS
    if model == "kmer":
        return C_KMER
    if model in ("rnatracker", "dm3loc"):
        return C_NET
    if model in ("length", "engineered"):
        return C_ENG
    return C_FM


def fig1(summ, boot, metric, outdir):
    col = f"test_{metric}"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.4),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    # (a) per-setting model bars
    sub = summ[summ.label_scheme == "soma_vs_neurite"]
    xticks, xlabels = [], []
    x = 0
    for t in BIN_TRACKS:
        d = sub[sub.track == t]
        present = [m for m in MODEL_ORDER if m in set(d.model)]
        vals = [float(d[d.model == m][col].iloc[0]) for m in present]
        kmer_v = float(d[d.model == "kmer"][col].iloc[0]) if "kmer" in set(d.model) else np.nan
        for m, v in zip(present, vals):
            axA.bar(x, v, color=_color(m), width=0.9,
                    edgecolor="black", linewidth=0.3)
            x += 1
        if np.isfinite(kmer_v):
            axA.plot([x - len(present) - 0.4, x - 0.6], [kmer_v, kmer_v],
                     "--", color="black", lw=0.8, zorder=5)
        xticks.append(x - len(present) / 2 - 0.5)
        xlabels.append(TRACK_LABEL[t])
        x += 1.5
    axA.set_xticks(xticks); axA.set_xticklabels(xlabels)
    axA.set_ylabel(metric.upper().replace("_", "-"))
    axA.set_ylim(0.5, max(0.8, sub[col].max() + 0.03))
    axA.set_title("a  All models per setting (dashed = k-mer)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_KMER, C_FM, C_NET, C_ENG, C_FUS)]
    axA.legend(handles, ["k-mer", "foundation model", "task-specific net", "length/engineered", "fusion"],
               fontsize=7, ncol=2, loc="upper left")

    # (b) best single FM - kmer, with CI (not significant)
    bf = boot[(boot.group == "bestFM_vs_kmer") & (boot.metric == metric)]
    bf = bf.set_index("track").reindex(BIN_TRACKS).reset_index()
    y = np.arange(len(bf))[::-1]
    axB.errorbar(bf["diff"], y, xerr=[bf["diff"] - bf.ci_lo, bf.ci_hi - bf["diff"]],
                 fmt="o", color=C_FM, capsize=3)
    axB.axvline(0, color="black", lw=0.8)
    axB.set_yticks(y); axB.set_yticklabels([TRACK_LABEL[t].replace("\n", " ") for t in bf.track], fontsize=8)
    axB.set_xlabel(f"Δ{metric.upper().replace('_','-')}  (best FM − k-mer)")
    axB.set_title("b  Foundation models do not beat k-mer")
    _save(fig, str(outdir / f"fig1_benchmark_baseline_{metric}"))


def fig2(summ, boot, metric, outdir):
    col = f"test_{metric}"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.4))
    # (a) fusion - kmer with CI (all significant)
    fk = boot[(boot.group == "fusion_vs_kmer") & (boot.metric == metric)]
    fk = fk.set_index("track").reindex(BIN_TRACKS).reset_index()
    y = np.arange(len(fk))[::-1]
    axA.errorbar(fk["diff"], y, xerr=[fk["diff"] - fk.ci_lo, fk.ci_hi - fk["diff"]],
                 fmt="o", color=C_FUS, capsize=3)
    axA.axvline(0, color="black", lw=0.8)
    axA.set_yticks(y); axA.set_yticklabels([TRACK_LABEL[t].replace("\n", " ") for t in fk.track], fontsize=8)
    axA.set_xlabel(f"Δ{metric.upper().replace('_','-')}  (fusion − k-mer)")
    axA.set_title("a  Fusion vs k-mer (all significant)")
    for yi, (_, r) in zip(y, fk.iterrows()):
        axA.text(r.ci_hi + 0.002, yi, "*" if r.significant else "ns", va="center", fontsize=9)

    # (b) component decomposition (full + isoform 3'UTR)
    decomp = {"track3_full": "full gene", "track1b_isoform": "3'UTR isoform"}
    comp_models = ["length", "engineered", "rnafm", "fusion_rnafm_eng"]
    comp_lab = ["length", "engineered", "RNA-FM", "fusion"]
    width = 0.38
    for i, (t, name) in enumerate(decomp.items()):
        d = summ[summ.track == t]
        vals = []
        for m in comp_models:
            row = d[d.model == m]
            vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
        xs = np.arange(len(comp_models)) + (i - 0.5) * width
        bars = axB.bar(xs, vals, width=width, label=name,
                       color=[C_ENG, C_ENG, C_FM, C_FUS], alpha=0.55 + 0.45 * i,
                       edgecolor="black", linewidth=0.3)
    axB.set_xticks(np.arange(len(comp_models)))
    axB.set_xticklabels(comp_lab)
    axB.set_ylabel(metric.upper().replace("_", "-"))
    axB.set_ylim(0.5, 0.78)
    axB.set_title("b  Component decomposition (light=full, dark=3'UTR iso)")
    _save(fig, str(outdir / f"fig2_fusion_model_{metric}"))


def fig3(summ, boot, metric, outdir):
    col = f"test_{metric}"
    fig, ax = plt.subplots(figsize=(5, 4.2))
    d = summ[summ.track == "ablation_region"].set_index("model")
    regions = [("rnafm_utr3", "3'UTR"), ("rnafm_cds", "CDS"), ("rnafm_full", "full")]
    vals = [float(d.loc[m, col]) if m in d.index else np.nan for m, _ in regions]
    ax.bar([r[1] for r in regions], vals, color=[C_FM, C_FM, C_FUS],
           edgecolor="black", linewidth=0.4)
    ax.set_ylabel(metric.upper().replace("_", "-"))
    ax.set_ylim(0.5, max(0.7, np.nanmax(vals) + 0.03))
    ax.set_title("Region ablation (matched genes, RNA-FM)")
    ra = boot[(boot.group == "region_ablation") & (boot.metric == metric)]
    txt = []
    for _, r in ra.iterrows():
        txt.append(f"{r.A.replace('rnafm_','')} vs {r.B.replace('rnafm_','')}: "
                   f"Δ={r['diff']:+.3f} p={r.p:.3f}")
    ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=7.5)
    _save(fig, str(outdir / f"fig3_region_ablation_{metric}"))


def fig4(summ, boot, metric, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    fc = boot[(boot.group == "fine_fusion_vs_kmer") & (boot.metric == metric)]
    settings = [("fine_full_gene", "gene", -0.2, C_FM), ("fine_full_isoform", "isoform", 0.2, C_FUS)]
    ybase = np.arange(len(COMPARTMENTS))[::-1]
    for track, name, off, color in settings:
        d = fc[fc.track == track].set_index("label").reindex(COMPARTMENTS).reset_index()
        ax.errorbar(d["diff"], ybase + off,
                    xerr=[d["diff"] - d.ci_lo, d.ci_hi - d["diff"]],
                    fmt="o", color=color, capsize=3, label=name)
        for yi, (_, r) in zip(ybase + off, d.iterrows()):
            if pd.notna(r.p):
                ax.text(r.ci_hi + 0.003, yi, "*" if r.significant else "", va="center", fontsize=10)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ybase); ax.set_yticklabels(COMPARTMENTS)
    ax.set_xlabel(f"Δ{metric.upper().replace('_','-')}  (fusion − k-mer)")
    ax.set_title("Multi-label: fusion vs k-mer per compartment")
    ax.legend(title="resolution", fontsize=8)
    _save(fig, str(outdir / f"fig4_multilabel_{metric}"))


def fig_combined(summ, boot, outdir):
    """One big figure: rows = ROC-AUC / AUPRC; cols = all-model bars / fusion−k-mer Δ."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.4),
                             gridspec_kw={"width_ratios": [2.4, 1]})
    panel = iter("abcd")
    for row, metric in enumerate(["roc_auc", "pr_auc"]):
        col = f"test_{metric}"
        mlabel = metric.upper().replace("_", "-")
        axL, axR = axes[row]

        # left: per-setting all-model bars
        sub = summ[summ.label_scheme == "soma_vs_neurite"]
        xticks, xlabels, x = [], [], 0
        for t in BIN_TRACKS:
            d = sub[sub.track == t]
            present = [m for m in MODEL_ORDER if m in set(d.model)]
            kmer_v = float(d[d.model == "kmer"][col].iloc[0]) if "kmer" in set(d.model) else np.nan
            for m in present:
                axL.bar(x, float(d[d.model == m][col].iloc[0]), color=_color(m),
                        width=0.9, edgecolor="black", linewidth=0.3)
                x += 1
            if np.isfinite(kmer_v):
                axL.plot([x - len(present) - 0.4, x - 0.6], [kmer_v, kmer_v], "--",
                         color="black", lw=0.8, zorder=5)
            xticks.append(x - len(present) / 2 - 0.5); xlabels.append(TRACK_LABEL[t]); x += 1.5
        axL.set_xticks(xticks); axL.set_xticklabels(xlabels)
        axL.set_ylabel(mlabel); axL.set_ylim(0.5, max(0.8, sub[col].max() + 0.03))
        axL.set_title(f"{next(panel)}  All models per setting — {mlabel} (dashed = k-mer)")
        if row == 0:
            h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_KMER, C_FM, C_NET, C_ENG, C_FUS)]
            axL.legend(h, ["k-mer", "foundation model", "task-specific net", "length/engineered", "fusion"],
                       fontsize=7, ncol=2, loc="upper left")

        # right: fusion − kmer Δ ± CI
        fk = boot[(boot.group == "fusion_vs_kmer") & (boot.metric == metric)]
        fk = fk.set_index("track").reindex(BIN_TRACKS).reset_index()
        y = np.arange(len(fk))[::-1]
        axR.errorbar(fk["diff"], y, xerr=[fk["diff"] - fk.ci_lo, fk.ci_hi - fk["diff"]],
                     fmt="o", color=C_FUS, capsize=3)
        axR.axvline(0, color="black", lw=0.8)
        axR.set_yticks(y); axR.set_yticklabels([TRACK_LABEL[t].replace("\n", " ") for t in fk.track], fontsize=8)
        axR.set_xlabel(f"Δ{mlabel}  (fusion − k-mer)")
        axR.set_title(f"{next(panel)}  Fusion vs k-mer")
        for yi, (_, r) in zip(y, fk.iterrows()):
            axR.text(r.ci_hi + 0.002, yi, "*" if r.significant else "ns", va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, str(outdir / "fig_main_combined_roc_pr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc"])
    ap.add_argument("--combined", action="store_true",
                    help="also emit one big ROC+PR figure (fig_main_combined_roc_pr)")
    args = ap.parse_args()

    summ = pd.read_csv(args.results_dir / "summary_table.csv")
    boot = pd.read_csv(args.results_dir / "bootstrap_all.csv")
    outdir = args.results_dir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[figures] metric={args.metric} -> {outdir}")
    fig1(summ, boot, args.metric, outdir)
    fig2(summ, boot, args.metric, outdir)
    fig3(summ, boot, args.metric, outdir)
    fig4(summ, boot, args.metric, outdir)
    if args.combined:
        fig_combined(summ, boot, outdir)
    print("done.")


if __name__ == "__main__":
    main()

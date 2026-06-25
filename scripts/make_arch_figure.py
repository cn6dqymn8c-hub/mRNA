#!/usr/bin/env python3
"""
Schematic of the proposed attention-gated fusion model (no data needed).

Input transcript (5'UTR–CDS–3'UTR) → two complementary views: a frozen
foundation-model embedding (sliding-window mean-pool) and an interpretable
engineered-feature vector → per-view projection to a shared space → an
input-dependent gate (softmax over views) whose weights scale each pathway →
weighted sum → MLP head → per-compartment localization probability.

Writes results/figures/fig_architecture.(png|pdf).
Usage:  python scripts/make_arch_figure.py --out results/figures/fig_architecture
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

C_FM, C_ENG, C_PROJ, C_GATE, C_FUS, C_HEAD = "#2f6f95", "#4f8a5b", "#cfe3ef", "#f4d35e", "#d1495b", "#dadada"
C_5U, C_CDS, C_3U = "#9ecae1", "#fdae6b", "#a1d99b"
GATE_W = (0.62, 0.38)  # illustrative gate weights (FM view / engineered view)


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, ec="black"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.4,rounding_size=2.5",
                                linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)


def arrow(ax, x1, y1, x2, y2, lw=1.4, color="black", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color,
                                 linestyle=ls, zorder=2))


def dim(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5, color="#555",
            style="italic", zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figures/fig_architecture"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- input: transcript cartoon (5'UTR–CDS–3'UTR) ----
    seg = [("5'UTR", C_5U, 3), ("CDS", C_CDS, 7), ("3'UTR", C_3U, 4)]
    x0, ytop = 2.5, 52
    for name, c, wsег in seg:
        ax.add_patch(Rectangle((x0, ytop - 2.5), wsег, 5, facecolor=c, edgecolor="black", lw=0.8, zorder=3))
        ax.text(x0 + wsег / 2, ytop, name, ha="center", va="center", fontsize=6.5, zorder=4)
        x0 += wsег
    ax.text(9.5, 58, "transcript\n(≤ 31 kb)", ha="center", fontsize=8, fontweight="bold")
    ax.annotate("", xy=(2.5, 47.5), xytext=(16.5, 47.5),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#999"))

    # ---- two complementary views ----
    box(ax, 31, 76, 23, 19,
        "Foundation-model embedding\n(RNA-FM / mRNA-FM, FROZEN)\nsliding-window over full\nsequence → token mean-pool", C_FM, fs=8)
    ax.text(31, 76, "", )  # spacer
    box(ax, 31, 24, 23, 19,
        "Engineered features\nlength · GC · dinucleotide\n· localization motifs\n(ARE/CPE/polyU/G-quad…)", C_ENG, fs=8)
    arrow(ax, 16.5, 53, 19.3, 72); dim(ax, 17.5, 64, "")
    arrow(ax, 16.5, 47, 19.3, 28)

    # ---- per-view projection ----
    box(ax, 54, 76, 15, 11, "Linear→ReLU\n→LayerNorm", C_PROJ, fs=8)
    box(ax, 54, 24, 15, 11, "Linear→ReLU\n→LayerNorm", C_PROJ, fs=8)
    arrow(ax, 42.5, 76, 46.5, 76); dim(ax, 44.5, 79, "d≈640")
    arrow(ax, 42.5, 24, 46.5, 24); dim(ax, 44.5, 27, "d≈30")
    dim(ax, 64, 79.5, "h₁ (128)")
    dim(ax, 64, 28.5, "h₂ (128)")

    # ---- gate (focal): takes both projections, outputs softmax weights ----
    box(ax, 73, 50, 14, 13, "GATE\nLinear → softmax\nover views", C_GATE, fs=8, bold=True)
    arrow(ax, 61.5, 73, 67, 55, lw=1.0, color="#555")
    arrow(ax, 61.5, 27, 67, 45, lw=1.0, color="#555")
    # little softmax weight bars inside/under the gate
    ax.add_patch(Rectangle((69.5, 40.5), 3.0 * GATE_W[0] / 0.62 * 0.62, 1.6, facecolor=C_FM, edgecolor="black", lw=0.4, zorder=4))
    ax.add_patch(Rectangle((69.5, 38.4), 3.0 * GATE_W[1] / 0.62 * 0.62, 1.6, facecolor=C_ENG, edgecolor="black", lw=0.4, zorder=4))
    ax.text(78.2, 41.3, f"w₁={GATE_W[0]:.2f}", fontsize=6, va="center", color=C_FM)
    ax.text(78.2, 39.2, f"w₂={GATE_W[1]:.2f}", fontsize=6, va="center", color=C_ENG)

    # ---- weighted sum: pathways scaled by gate weights (arrow thickness ∝ weight) ----
    box(ax, 90, 62, 13, 12, "Weighted sum\nΣ wᵢ·hᵢ", C_FUS, fs=8, bold=True)
    arrow(ax, 61.5, 76, 84.5, 64, lw=1.0 + 5 * GATE_W[0], color=C_FM)     # FM pathway (thick)
    arrow(ax, 61.5, 24, 84.5, 60, lw=1.0 + 5 * GATE_W[1], color=C_ENG)    # engineered pathway (thin)
    arrow(ax, 73, 43.5, 88, 56, lw=0.9, ls="--", color="#a08000")        # gate → weights
    dim(ax, 80, 52, "weights")

    # ---- MLP head → output (right column, straight down) ----
    box(ax, 90, 38, 15, 12, "MLP head\nLN→Dropout→Linear\n→ReLU→Dropout→Linear", C_HEAD, fs=8)
    arrow(ax, 90, 56, 90, 44)
    box(ax, 90, 14, 17, 12, "Per-compartment\nlocalization probability\n(soma / neurite; 5-class)", "#ffffff", fs=8, bold=True)
    arrow(ax, 90, 32, 90, 20)

    ax.text(50, 97.5, "Attention-gated late fusion of complementary sequence views",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(31, 64.5, "learned representation", ha="center", fontsize=7, color=C_FM, style="italic")
    ax.text(31, 11.5, "interpretable representation", ha="center", fontsize=7, color=C_ENG, style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Schematic of the proposed attention-gated fusion model (no data needed).

Transcript sequence → two views (frozen foundation-model embedding +
interpretable engineered features) → per-view projection → input-dependent gate
(softmax over views) → weighted sum → MLP head → per-compartment localization
probability. Writes results/figures/fig_architecture.(png|pdf).

Usage:  python scripts/make_arch_figure.py --out results/figures/fig_architecture
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

C_FM, C_ENG, C_PROJ, C_GATE, C_FUS, C_HEAD = "#3a7ca5", "#5c946e", "#cde", "#f4d35e", "#d1495b", "#e0e0e0"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.4,rounding_size=2",
                                linewidth=1.0, edgecolor="black", facecolor=fc, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=3)


def arrow(ax, x1, y1, x2, y2, style="-|>", lw=1.3, color="black", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, lw=lw, color=color,
                                 linestyle=ls, zorder=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/figures/fig_architecture"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # input
    box(ax, 9, 50, 15, 14, "Transcript\nsequence\n(≤ 31 kb)", "#ffffff", fs=9, bold=True)

    # two views
    box(ax, 31, 74, 22, 18,
        "Foundation-model\nembedding\n(RNA-FM / mRNA-FM,\nfrozen; sliding-window\nmean-pool → d≈640)", C_FM, fs=8)
    box(ax, 31, 26, 22, 18,
        "Engineered features\n(length, GC,\ndinucleotide,\nlocalization motifs;\n→ d≈30)", C_ENG, fs=8)
    arrow(ax, 16.5, 53, 20, 70)
    arrow(ax, 16.5, 47, 20, 30)

    # per-view projection
    box(ax, 53, 74, 16, 12, "Linear→ReLU\n→LayerNorm\n(→ 128)", C_PROJ, fs=8)
    box(ax, 53, 26, 16, 12, "Linear→ReLU\n→LayerNorm\n(→ 128)", C_PROJ, fs=8)
    arrow(ax, 42, 74, 45, 74)
    arrow(ax, 42, 26, 45, 26)

    # gate
    box(ax, 72, 50, 16, 14, "Gate\nLinear→softmax\nover views\n(per-transcript\nweights w₁,w₂)", C_GATE, fs=8)
    arrow(ax, 61, 72, 66, 55)
    arrow(ax, 61, 28, 66, 45)

    # weighted sum (right column)
    box(ax, 88, 56, 13, 12, "Weighted\nsum\nΣ wᵢ·hᵢ\n(→ 128)", C_FUS, fs=8)
    arrow(ax, 80, 50, 81.5, 54)
    # projected view vectors hᵢ also feed the weighted sum (dashed)
    arrow(ax, 61, 74, 84, 60, lw=0.9, ls="--", color="#888")
    arrow(ax, 61, 26, 84, 52, lw=0.9, ls="--", color="#888")

    # MLP head → output, straight down the right column (no collision)
    box(ax, 88, 33, 15, 13, "MLP head\nLN→Drop→\nLinear→ReLU→\nDrop→Linear", C_HEAD, fs=8)
    arrow(ax, 88, 50, 88, 40)
    box(ax, 88, 11, 17, 12, "Per-compartment\nlocalization probability\n(soma / neurite;\n5-class fine)", "#ffffff", fs=8, bold=True)
    arrow(ax, 88, 26.5, 88, 17.5)

    ax.text(50, 97, "Proposed model: attention-gated late fusion of complementary sequence views",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(72, 40, "gate weights wᵢ", ha="center", fontsize=7, color="#a08000", style="italic")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Programmatic pipeline-overview schematic (A–E) for the neuronal mRNA
localization predictor. Editable starting point; not a data plot.
Outputs results/figures/fig_overview_schematic.(png|pdf).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {  # legend colours (match the design language)
    "fm": "#cfe0f3", "fm_e": "#3a7ca5",
    "eng": "#d6ecd6", "eng_e": "#5c946e",
    "oh": "#e7d9ef", "oh_e": "#9b6a9e",
    "fus": "#fdebcf", "fus_e": "#e0922f",
    "out": "#f6d6d6", "out_e": "#c0504d",
    "panel": "#f7f9fb", "panel_e": "#9aa7b2", "ink": "#222222",
}

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def panel(x, y, w, h, title, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=C["panel"], ec=ec, lw=1.8))
    ax.text(x + 1.5, y + h - 1.6, title, fontsize=11, fontweight="bold", color=ec, va="top")


def box(x, y, w, h, text, fc, ec, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center", color=C["ink"])


def arrow(x1, y1, x2, y2, style="-|>", ls="-", lw=1.6, col=C["ink"]):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 lw=lw, color=col, linestyle=ls,
                                 connectionstyle="arc3,rad=0"))


ax.text(50, 98, "Hybrid multi-view architecture for neuronal mRNA localization prediction",
        fontsize=15, fontweight="bold", ha="center")

# ---------------- A. Curated atlas + partitioning ----------------
panel(1, 63, 47, 32, "A  Curated atlas & leakage-safe partitioning", C["panel_e"])
box(3, 84, 26, 8, "Curated neuronal mRNA localization atlas\n(sequence · source · species · gene · transcript · label)",
    "white", C["panel_e"], 7.5)
box(3, 74, 12, 8, "18 sources\nmouse / rat / human\n(RNA-seq · Ribo-seq · MPRA · 3'-seq)", "white", C["panel_e"], 6.8)
box(16, 74, 13, 8, "Annotation schema\nmultilabel · soft labels\n+ per-sample source mask", "white", C["panel_e"], 6.8)
box(31, 86, 16, 6, "Train 70%  (ortholog groups A,C,E…)", C["eng"], C["eng_e"], 7)
box(31, 79, 16, 6, "Val 15%  (groups B,F…)", C["fus"], C["fus_e"], 7)
box(31, 72, 16, 6, "Test 15%  (groups D,G…)", C["out"], C["out_e"], 7)
box(3, 64, 44, 8, "Leakage-safe grouping: ortholog group · gene · identical sequence (union-find)  →  frozen 70/15/15",
    "white", C["panel_e"], 7.5)

# ---------------- B. Preprocessing ----------------
panel(50, 63, 49, 32, "B  Label / sequence preprocessing", C["panel_e"])
steps = ["Synap→\nNeuropil", "Label\nharmoniz.", "Region\nutr3/cds/full", "Windowing\nsegmentation", "Pad /\ntruncate", "Standardize\n(z-score)"]
for i, s in enumerate(steps):
    box(51.5 + i * 7.8, 84, 7.2, 8, s, "white", C["panel_e"], 6.6)
    if i:
        arrow(51.5 + i * 7.8 - 0.6, 88, 51.5 + i * 7.8, 88)
box(51.5, 73, 22, 7, "Leakage-safe safeguards\nunknown ≠ negative · ortholog groups", "white", C["panel_e"], 6.8)
box(75, 78, 11, 4.5, "FM sequences", C["fm"], C["fm_e"], 7)
box(75, 73, 11, 4.5, "Engineered features", C["eng"], C["eng_e"], 7)
box(87, 75.5, 11, 4.5, "One-hot sequences", C["oh"], C["oh_e"], 7)
box(51.5, 64, 47, 7, "Outputs to the three encoders (same data / split / labels — only the encoder differs)",
    "white", C["panel_e"], 7.5)

# ---------------- C. Three-view encoding ----------------
panel(1, 33, 98, 28, "C  Multi-view encoding — three complementary perspectives", C["panel_e"])
# FM view
box(3, 36, 30, 22, "", C["fm"], C["fm_e"])
ax.text(18, 57, "Foundation-model view", fontsize=9.5, fontweight="bold", ha="center", color=C["fm_e"])
box(4.5, 47, 9, 7, "Sliding windows\n→ tokenizer", "white", C["fm_e"], 6.5)
box(15, 47, 9, 7, "Pretrained encoder\nRNA-FM / mRNA-FM\n/ DNABERT-2", "white", C["fm_e"], 6.3)
box(25.5, 47, 6.5, 7, "mean /\nattn pool", "white", C["fm_e"], 6.3)
box(11, 38, 14, 5, "z_FM  (frozen embedding, d_FM)", "white", C["fm_e"], 7)
# Handcrafted view
box(34.5, 36, 30, 22, "", C["eng"], C["eng_e"])
ax.text(49.5, 57, "Handcrafted view", fontsize=9.5, fontweight="bold", ha="center", color=C["eng_e"])
for i, t in enumerate(["Length /\ncomposition", "GC /\nk-mer", "Motif /\nRBP / struct", "Region\ndescriptors"]):
    box(35.5 + i * 7.1, 47, 6.6, 7, t, "white", C["eng_e"], 6.2)
box(40, 38, 19, 5, "z_eng  (standardized features, d_eng)", "white", C["eng_e"], 7)
# One-hot view
box(66, 36, 32, 22, "", C["oh"], C["oh_e"])
ax.text(82, 57, "One-hot view / task-specific net", fontsize=9.5, fontweight="bold", ha="center", color=C["oh_e"])
box(67.5, 47, 8, 7, "One-hot\n(4×L)", "white", C["oh_e"], 6.5)
box(77, 47, 12, 7, "Conv1D + BiGRU/Attn\n(RNATracker / DM3Loc)", "white", C["oh_e"], 6.0)
box(90, 47, 7, 7, "auxiliary\noutputs", "white", C["oh_e"], 6.2)
box(73, 38, 18, 5, "z_task / auxiliary predictions", "white", C["oh_e"], 7)
arrow(18, 47, 18, 43); arrow(49.5, 47, 49.5, 43); arrow(82, 47, 82, 43)

# ---------------- D. Fusion ----------------
panel(1, 3, 52, 28, "D  Attention-gated late fusion", C["fus_e"])
box(3, 20, 8, 5, "z_FM", C["fm"], C["fm_e"], 7)
box(3, 11, 8, 5, "z_eng", C["eng"], C["eng_e"], 7)
box(13, 20, 9, 5, "Projection (MLP) → h_FM", "white", C["fus_e"], 6.5)
box(13, 11, 9, 5, "Projection (MLP) → h_eng", "white", C["fus_e"], 6.5)
box(24, 13.5, 14, 10,
    "Attention gating\nα = softmax(W[h_FM;h_eng]+b)\nh = α_FM·h_FM + α_eng·h_eng", C["fus"], C["fus_e"], 6.5)
box(40, 15, 11, 7, "Multi-label\nsigmoid classifier", "white", C["fus_e"], 7)
arrow(11, 22.5, 13, 22.5); arrow(11, 13.5, 13, 13.5)
arrow(22, 22.5, 24, 18.5); arrow(22, 13.5, 24, 16)
arrow(38, 18.5, 40, 18.5)

# ---------------- E. Outputs & objectives ----------------
panel(54, 3, 45, 28, "E  Outputs, objectives & evaluation", C["out_e"])
box(56, 18, 19, 11,
    "Output heads (multilabel)\n1) binary soma vs neurite\n2) 5 compartments:\nCell body·Dendrite·Neuropil·Axon·Neurite",
    C["out"], C["out_e"], 6.5)
box(77, 22, 20, 7, "Training: masked BCE + pos_weight\nearly stop on val macro-AUC", "white", C["out_e"], 6.5)
box(77, 13.5, 20, 7, "Leakage-safe eval:\nthresholds on val, applied to test\ngroup bootstrap (split_group)", "white", C["out_e"], 6.3)
box(56, 6, 41, 9,
    "Metrics:  AUROC ↑   AUPRC ↑   F1 micro/macro ↑   Hamming ↓   subset-acc ↑   (ROC + PR dual-metric, Bonferroni)",
    "white", C["out_e"], 7)

# ---- inter-panel main-flow arrows ----
arrow(24, 63, 24, 61, lw=2.2)      # A -> C
arrow(74, 63, 74, 61, lw=2.2)      # B -> C
arrow(24, 33, 26, 31, lw=2.2)      # C(FM) -> D
arrow(49.5, 33, 30, 24, lw=2.2)    # C(eng) -> D
arrow(82, 33, 70, 24, ls="--", lw=1.6, col=C["oh_e"])  # one-hot -> E (aux, dashed)
arrow(51, 18.5, 56, 23, lw=2.2)    # D -> E

# legend
for i, (k, lab) in enumerate([("fm", "FM view"), ("eng", "Handcrafted"), ("oh", "One-hot / task net"),
                              ("fus", "Fusion"), ("out", "Output / training")]):
    ax.add_patch(FancyBboxPatch((3 + i * 13, 0.2), 1.6, 1.6, boxstyle="round,pad=0.1",
                                fc=C[k], ec=C[k + "_e"]))
    ax.text(5 + i * 13, 1.0, lab, fontsize=7.5, va="center")
ax.text(80, 1.0, "→ main flow    ⇢ auxiliary flow", fontsize=7.5, va="center")

plt.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"results/figures/fig_overview_schematic.{ext}", dpi=200, bbox_inches="tight")
print("wrote results/figures/fig_overview_schematic.png/pdf")

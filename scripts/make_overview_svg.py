#!/usr/bin/env python3
"""Polished SVG pipeline-overview schematic (gradients / soft shadows / rounded
boxes), rendered to PNG via headless Chromium. Editable vector source.
Outputs results/figures/fig_overview_schematic.svg (+ .png via render step)."""
import os

W, H = 1480, 1040
P = []  # svg fragments

GRAD = {  # (light, dark, border)
    "fm":  ("#eaf2fb", "#cfe0f3", "#3a7ca5"),
    "eng": ("#edf7ed", "#d6ecd6", "#5c946e"),
    "oh":  ("#f3ecf8", "#e7d9ef", "#9b6a9e"),
    "fus": ("#fef4e3", "#fdebcf", "#e0922f"),
    "out": ("#fbe9e9", "#f6d6d6", "#c0504d"),
    "pan": ("#ffffff", "#f4f7fa", "#9aa7b2"),
}

def defs():
    s = ['<defs>']
    for k, (l, d, _) in GRAD.items():
        s.append(f'<linearGradient id="g_{k}" x1="0" y1="0" x2="0" y2="1">'
                 f'<stop offset="0" stop-color="{l}"/><stop offset="1" stop-color="{d}"/></linearGradient>')
    s.append('<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
             '<feDropShadow dx="0" dy="1.6" stdDeviation="2.2" flood-color="#000" flood-opacity="0.16"/></filter>')
    s.append('<marker id="arr" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
             '<path d="M0,0 L9,4.5 L0,9 z" fill="#3b4654"/></marker>')
    s.append('<marker id="arrd" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto">'
             '<path d="M0,0 L9,4.5 L0,9 z" fill="#9b6a9e"/></marker>')
    s.append('</defs>')
    return "".join(s)

def esc(t): return t.replace("&", "&amp;").replace("<", "&lt;")

def text(x, y, t, size=13, w="normal", col="#222", anchor="middle"):
    lines = t.split("\n")
    out = [f'<text x="{x}" y="{y-(len(lines)-1)*size*0.62}" font-family="Helvetica,Arial,sans-serif" '
           f'font-size="{size}" font-weight="{w}" fill="{col}" text-anchor="{anchor}">']
    for i, ln in enumerate(lines):
        out.append(f'<tspan x="{x}" dy="{0 if i==0 else size*1.24}">{esc(ln)}</tspan>')
    out.append('</text>')
    P.append("".join(out))

def panel(x, y, w, h, title, fam="pan"):
    _, _, b = GRAD[fam]
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="url(#g_{fam})" '
             f'stroke="{b}" stroke-width="1.8" filter="url(#sh)"/>')
    P.append(f'<text x="{x+16}" y="{y+24}" font-family="Helvetica,Arial,sans-serif" font-size="15" '
             f'font-weight="700" fill="{b}">{esc(title)}</text>')

def box(x, y, w, h, t, fam="pan", size=11.5):
    _, _, b = GRAD[fam]
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="url(#g_{fam})" '
             f'stroke="{b}" stroke-width="1.3" filter="url(#sh)"/>')
    text(x+w/2, y+h/2+size*0.34, t, size, "normal", "#243", "middle")

def whitebox(x, y, w, h, t, fam="pan", size=11.5):
    _, _, b = GRAD[fam]
    P.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#ffffff" '
             f'stroke="{b}" stroke-width="1.2" filter="url(#sh)"/>')
    text(x+w/2, y+h/2+size*0.34, t, size, "normal", "#243", "middle")

def arrow(x1, y1, x2, y2, dashed=False, col="#3b4654"):
    da = ' stroke-dasharray="6 5"' if dashed else ''
    mk = 'arrd' if dashed else 'arr'
    P.append(f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{col}" stroke-width="2.4" fill="none"{da} '
             f'marker-end="url(#{mk})"/>')

# ---------- build ----------
P.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
P.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
P.append(defs())
text(W/2, 36, "Hybrid multi-view architecture for neuronal mRNA localization prediction", 22, "700", "#1b2733")

# A
panel(24, 56, 690, 300, "A   Curated atlas & leakage-safe partitioning")
whitebox(44, 92, 380, 60, "Curated neuronal mRNA localization atlas\nsequence · source · species · gene · transcript · label", size=11)
whitebox(44, 168, 182, 70, "17 published sources · mouse / rat / human\nRNA-seq · Ribo-seq · MPRA · 3'-seq", size=10.5)
whitebox(238, 168, 186, 70, "Annotation schema\nmultilabel · soft (continuous) labels\n+ per-sample source mask", size=10.5)
box(452, 96, 240, 40, "Train  70%   (ortholog groups A, C, E …)", "eng", 12)
box(452, 146, 240, 40, "Validation  15%   (groups B, F …)", "fus", 12)
box(452, 196, 240, 40, "Test  15%   (groups D, G …)", "out", 12)
whitebox(44, 256, 648, 56, "Leakage-safe grouping:  ortholog group · gene · identical sequence (union-find)   →   frozen 70 / 15 / 15 split", size=11.5)

# B
panel(742, 56, 714, 300, "B   Label / sequence preprocessing")
steps = ["Label\nharmonization", "Region extract\nfull / CDS / 3'UTR", "Windowing /\nsegmentation", "Pad /\ntruncate", "Standardize\n(z-score)"]
for i, st in enumerate(steps):
    x = 762 + i*132
    whitebox(x, 92, 100, 66, st, size=10.3)
    if i: arrow(x-12, 125, x-2, 125)
whitebox(762, 184, 320, 60, "Leakage-safe safeguards\nunknown ≠ negative · ortholog groups", size=10.8)
box(1100, 176, 150, 34, "FM sequences", "fm", 11)
box(1100, 216, 150, 34, "Engineered features", "eng", 11)
box(1262, 196, 174, 34, "One-hot sequences", "oh", 11)
whitebox(762, 268, 674, 56, "Outputs to the three encoders  —  same data / split / labels, only the encoder differs", size=11.5)

# C
panel(24, 376, 1432, 300, "C   Multi-view encoding — three complementary perspectives")
# FM
box(44, 410, 440, 250, "", "fm")
text(264, 432, "Foundation-model view", 14, "700", GRAD["fm"][2])
whitebox(60, 470, 120, 64, "Sliding windows\n→ tokenizer", "fm", 10.5)
whitebox(196, 470, 150, 64, "Pretrained encoder\nRNA-FM / mRNA-FM /\nUTR-BERT / DNABERT-2", "fm", 9.6)
whitebox(360, 470, 108, 64, "mean / attn\npooling", "fm", 10.5)
whitebox(150, 576, 230, 50, "z_FM   (frozen embedding, d_FM)", "fm", 12)
# Eng
box(508, 410, 440, 250, "", "eng")
text(728, 432, "Handcrafted view", 14, "700", GRAD["eng"][2])
for i, t in enumerate(["Length /\ncomposition", "GC /\nk-mer", "Motif / RBP /\nstructure", "Region\ndescriptors"]):
    whitebox(524+i*104, 470, 94, 64, t, "eng", 10)
whitebox(620, 576, 230, 50, "z_eng   (standardized features, d_eng)", "eng", 12)
# OH
box(972, 410, 462, 250, "", "oh")
text(1203, 432, "One-hot view / task-specific net", 14, "700", GRAD["oh"][2])
whitebox(988, 470, 110, 64, "One-hot\n(4 × L)", "oh", 10.5)
whitebox(1114, 470, 180, 64, "Conv1D + BiGRU + Attn\n(RNATracker / DM3Loc)", "oh", 9.6)
whitebox(1310, 470, 108, 64, "auxiliary\noutputs", "oh", 10.5)
whitebox(1090, 576, 250, 50, "z_task  /  auxiliary predictions", "oh", 12)
arrow(264, 534, 264, 574); arrow(728, 534, 728, 574); arrow(1203, 534, 1203, 574)

# D
panel(24, 696, 760, 320, "D   Attention-gated late fusion", "fus")
box(44, 740, 120, 50, "z_FM", "fm", 12)
box(44, 826, 120, 50, "z_eng", "eng", 12)
whitebox(192, 740, 168, 50, "Projection (MLP) → h_FM", "fus", 10.5)
whitebox(192, 826, 168, 50, "Projection (MLP) → h_eng", "fus", 10.5)
box(392, 768, 210, 90, "Attention gating\nα = softmax(W[h_FM ; h_eng] + b)\nh = α_FM·h_FM + α_eng·h_eng", "fus", 10.5)
whitebox(636, 786, 128, 54, "Multi-label\nsigmoid classifier", "fus", 11)
arrow(164, 765, 192, 765); arrow(164, 851, 192, 851)
arrow(360, 765, 392, 800); arrow(360, 851, 392, 826)
arrow(602, 813, 636, 813)

# E
panel(800, 696, 656, 320, "E   Outputs, objectives & evaluation", "out")
box(820, 736, 300, 130, "Output heads (multilabel)\n\n1)  binary  soma vs neurite\n2)  5 compartments:\nCell body · Dendrite · Neuropil · Axon · Neurite", "out", 11)
whitebox(1138, 736, 300, 58, "Training:  masked BCE + pos_weight\nearly stop on validation macro-AUC", "out", 10.6)
whitebox(1138, 808, 300, 58, "Leakage-safe eval:  thresholds on val,\napplied to test · group bootstrap (split_group)", "out", 10)
whitebox(820, 886, 618, 56, "Metrics:   AUROC ↑   AUPRC ↑   F1 micro/macro ↑   Hamming ↓   subset-acc ↑    (ROC + PR dual-metric, Bonferroni)", "out", 11.5)

# inter-panel flow
arrow(360, 356, 360, 374); arrow(1100, 356, 1100, 374)
arrow(264, 660, 300, 694); arrow(728, 660, 360, 740)
arrow(1203, 660, 980, 760, dashed=True, col="#9b6a9e")
arrow(764, 813, 820, 800)

# legend
lx = 24
for k, lab in [("fm","FM view"),("eng","Handcrafted"),("oh","One-hot / task net"),("fus","Fusion"),("out","Output / training")]:
    P.append(f'<rect x="{lx}" y="1022" width="16" height="12" rx="3" fill="url(#g_{k})" stroke="{GRAD[k][2]}"/>')
    text(lx+22, 1032, lab, 11, "normal", "#333", "start"); lx += 165
text(1180, 1032, "→ main flow      ⇢ auxiliary flow", 11, "normal", "#333", "start")
P.append('</svg>')

svg = "".join(P)
os.makedirs("results/figures", exist_ok=True)
open("results/figures/fig_overview_schematic.svg", "w", encoding="utf-8").write(svg)
print("wrote results/figures/fig_overview_schematic.svg")

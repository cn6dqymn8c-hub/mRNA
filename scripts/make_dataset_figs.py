#!/usr/bin/env python3
"""Single-figure dataset overview for the neuronal mRNA localization dataset.

Outputs to results/figures/:
  fig_dataset_overview.(png|pdf)   one figure:
     top row  : (a) curation funnel  (b) compartment × species  (c) length × compartment
     bottom   : (d) compartment co-occurrence UpSet (full width)
  fig_compartment_upset.(png|pdf)  the same UpSet, standalone (large, for talks/supp).

All compartment counts are at the unique-isoform-sequence level (labels unioned
across records sharing a sequence).
"""
import csv, glob, os
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "data_训练/mixed_bulkgene_isoform_neuropil"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
LAB_ORDER = ["Cell_body", "Neuropil", "Dendrite", "Neurite", "Ribosome", "Cytoplasm", "Axon"]
SP_ORDER = ["mouse", "rat", "human"]
C_SP = {"mouse": "#3a7ca5", "rat": "#d1495b", "human": "#5c946e"}
C_LAB = "#3a7ca5"
TOPN = 20

def clean(s): return (s or "").strip().upper().replace("U", "T")

# ---- load isoform-level (unique sequence) records --------------------------
seq_sp, seq_lab, seq_len, rec_sp = {}, defaultdict(set), {}, Counter()
gene_sp = defaultdict(set)
for f in glob.glob(os.path.join(D, "*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            sp = r.get("species", "?")
            gene_sp[sp].add((r.get("gene_name", "") or "").strip())
            seq = clean(r.get("sequence", ""))
            if not seq:
                continue
            rec_sp[sp] += 1
            seq_sp[seq] = sp
            seq_len[seq] = len(seq)
            for L in str(r.get("location", "")).split(","):
                L = L.strip()
                if L:
                    seq_lab[seq].add(L)

seqs = list(seq_sp)
N = len(seqs)
seq_sp_cnt = Counter(seq_sp.values())
gene_sp_cnt = {k: len(v) for k, v in gene_sp.items()}
idx = {l: i for i, l in enumerate(LAB_ORDER)}

sp_lab = {sp: Counter() for sp in SP_ORDER}
for s in seqs:
    for L in seq_lab[s]:
        if seq_sp[s] in sp_lab:
            sp_lab[seq_sp[s]][L] += 1
set_size = {l: sum(1 for s in seqs if l in seq_lab[s]) for l in LAB_ORDER}
len_by_lab = {l: [seq_len[s] for s in seqs if l in seq_lab[s]] for l in LAB_ORDER}
combo_cnt = Counter(frozenset(l for l in seq_lab[s] if l in idx) for s in seqs)
combo_cnt.pop(frozenset(), None)
co = np.zeros((len(LAB_ORDER), len(LAB_ORDER)))
for s in seqs:
    ls = [l for l in seq_lab[s] if l in idx]
    for a in ls:
        co[idx[a], idx[a]] += 1
    for a, b in combinations(ls, 2):
        co[idx[a], idx[b]] += 1
        co[idx[b], idx[a]] += 1


# ---- panel drawers ---------------------------------------------------------
def draw_funnel(ax):
    stages = [("Records", rec_sp), ("Unique\nsequences", seq_sp_cnt), ("Unique\ngenes", gene_sp_cnt)]
    ypos = [2, 1, 0]
    for y, (name, d) in zip(ypos, stages):
        left = 0
        tot = sum(d.get(sp, 0) for sp in SP_ORDER)
        for sp in SP_ORDER:
            ax.barh(y, d.get(sp, 0), left=left, color=C_SP[sp], edgecolor="white", height=0.42)
            left += d.get(sp, 0)
        ax.text(left + max(rec_sp.values()) * 0.01, y, f"{tot:,}", va="center", fontsize=9, fontweight="bold")
    ax.set_ylim(-0.7, 2.7)
    ax.set_yticks(ypos); ax.set_yticklabels([s.replace("\n", " ") for s in ["Records", "Sequences", "Genes"]])
    ax.set_xlabel("count"); ax.set_title("a  Records → sequences → genes", loc="left", fontweight="bold")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=C_SP[sp]) for sp in SP_ORDER], SP_ORDER,
              loc="lower right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def draw_species_compartment(ax):
    y = np.arange(len(LAB_ORDER)); h = 0.26
    for k, sp in enumerate(SP_ORDER):
        ax.barh(y + (1 - k) * h, [sp_lab[sp][l] for l in LAB_ORDER], height=h, color=C_SP[sp], label=sp)
    ax.set_yticks(y); ax.set_yticklabels(LAB_ORDER); ax.invert_yaxis()
    ax.set_xlabel("unique sequences"); ax.set_title("b  Compartment × species", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)


def draw_length(ax):
    data = [np.log10(np.array(len_by_lab[l]) + 1) for l in LAB_ORDER]
    parts = ax.violinplot(data, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_LAB); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(LAB_ORDER) + 1)); ax.set_xticklabels(LAB_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("log10(sequence length)")
    ax.set_title("d  Sequence length by compartment", loc="left", fontweight="bold")
    ax.text(0.5, -0.42, "mixes full-length & 3'UTR-fragment sequences", transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="#666666", style="italic")
    ax.spines[["top", "right"]].set_visible(False)


def draw_heatmap(ax):
    disp = np.log10(co + 1)
    im = ax.imshow(disp, cmap="magma_r", aspect="auto")   # fill the cell -> left-aligns with panel a
    ax.set_xticks(range(len(LAB_ORDER))); ax.set_xticklabels(LAB_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(LAB_ORDER))); ax.set_yticklabels(LAB_ORDER, fontsize=8)
    for i in range(len(LAB_ORDER)):
        for j in range(len(LAB_ORDER)):
            ax.text(j, i, f"{int(co[i, j]):,}", ha="center", va="center", fontsize=6,
                    color="white" if disp[i, j] > disp.max() * 0.55 else "black")
    ax.set_title("c  Compartment co-occurrence (diagonal = total)", loc="left", fontweight="bold")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log10(seq+1)", fontsize=8)


def draw_upset(axbar, axmat, axset, label="d"):
    top = combo_cnt.most_common(TOPN)
    tot_seq = sum(combo_cnt.values())
    others_cnt = tot_seq - sum(c for _, c in top)
    others_n = len(combo_cnt) - len(top)
    rows = [l for l, _ in sorted(set_size.items(), key=lambda kv: -kv[1]) if set_size[l] > 0]
    yof = lambda l: len(rows) - 1 - rows.index(l)
    maxset = max(set_size.values())

    sizes = [c for _, c in top] + [others_cnt]
    x = np.arange(len(sizes))
    axbar.bar(x, sizes, color=["#3a7ca5"] * len(top) + ["#b0b0b0"], edgecolor="white")
    for xi, s in zip(x, sizes):
        axbar.text(xi, s + max(sizes) * 0.01, f"{s:,}", ha="center", va="bottom", fontsize=7)
    axbar.set_ylabel("sequences in combination")
    axbar.set_title(f"{label}  Compartment co-occurrence (UpSet) — top {len(top)} of {len(combo_cnt)} "
                    f"combinations ({sum(c for _, c in top) / tot_seq:.0%}; grey = other {others_n})",
                    loc="left", fontweight="bold", fontsize=11)
    axbar.spines[["top", "right"]].set_visible(False); axbar.set_xticks([])

    for l in rows:
        axmat.axhline(yof(l), color="#eeeeee", lw=8, zorder=0)
    for xi, (combo, c) in enumerate(top):
        axmat.scatter([xi] * len(rows), [yof(l) for l in rows], color="#d9d9d9", s=55, zorder=1)
        ys = [yof(l) for l in combo]
        if ys:
            axmat.scatter([xi] * len(ys), ys, color="#222222", s=55, zorder=2)
            if len(ys) > 1:
                axmat.plot([xi, xi], [min(ys), max(ys)], color="#222222", lw=2, zorder=2)
    axmat.scatter([len(top)] * len(rows), [yof(l) for l in rows], color="#ececec", s=55, zorder=1)
    axmat.set_yticks(range(len(rows))); axmat.set_yticklabels([])
    axmat.set_xticks([]); axmat.set_ylim(-0.6, len(rows) - 0.4)
    axmat.spines[["top", "right", "bottom", "left"]].set_visible(False)
    axmat.tick_params(left=False, labelleft=False)

    for l in rows:
        axset.barh(yof(l), set_size[l], color="#8d99ae", height=0.6)
        axset.text(set_size[l] + maxset * 0.03, yof(l), f"{set_size[l]:,}",
                   ha="right", va="center", fontsize=8, color="#333333")
    axset.invert_xaxis(); axset.set_xlim(maxset * 1.6, 0)
    axset.set_xticks([0, 10000, 20000, 30000]); axset.tick_params(axis="x", labelsize=8)
    axset.set_yticks(range(len(rows))); axset.set_yticklabels(rows[::-1], fontsize=9)
    axset.set_xlabel("set size (sequences)")
    axset.spines[["top", "right", "left"]].set_visible(False)


# ============================ overview (2×2) ================================
# a funnel | b compartment×species ; c co-occurrence heatmap | d length.
# heatmap uses aspect='auto' so it fills the cell and left-aligns with (a) above.
fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)
draw_funnel(fig.add_subplot(gs[0, 0]))
draw_species_compartment(fig.add_subplot(gs[0, 1]))
draw_heatmap(fig.add_subplot(gs[1, 0]))
draw_length(fig.add_subplot(gs[1, 1]))
fig.suptitle(f"Dataset overview — {N:,} unique isoform sequences "
             f"({sum(gene_sp_cnt.values()):,} genes; mouse/rat/human)", fontweight="bold", fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_dataset_overview.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_dataset_overview (2x2)")

# ============================ standalone UpSet ==============================
fig2 = plt.figure(figsize=(15, 7))
gs2 = fig2.add_gridspec(2, 2, width_ratios=[1.5, 4.2], height_ratios=[3, 2.2], hspace=0.06, wspace=0.05)
axbar = fig2.add_subplot(gs2[0, 1]); axmat = fig2.add_subplot(gs2[1, 1], sharex=axbar)
axset = fig2.add_subplot(gs2[1, 0], sharey=axmat)
draw_upset(axbar, axmat, axset, label="")
for ext in ("png", "pdf"):
    fig2.savefig(f"{OUT}/fig_compartment_upset.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig2)
print(f"wrote fig_compartment_upset (standalone); N={N}, genes={sum(gene_sp_cnt.values())}")

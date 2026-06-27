#!/usr/bin/env python3
"""Dataset-overview figures for the neuronal mRNA localization dataset.
Outputs to results/figures/:
  fig_dataset_overview.(png|pdf)   2x2: dedup funnel / species x compartment / co-occurrence / length
  fig_compartment_upset.(png|pdf)  hand-rolled UpSet of compartment co-occurrence
"""
import csv, glob, os
from collections import defaultdict, Counter
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

D = "data_训练/mixed_bulkgene_isoform_neuropil"
OUT = "results/figures"
os.makedirs(OUT, exist_ok=True)
LAB_ORDER = ["Cell_body", "Neuropil", "Dendrite", "Neurite", "Ribosome", "Cytoplasm", "Axon"]
SP_ORDER = ["mouse", "rat", "human"]
C_SP = {"mouse": "#3a7ca5", "rat": "#d1495b", "human": "#5c946e"}
C_LAB = "#3a7ca5"

def clean(s): return (s or "").strip().upper().replace("U", "T")

# ---- load isoform-level (unique sequence) records --------------------------
seq_sp, seq_lab, seq_len = {}, defaultdict(set), {}
rec_sp = Counter()  # record-level species
for f in glob.glob(os.path.join(D, "*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            seq = clean(r.get("sequence", ""))
            if not seq:
                continue
            sp = r.get("species", "?")
            rec_sp[sp] += 1
            seq_sp[seq] = sp
            seq_len[seq] = len(seq)
            for L in str(r.get("location", "")).split(","):
                L = L.strip()
                if L:
                    seq_lab[seq].add(L)

seqs = list(seq_sp)
N = len(seqs)
# species-level counts
seq_sp_cnt = Counter(seq_sp.values())
gene_sp = defaultdict(set)
for f in glob.glob(os.path.join(D, "*.csv")):
    with open(f) as fh:
        for r in csv.DictReader(fh):
            gene_sp[r.get("species", "?")].add((r.get("gene_name", "") or "").strip())
gene_sp_cnt = {k: len(v) for k, v in gene_sp.items()}

# species x compartment (sequence counts)
sp_lab = {sp: Counter() for sp in SP_ORDER}
for s in seqs:
    sp = seq_sp[s]
    for L in seq_lab[s]:
        if sp in sp_lab:
            sp_lab[sp][L] += 1

# co-occurrence matrix (pairwise, sequence counts)
co = np.zeros((len(LAB_ORDER), len(LAB_ORDER)))
idx = {l: i for i, l in enumerate(LAB_ORDER)}
for s in seqs:
    ls = [l for l in seq_lab[s] if l in idx]
    for a in ls:
        co[idx[a], idx[a]] += 1
    for a, b in combinations(ls, 2):
        co[idx[a], idx[b]] += 1
        co[idx[b], idx[a]] += 1

# length by compartment
len_by_lab = {l: [] for l in LAB_ORDER}
for s in seqs:
    for L in seq_lab[s]:
        if L in len_by_lab:
            len_by_lab[L].append(seq_len[s])

# combos for UpSet
combo_cnt = Counter(frozenset(l for l in seq_lab[s] if l in idx) for s in seqs)
combo_cnt.pop(frozenset(), None)

# ============================ FIGURE 1: overview ============================
fig = plt.figure(figsize=(13, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

# (a) dedup funnel
axa = fig.add_subplot(gs[0, 0])
stages = [("Records", rec_sp), ("Unique\nsequences", seq_sp_cnt), ("Unique\ngenes", gene_sp_cnt)]
ypos = [2, 1, 0]
for y, (name, d) in zip(ypos, stages):
    left = 0
    tot = sum(d.get(sp, 0) for sp in SP_ORDER)
    for sp in SP_ORDER:
        v = d.get(sp, 0)
        axa.barh(y, v, left=left, color=C_SP[sp], edgecolor="white", height=0.62)
        left += v
    axa.text(left + max(rec_sp.values()) * 0.01, y, f"{tot:,}", va="center", fontsize=10, fontweight="bold")
axa.set_yticks(ypos); axa.set_yticklabels([s[0] for s in stages])
axa.set_xlabel("count"); axa.set_title("a  Curation funnel (stacked by species)", loc="left", fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=C_SP[sp]) for sp in SP_ORDER]
axa.legend(handles, SP_ORDER, loc="lower right", frameon=False, fontsize=9)
axa.spines[["top", "right"]].set_visible(False)

# (b) species x compartment grouped bars
axb = fig.add_subplot(gs[0, 1])
y = np.arange(len(LAB_ORDER)); h = 0.26
for k, sp in enumerate(SP_ORDER):
    vals = [sp_lab[sp][l] for l in LAB_ORDER]
    axb.barh(y + (1 - k) * h, vals, height=h, color=C_SP[sp], label=sp)
axb.set_yticks(y); axb.set_yticklabels(LAB_ORDER)
axb.invert_yaxis()
axb.set_xlabel("unique sequences"); axb.set_title("b  Compartment × species", loc="left", fontweight="bold")
axb.legend(frameon=False, fontsize=9); axb.spines[["top", "right"]].set_visible(False)

# (c) co-occurrence heatmap
axc = fig.add_subplot(gs[1, 0])
M = co.copy()
disp = np.log10(M + 1)
im = axc.imshow(disp, cmap="magma_r")
axc.set_xticks(range(len(LAB_ORDER))); axc.set_xticklabels(LAB_ORDER, rotation=45, ha="right", fontsize=8)
axc.set_yticks(range(len(LAB_ORDER))); axc.set_yticklabels(LAB_ORDER, fontsize=8)
for i in range(len(LAB_ORDER)):
    for j in range(len(LAB_ORDER)):
        axc.text(j, i, f"{int(M[i, j]):,}", ha="center", va="center", fontsize=6,
                 color="white" if disp[i, j] > disp.max() * 0.55 else "black")
axc.set_title("c  Compartment co-occurrence (diagonal = total)", loc="left", fontweight="bold")
cb = fig.colorbar(im, ax=axc, fraction=0.046, pad=0.04); cb.set_label("log10(seq+1)", fontsize=8)

# (d) length by compartment (log)
axd = fig.add_subplot(gs[1, 1])
data = [np.log10(np.array(len_by_lab[l]) + 1) for l in LAB_ORDER]
parts = axd.violinplot(data, showmedians=True)
for pc in parts["bodies"]:
    pc.set_facecolor(C_LAB); pc.set_alpha(0.6)
axd.set_xticks(range(1, len(LAB_ORDER) + 1)); axd.set_xticklabels(LAB_ORDER, rotation=45, ha="right", fontsize=8)
axd.set_ylabel("log10(sequence length)")
axd.set_title("d  Sequence length by compartment", loc="left", fontweight="bold")
axd.spines[["top", "right"]].set_visible(False)

fig.suptitle(f"Dataset overview — {N:,} unique isoform sequences "
             f"({sum(gene_sp_cnt.values()):,} genes; mouse/rat/human)", fontweight="bold")
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/fig_dataset_overview.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote fig_dataset_overview")

# ============================ FIGURE 2: UpSet (complete 3-part) =============
TOPN = 20
top = combo_cnt.most_common(TOPN)
n_combo_total = len(combo_cnt)
tot_seq = sum(combo_cnt.values())
others_cnt = tot_seq - sum(c for _, c in top)
others_n = n_combo_total - len(top)
# set size = true total sequences per compartment (co-occurrence diagonal)
set_size = {l: int(co[idx[l], idx[l]]) for l in LAB_ORDER}
rows = [l for l, _ in sorted(set_size.items(), key=lambda kv: -kv[1]) if set_size[l] > 0]
rr = {l: i for i, l in enumerate(rows)}          # rr=0 -> largest set
yof = lambda l: len(rows) - 1 - rr[l]            # largest set on top

fig2 = plt.figure(figsize=(13, 6.8))
gs2 = GridSpec(2, 2, width_ratios=[1.5, 4.2], height_ratios=[3, 2.2],
               hspace=0.06, wspace=0.05, figure=fig2)
axbar = fig2.add_subplot(gs2[0, 1])              # top-right: intersection sizes
axmat = fig2.add_subplot(gs2[1, 1], sharex=axbar)  # bottom-right: dot matrix
axset = fig2.add_subplot(gs2[1, 0], sharey=axmat)  # bottom-left: set sizes

# --- intersection-size bars (top-N + aggregated 'others') ---
sizes = [c for _, c in top] + [others_cnt]
x = np.arange(len(sizes))
colors = ["#3a7ca5"] * len(top) + ["#b0b0b0"]
axbar.bar(x, sizes, color=colors, edgecolor="white")
for xi, s in zip(x, sizes):
    axbar.text(xi, s + max(sizes) * 0.01, f"{s:,}", ha="center", va="bottom", fontsize=7)
axbar.set_ylabel("sequences in combination")
axbar.set_title(f"Compartment co-occurrence (UpSet) — top {len(top)} of {n_combo_total} combinations "
                f"({sum(c for _, c in top) / tot_seq:.0%} of sequences; grey = other {others_n} combos)",
                loc="left", fontweight="bold", fontsize=11)
axbar.spines[["top", "right"]].set_visible(False); axbar.set_xticks([])

# --- dot matrix ---
for l in rows:
    axmat.axhline(yof(l), color="#eeeeee", lw=8, zorder=0)
for xi, (combo, c) in enumerate(top):
    axmat.scatter([xi] * len(rows), [yof(l) for l in rows], color="#d9d9d9", s=62, zorder=1)
    ys = [yof(l) for l in combo]
    if ys:
        axmat.scatter([xi] * len(ys), ys, color="#222222", s=62, zorder=2)
        if len(ys) > 1:
            axmat.plot([xi, xi], [min(ys), max(ys)], color="#222222", lw=2, zorder=2)
# 'others' column: all-grey (no specific pattern)
ox = len(top)
axmat.scatter([ox] * len(rows), [yof(l) for l in rows], color="#ececec", s=62, zorder=1)
axmat.set_yticks(range(len(rows))); axmat.set_yticklabels([])
axmat.set_xticks([]); axmat.set_ylim(-0.6, len(rows) - 0.4)
axmat.spines[["top", "right", "bottom", "left"]].set_visible(False)
axmat.tick_params(left=False, labelleft=False)

# --- set-size bars (left, growing leftwards) with count at each bar tip ---
maxset = max(set_size.values())
for l in rows:
    axset.barh(yof(l), set_size[l], color="#8d99ae", height=0.6)
    axset.text(set_size[l] + maxset * 0.03, yof(l), f"{set_size[l]:,}",
               ha="right", va="center", fontsize=8, color="#333333")
axset.invert_xaxis()
axset.set_xlim(maxset * 1.6, 0)
axset.set_yticks(range(len(rows))); axset.set_yticklabels(rows[::-1], fontsize=9)
axset.set_xlabel("set size (sequences)")
axset.spines[["top", "right", "left"]].set_visible(False)
for ext in ("png", "pdf"):
    fig2.savefig(f"{OUT}/fig_compartment_upset.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig2)
print("wrote fig_compartment_upset")
print(f"N_seq={N}, genes={sum(gene_sp_cnt.values())}, top combo={top[0]}")

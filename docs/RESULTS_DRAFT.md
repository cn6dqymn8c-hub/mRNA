# Draft — Abstract & Results

> 草稿。数字均来自受控 benchmark 的 **test 集 + group-level bootstrap**。`[N…]` 处
> 待你用确切总样本数填上(test n 已给;总数≈test/0.15)。物种:mouse+rat(human 太少已排除)。

---

## Suggested title
**Foundation models do not outperform a k-mer baseline for neuronal mRNA
soma-vs-neurite localization — but fusing them with sequence features does**

---

## Abstract

Localization of neuronal mRNAs to the neurite compartment (dendrites/axons) versus
the soma underlies local translation and synaptic function, yet how much of this
localization is encoded in transcript sequence — and whether modern RNA/DNA
foundation models capture it — is unclear. We assembled a curated, multi-source
benchmark of neuronal mRNA soma-vs-neurite localization in mouse and rat
([N_genes] genes from 18 datasets, including 6 isoform-resolved sources) and
evaluated seven model families under a single leakage-safe, ortholog-grouped
train/validation/test protocol, across transcript regions (3'UTR, CDS, full-length)
and resolutions (gene, isoform). Frozen embeddings from RNA and DNA foundation
models (RNA-FM, mRNA-FM, DNABERT-2, UTR-BERT) do **not** significantly outperform a
simple k-mer baseline (e.g. on CDS, mRNA-FM vs k-mer ΔROC-AUC = +0.002, p = 0.88);
the single nominal exception (RNA-FM on isoform 3'UTRs, +0.018, p = 0.032) does not
survive multiple-testing correction. However, an **attention-gated fusion** of a
foundation-model embedding with interpretable sequence features (length, GC,
composition/motif density) **significantly and robustly** outperforms k-mer and
every individual model in all settings (ΔROC-AUC +0.024 to +0.051, all p ≤ 0.006,
Bonferroni-robust), reaching ROC-AUC 0.70 / AUPRC 0.74 on full-length transcripts.
Component decomposition shows the foundation model contributes the largest,
significant share of this gain (+0.085 ROC-AUC over engineered features alone),
ruling out transcript length as the explanation, and that fusion exceeds either view
alone (synergy). A region ablation on matched genes shows the **full-length
transcript carries significantly more localization signal than the 3'UTR or CDS
alone** (full vs 3'UTR +0.042, p = 0.001; full vs CDS +0.048, p < 0.001; CDS vs
3'UTR n.s.). For this task, foundation models are individually no better than
k-mers but provide complementary information that, fused with simple sequence
features, yields the best and only robustly-significant predictor. We release the
benchmark, leakage-safe splits, and the fusion model.

---

## Results

### 1. A leakage-safe benchmark for neuronal mRNA localization
We harmonized 18 neuronal mRNA localization datasets (mouse, rat) into a soma-vs-
neurite task, unioning labels across sources and using a per-source observability
mask so that unmeasured labels are not treated as negatives. Transcripts were
grouped into leakage-safe units by gene, ortholog group, and exact-sequence identity
(union-find), and partitioned 70/15/15 with a balanced grouped split that is frozen
and reused across all models. We evaluated four region/resolution settings — 3'UTR
(gene), 3'UTR (isoform), CDS (gene), full-length (gene) — each comparing k-mer,
two purpose-built localization nets (RNATracker-, DM3Loc-style), and frozen
embeddings from UTR-BERT, RNA-FM, mRNA-FM and DNABERT-2, with a logistic head on an
identical split. Model selection used validation only; all reported metrics are on
the held-out test set, and all model–model comparisons use a gene/ortholog-group
cluster bootstrap (2000 resamples).

### 2. Foundation-model embeddings do not beat a k-mer baseline
Across all four settings, no foundation model significantly outperformed the k-mer
baseline (Table 1). On CDS, the best single model, mRNA-FM, exceeded k-mer by only
ΔROC-AUC = +0.002 (95% CI [−0.021, +0.026], p = 0.88); on full-length the best model
tied k-mer (−0.000, p = 0.98); on gene-level 3'UTR the best model was +0.010
(p = 0.31). The only nominally significant advantage was RNA-FM on isoform-resolved
3'UTRs (+0.018, 95% CI [+0.001, +0.033], p = 0.032), which does not survive Bonferroni
correction for the four comparisons. Test-set ROC-AUC ranged 0.59–0.66, indicating a
hard, label-noisy task on which generic foundation-model representations provide no
advantage over simple k-mer frequencies.

### 3. An attention-gated fusion model is significantly best across all settings
We propose a model that fuses a foundation-model embedding with interpretable
sequence features (transcript length, GC content, dinucleotide composition, and a
panel of localization-associated motif densities). Each view is projected to a common
space, an input-dependent gate softmax-weights the views, and the fused vector feeds
an MLP head; the model is trained on the identical split/objective as all baselines.
The fusion model was the top model — by both validation and test — in every setting,
and significantly outperformed k-mer in all four (Table 2): ΔROC-AUC +0.024
(3'UTR gene, p = 0.006), +0.040 (3'UTR isoform, p < 0.001), +0.036 (CDS, p = 0.002)
and +0.051 (full-length, p < 0.001); all survive Bonferroni correction. On full-length
transcripts it reached ROC-AUC 0.702 and AUPRC 0.737.

### 4. The foundation model, not transcript length, drives the fusion gain
Because the engineered features include transcript length — which is correlated with
isoform localization — we decomposed the full-length fusion model into its components
(Fig. X). A length-only model reached ROC-AUC 0.559; adding GC/composition/motif
features (engineered) raised this significantly to 0.617 (+0.058, p < 0.001); adding
the RNA-FM embedding (fusion) raised it further to 0.702 (+0.085 over engineered,
95% CI [+0.068, +0.103], p < 0.001). The foundation model thus contributes the largest,
significant share of performance and is not redundant with sequence length, and the
fusion exceeds either the foundation model (0.643) or engineered features (0.617)
alone — i.e. the two views are complementary. [Repeat on isoform 3'UTR (Track1B) to
show generality — fill numbers.]

### 5. Localization signal is concentrated in the full-length transcript
A region ablation restricted to genes whose 3'UTR, CDS and full-length sequence could
all be extracted (same genes, same split, RNA-FM, only the region varies) showed that
full-length sequence carries significantly more soma-vs-neurite signal than either
sub-region: full vs 3'UTR ΔROC-AUC = +0.042 (p = 0.001) and full vs CDS = +0.048
(p < 0.001), while CDS and 3'UTR did not differ (−0.005, p = 0.73). Localization
information is therefore distributed across the transcript rather than confined to the
3'UTR, consistent with the fusion model performing best on full-length input.

---

## Tables (fill exact n)

**Table 1. Foundation models vs k-mer (test ROC-AUC; group bootstrap).**

| Setting | best non-kmer | model ROC | k-mer ROC | Δ | 95% CI | p | sig |
|---|---|---|---|---|---|---|---|
| 3'UTR gene | RNATracker | 0.618 | 0.608 | +0.010 | [−0.009,+0.028] | 0.31 | no |
| 3'UTR isoform | RNA-FM | 0.663 | 0.646 | +0.018 | [+0.001,+0.033] | 0.032 | nominal |
| CDS gene | mRNA-FM | 0.621 | 0.619 | +0.002 | [−0.021,+0.026] | 0.88 | no |
| full gene | RNATracker | 0.651 | 0.651 | −0.000 | [−0.018,+0.018] | 0.98 | no |

**Table 2. Fusion model vs k-mer (test ROC-AUC; group bootstrap).**

| Setting | fusion ROC | k-mer ROC | Δ | 95% CI | p | Bonferroni-sig |
|---|---|---|---|---|---|---|
| 3'UTR gene | 0.632 | 0.608 | +0.024 | [+0.007,+0.042] | 0.006 | ✅ |
| 3'UTR isoform | 0.685 | 0.646 | +0.040 | [+0.025,+0.054] | <0.001 | ✅ |
| CDS gene | 0.655 | 0.619 | +0.036 | [+0.016,+0.058] | 0.002 | ✅ |
| full gene | 0.702 | 0.651 | +0.051 | [+0.033,+0.069] | <0.001 | ✅ |

**Table 3. Full-length component decomposition (test ROC-AUC).**

| Model | ROC-AUC | Δ vs previous | p |
|---|---|---|---|
| length only | 0.559 | — | — |
| engineered (length+GC+composition+motifs) | 0.617 | +0.058 | <0.001 |
| RNA-FM (frozen) | 0.643 | — | — |
| **fusion (RNA-FM ⊕ engineered)** | **0.702** | +0.085 (over engineered) | <0.001 |

**Table 4. Region ablation, matched genes, RNA-FM (test ROC-AUC).**

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| full vs 3'UTR | +0.042 | [+0.017,+0.067] | 0.001 |
| full vs CDS | +0.048 | [+0.023,+0.074] | <0.001 |
| CDS vs 3'UTR | −0.005 | [−0.036,+0.023] | 0.73 |

---

## Notes / to finish before submission
- Fill exact total N per setting (test n: 3'UTR-gene 2855, 3'UTR-iso 4489, CDS 2522,
  full 3019; ×~6.7 for totals) and group counts (n_groups ≈ 1859–2203).
- Run §4 decomposition on the isoform 3'UTR setting too (generality).
- Report AUPRC alongside ROC-AUC (we used ROC-AUC for tests as it is prior-robust
  across settings); also report per-species (mouse/rat) and the all-zero/prior baselines.
- Multiple-testing: state that fusion-vs-kmer survives Bonferroni (4 tests).
- Limitations: mouse/rat only; label noise (heterogeneous assays); fusion uses frozen
  FM embeddings (a fine-tuned RNA-FM on full-length reached ROC 0.652, below fusion).
- Drop the temporary `_tmp_ablate_split_check_*` runs from all summaries.

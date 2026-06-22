# Draft — Abstract & Results (nature-writing skill-structured)

> Drafted with the `nature-writing` evidence-ladder method.
> **Detected axes** — paper_type: algorithmic (proposes a model) · sections: abstract,
> experiments(results) · language: zh-to-en · journal: nat-comms.
> **One-sentence argument**: *Generic RNA/DNA foundation models do not individually
> beat a k-mer baseline for neuronal mRNA soma-vs-neurite localization, but an
> attention-gated fusion of a foundation-model embedding with interpretable sequence
> features is significantly and robustly the best predictor, with the foundation
> model — not transcript length — driving the gain.*
> Numbers are test-set, group-level bootstrap; `[N…]` = fill exact totals. Mouse+rat
> (human excluded, too few). Results report findings only (no interpretation); see
> Discussion in `PAPER_DRAFT_zh.md`.

---

## Suggested title
**Foundation models do not outperform a k-mer baseline for neuronal mRNA
soma-vs-neurite localization — but fusing them with sequence features does**

---

## Abstract

*(background)* Localization of neuronal mRNAs to the neurite compartment
(dendrites/axons) versus the soma underlies local translation and synaptic function,
and is thought to be largely encoded in transcript sequence. *(gap)* Whether modern
RNA/DNA foundation models capture this signal, and how they compare to simple
baselines under leakage-safe evaluation, has not been established. *(approach)* We
assembled a curated, multi-source benchmark of soma-vs-neurite localization in mouse
and rat ([N_genes] genes, 18 datasets including 6 isoform-resolved sources) and
evaluated seven model families under one leakage-safe, ortholog-grouped
train/validation/test protocol across transcript regions (3'UTR, CDS, full-length)
and resolutions (gene, isoform); all comparisons use a gene/ortholog-group cluster
bootstrap. *(key results)* Frozen embeddings from RNA and DNA foundation models
(RNA-FM, mRNA-FM, DNABERT-2, UTR-BERT) did not significantly outperform a k-mer
baseline (best case, mRNA-FM on CDS, ΔROC-AUC = +0.002, p = 0.88; the single nominal
exception did not survive multiple-testing correction). An attention-gated fusion of
a foundation-model embedding with interpretable sequence features (length, GC,
composition, motif density) was the best model in every setting and significantly
exceeded k-mer in all four (ΔROC-AUC +0.024 to +0.051, all p ≤ 0.006,
Bonferroni-robust), reaching ROC-AUC 0.70 / AUPRC 0.74 on full-length transcripts.
Component decomposition showed the foundation model contributed the largest,
length-independent share of the gain (+0.085 ROC-AUC over engineered features on
full-length; +0.029 on isoform 3'UTRs; both p < 0.001), and that fusion exceeded
either view alone. A matched-gene region ablation showed full-length sequence carried
significantly more signal than the 3'UTR or CDS (full vs 3'UTR +0.042, p = 0.001).
*(significance)* For this task, foundation models are individually no better than
k-mers but carry complementary information that, fused with simple features, yields
the best and only robustly-significant predictor. *(boundary)* We establish this in
rodent neuronal localization with frozen embeddings; we release the benchmark,
leakage-safe splits, and model.

---

## Results (evidence ladder)

### 1 — System validation: a leakage-safe benchmark with a frozen, shared protocol
We harmonized 18 neuronal mRNA localization datasets (mouse, rat) into a
soma-vs-neurite task, unioning labels across sources under a per-source observability
mask. Transcripts were merged into leakage-safe units by gene, ortholog group and
exact-sequence identity (union-find) and partitioned 70/15/15 by a balanced grouped
split that was frozen and reused by every model. Four region/resolution settings
(3'UTR-gene, 3'UTR-isoform, CDS-gene, full-length-gene; test n = 2,855 / 4,489 /
2,522 / 3,019; [N…] groups) were evaluated under an identical split, head and
objective; model selection used validation only and all model–model contrasts used a
gene/ortholog-group cluster bootstrap (2,000 resamples).

### 2 — Baseline comparison: foundation-model embeddings did not beat k-mer
Across all four settings, no foundation model significantly exceeded the k-mer
baseline (Table 1). The best single model on CDS, mRNA-FM, exceeded k-mer by
ΔROC-AUC = +0.002 (95% CI [−0.021, +0.026], p = 0.88); on full-length the best model
tied k-mer (−0.000, p = 0.98); on 3'UTR-gene the best was +0.010 (p = 0.31). The only
nominally significant advantage, RNA-FM on isoform 3'UTRs (+0.018, 95% CI
[+0.001, +0.033], p = 0.032), did not survive Bonferroni correction over the four
comparisons. Test ROC-AUC spanned 0.59–0.66.

### 3 — Main result: an attention-gated fusion model was significantly best in all settings
A model fusing a foundation-model embedding with interpretable features (length, GC,
dinucleotide composition, localization-motif density) via an input-dependent softmax
gate, trained on the identical split and objective, was the top model by both
validation and test in every setting and significantly exceeded k-mer in all four
(Table 2): ΔROC-AUC +0.024 (3'UTR-gene, p = 0.006), +0.040 (3'UTR-isoform,
p < 0.001), +0.036 (CDS, p = 0.002) and +0.051 (full-length, p < 0.001); all survived
Bonferroni correction. On full-length transcripts it reached ROC-AUC 0.702 and
AUPRC 0.737.

### 4 — Mechanism: the foundation model, not transcript length, drove the gain
Decomposing the full-length fusion model into nested components (Table 3), a
length-only model reached ROC-AUC 0.559; adding GC/composition/motif features raised
this to 0.617 (+0.058, p < 0.001); adding the RNA-FM embedding raised it to 0.702
(+0.085 over engineered features, 95% CI [+0.068, +0.103], p < 0.001). The fusion
exceeded the foundation model (0.643) and the engineered features (0.617) alone. The
same decomposition on isoform 3'UTRs reproduced the ordering (Table 3b: length 0.560
→ engineered 0.656, +0.096 → fusion 0.685, +0.029 over engineered; both p < 0.001).
The foundation model thus contributed a significant increment that was not captured
by transcript length, in both settings.

### 5 — Generalization: localization signal was strongest in the full-length transcript
In a region ablation restricted to genes whose 3'UTR, CDS and full-length sequence
could all be extracted (same genes, same split, RNA-FM, region the only variable;
Table 4), full-length sequence exceeded both sub-regions: full vs 3'UTR
ΔROC-AUC = +0.042 (p = 0.001) and full vs CDS = +0.048 (p < 0.001), while CDS and
3'UTR did not differ (−0.005, p = 0.73).

### 6 — Boundaries
The task was hard and label-noisy (best ROC-AUC ≈ 0.70); evaluation was rodent-only
(human n too small); the fusion used frozen embeddings (a fine-tuned RNA-FM on
full-length reached ROC-AUC 0.652, below fusion); isoform-discordant pairs were
dominated by one curated source whose soma/neurite split is ~90% length-associated,
warranting length-residualized analysis.

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

**Table 3b. Isoform 3'UTR component decomposition (test ROC-AUC).**

| Model | ROC-AUC | Δ vs previous | p |
|---|---|---|---|
| length only | 0.560 | — | — |
| engineered | 0.656 | +0.096 | <0.001 |
| RNA-FM (frozen) | 0.663 | — | — |
| **fusion (RNA-FM ⊕ engineered)** | **0.685** | +0.029 (over engineered) | <0.001 |

**Table 4. Region ablation, matched genes, RNA-FM (test ROC-AUC).**

| comparison | Δ | 95% CI | p |
|---|---|---|---|
| full vs 3'UTR | +0.042 | [+0.017,+0.067] | 0.001 |
| full vs CDS | +0.048 | [+0.023,+0.074] | <0.001 |
| CDS vs 3'UTR | −0.005 | [−0.036,+0.023] | 0.73 |

---

## Adversarial self-review (rejection-risk audit)

| Reviewer objection | Status / how addressed |
|---|---|
| "The gain is just transcript length." | **Closed.** Length-only ROC 0.56; fusion adds +0.085 / +0.029 over all engineered features (incl. length), p<0.001, in two settings. |
| "You only froze the FMs." | **Partly.** We add a fine-tuned RNA-FM (ROC 0.652, < fusion); state that larger-scale fine-tuning is future work. |
| "Differences are within noise." | **Closed.** Group cluster bootstrap; all fusion-vs-kmer survive Bonferroni (4 tests). |
| "Cross-setting comparisons are unfair." | **Closed.** Contrasts are within-setting; the region claim uses a matched-gene ablation (same genes/split). |
| "Single proposed model, weak novelty." | Frame as benchmark(resource) + a model that is the only robustly-significant predictor + the length-independent FM-contribution finding + the full-length biological result. |
| "Only one species family / curated isoform source." | Stated as boundary; mouse+rat; isoform discordance length-residualized analysis flagged. |
| "Is the metric cherry-picked?" | ROC-AUC (prior-robust across settings) for tests; AUPRC reported alongside; prior/all-zero baselines included. |

## To finish before submission
- Fill exact total N and group counts per setting; report AUPRC + per-species (mouse/rat).
- Decomposition done on full-length AND isoform 3'UTR (both confirm FM > engineered > length).
- Train the final deployment model (full-length fusion, `--train-on-all`) + `predict.py`.
- Drop temporary `_tmp_ablate_split_check_*` runs from all summaries.
- Optional: fine multi-label (5-compartment) as a secondary axis; isoform-discrimination main figure with length residualization.

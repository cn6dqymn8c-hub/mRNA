# Middleton / GSE115480 Data

Date prepared: 2026-06-01

This directory contains normalized local copies of the official Middleton/GSE115480 soma-dendrite count data.

## Naming Convention

Use these names consistently:

| Short name | Meaning | File |
|---|---|---|
| `M1_gene_3utr_counts` | Gene-level 3'UTR-derived count table. Counts are from 3'UTR regions, then summarized per gene. | `GSE115480_combined_3utr_counts_per_gene.clean.tsv.gz` |
| `M2_utr_feature_counts` | Individual 3'UTR feature-level count table. Different 3'UTR features from the same gene remain separate. | `GSE115480_individual_3utr_counts.clean.tsv.gz` |
| `M3_raw_sample_counts` | Per-sample raw exon count files from the GEO RAW archive. | `data/raw/external/GSE115480_RAW/` |

In short:

- `M1_gene_3utr_counts` = one row per gene-like feature.
- `M2_utr_feature_counts` = one row per individual 3'UTR feature, for example `Gene_utr1`.
- `M3_raw_sample_counts` = one count file per sample.

## Source

- GEO accession: GSE115480
- Species: mouse (`Mus musculus`)
- Design: paired soma and dendrite RNA-seq from 16 individual primary mouse hippocampal neurons, total 32 samples.
- Compartments:
  - sample IDs ending with `D`: dendrite
  - sample IDs ending with `S`: soma

## Downloaded Raw Files

Raw downloaded files are stored in `data/raw/external/`:

- `GSE115480_combined_3utr_counts_per_gene.txt.gz`
- `GSE115480_individual_3utr_counts.txt.gz`
- `GSE115480_RAW.tar`
- `GSE115480_RAW/`: extracted per-sample exon count files, 32 files

## Clean Files

- `M1_gene_3utr_counts`
  - file: `GSE115480_combined_3utr_counts_per_gene.clean.tsv.gz`
  - 44,081 gene-level rows
  - columns: `feature_id` + 32 sample count columns
  - `feature_id` is mouse gene symbol / gene-like identifier from the GEO table
  - interpretation: 3'UTR-region reads summarized to gene level

- `M2_utr_feature_counts`
  - file: `GSE115480_individual_3utr_counts.clean.tsv.gz`
  - 74,532 3'UTR-level rows
  - columns: `feature_id` + 32 sample count columns
  - `feature_id` has form like `Gene_utr1`
  - interpretation: individual 3'UTR features are kept separate

- `GSE115480_sample_metadata.tsv`
  - 32 samples
  - 16 dendrite samples and 16 paired soma samples
  - columns: `sample_id`, `cell_id`, `compartment`

## Intended Use

This is not six-label gold data. It should be used as partial mouse soma/dendrite evidence:

- dendrite-enriched genes/UTRs -> weak/partial `Dendrite` positive
- soma-enriched genes/UTRs -> weak/partial `Cell_body` or Soma positive
- non-significant genes/UTRs -> unknown, not negatives
- other localization labels should be masked

For the next model-ready label table, start from `M1_gene_3utr_counts` unless we specifically want isoform/UTR-level analysis from `M2_utr_feature_counts`.

The next processing step should compute paired soma-vs-dendrite enrichment, for example `log2FC_dendrite_vs_soma` plus a statistical/filtering criterion, before creating model-ready weak labels.

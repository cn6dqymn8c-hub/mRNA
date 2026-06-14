#!/usr/bin/env bash
set -euo pipefail
cd /home/hqzhu/mRNA1
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} /home/hqzhu/miniconda3/envs/iloc/bin/python -u scripts/train_dnbert.py \
  --input-dir data_训练/gene_level_full_length_curated/transcript_sequence_inputs \
  --model-path DNABERT-2-117M \
  --outdir results/gene_level_full_length_curated_dnabert \
  --sample-level source_gene \
  --split-group ortholog_group \
  --ortholog-map ortholog/human_mouse_rat_gene_to_ortholog_group.tsv \
  --source-aware \
  --species-aware \
  --loss-mask source_labels \
  --epochs 3 \
  --batch-size 8 \
  --fp16

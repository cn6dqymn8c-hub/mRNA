#!/usr/bin/env bash
# ===========================================================================
# Multi-seed re-training for the SEED-DEPENDENT models, to report seed-robust
# mean ± sd instead of a single seed=0 draw.
#
#   seed-dependent : fusion, RNATracker, DM3Loc, RNA-FM fine-tune  (torch random
#                    init + batch shuffling)
#   seed-INDEPENDENT (NOT re-run here): k-mer, FM frozen + logistic head,
#                    engineered, length  -> deterministic, one run suffices.
#
# The frozen split is reused via --split-assignments, so --seed only varies
# TRAINING randomness (not the data partition). Output dirs are suffixed _s<seed>.
# seed 0 is assumed already done; this runs the extra seeds.
#
# Usage:
#   bash scripts/run_seeds.sh fusion     # fusion across all settings (cheap; cached embeds)
#   bash scripts/run_seeds.sh nets       # RNATracker + DM3Loc
#   bash scripts/run_seeds.sh finetune   # RNA-FM end-to-end fine-tune (EXPENSIVE)
#   bash scripts/run_seeds.sh all
#   SEEDS="1 2 3 4" bash scripts/run_seeds.sh fusion   # override seed list
# After running, aggregate:  python scripts/aggregate_seeds.py --glob "results/<setting>/fusion_rnafm_eng*"
# ===========================================================================
set -euo pipefail

PY=python
TRAIN=scripts/train_rnafm_multilabel.py
IN="data_训练/mixed_bulkgene_isoform_neuropil"
ORTHO="ortholog/human_mouse_rat_gene_to_ortholog_group.tsv"
GTF=(data_训练/Homo_sapiens.GRCh38.116.gtf.gz data_训练/Mus_musculus.GRCm39.116.gtf.gz data_训练/Rattus_norvegicus.GRCr8.116.gtf.gz)
NATIVE=(isoform)   # 命中 AllIsoforms_coordinates(Andreassi);其余原生源由 seqtype 自动识别
SP=results/_frozen_splits
M_RNAFM=./rnafm
M_MRNAFM=./mrnafm
SEEDS="${SEEDS:-1 2 3 4}"   # seed 0 already exists

# full-species common args (NO --species), seed is appended per run
C_BIN=(--label-scheme soma_vs_neurite --label-agg soft --source-mask --classifier logistic
       --min-support 150 --ortholog-map "$ORTHO")
C_FINE=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
        --min-support 200 --ortholog-map "$ORTHO")

# Emit the per-setting base args (region/level/gtf/native/split) on stdout.
base_args() {
  case "$1" in
    track1a_gene)        echo "--region utr3 --sample-level gene                 --gtf ${GTF[*]} --native-region-sources ${NATIVE[*]} --split-assignments $SP/split_U1_gene.csv" ;;
    track1b_isoform)     echo "--region utr3 --sample-level isoform_sequence_union --gtf ${GTF[*]} --native-region-sources ${NATIVE[*]} --split-assignments $SP/split_U1_gene.csv" ;;
    track2_gene)         echo "--region cds  --sample-level gene                 --gtf ${GTF[*]} --split-assignments $SP/split_U2_gene.csv" ;;
    track3_full)         echo "--region full --sample-level gene                 --split-assignments $SP/split_U3_gene.csv" ;;
    track3_full_isoform) echo "--region full --sample-level isoform_sequence_union --split-assignments $SP/split_U3_gene.csv" ;;
    fine_full_gene)      echo "--region full --sample-level gene                 --split-assignments $SP/split_fine_full_gene.csv" ;;
    fine_full_isoform)   echo "--region full --sample-level isoform_sequence_union --split-assignments $SP/split_fine_full_isoform.csv" ;;
    *) echo "UNKNOWN_SETTING $1" >&2; exit 1 ;;
  esac
}

BIN_SETTINGS=(track1a_gene track1b_isoform track2_gene track3_full track3_full_isoform)
FINE_SETTINGS=(fine_full_gene fine_full_isoform)

# FM for fusion: CDS uses codon mRNA-FM, everything else RNA-FM.
fm_for() { [ "$1" = "track2_gene" ] && echo "$M_MRNAFM" || echo "$M_RNAFM"; }
fus_tag() { [ "$1" = "track2_gene" ] && echo "fusion_mrnafm_eng" || echo "fusion_rnafm_eng"; }

run_fusion() {
  for setting in "${BIN_SETTINGS[@]}"; do C=("${C_BIN[@]}");  _fusion_setting "$setting"; done
  for setting in "${FINE_SETTINGS[@]}"; do C=("${C_FINE[@]}"); _fusion_setting "$setting"; done
}
_fusion_setting() {
  local setting="$1" fm tag; fm=$(fm_for "$setting"); tag=$(fus_tag "$setting")
  read -r -a B <<< "$(base_args "$setting")"
  for s in $SEEDS; do
    echo "### [fusion] $setting seed=$s"
    $PY $TRAIN --input-dir "$IN" "${B[@]}" "${C[@]}" --seed "$s" \
      --fusion --features fm engineered --model-dir "$fm" --max-tokens 1022 \
      --output-dir "results/$setting/${tag}_s$s"
  done
}

run_nets() {
  for setting in "${BIN_SETTINGS[@]}"; do C=("${C_BIN[@]}");  _nets_setting "$setting"; done
  for setting in "${FINE_SETTINGS[@]}"; do C=("${C_FINE[@]}"); _nets_setting "$setting"; done
}
_nets_setting() {
  local setting="$1"; read -r -a B <<< "$(base_args "$setting")"
  for s in $SEEDS; do
    for arch in rnatracker dm3loc; do
      echo "### [$arch] $setting seed=$s"
      $PY $TRAIN --input-dir "$IN" "${B[@]}" "${C[@]}" --seed "$s" \
        --arch "$arch" --ts-max-len 31000 \
        --output-dir "results/$setting/${arch}_s$s"
    done
  done
}

# RNA-FM end-to-end fine-tune. EXPENSIVE -> full-length settings only (binary +
# fine). Extend the loop if you want it on 3'UTR/CDS too.
run_finetune() {
  local FT_SETTINGS=(track3_full track3_full_isoform fine_full_gene fine_full_isoform)
  for setting in "${FT_SETTINGS[@]}"; do
    case "$setting" in fine_*) C=("${C_FINE[@]}");; *) C=("${C_BIN[@]}");; esac
    read -r -a B <<< "$(base_args "$setting")"
    for s in $SEEDS; do
      echo "### [finetune] $setting seed=$s"
      $PY $TRAIN --input-dir "$IN" "${B[@]}" "${C[@]}" --seed "$s" \
        --finetune --model-dir "$M_RNAFM" --ft-long-seq-policy sliding_mean \
        --output-dir "results/$setting/rnafm_finetune_sliding_s$s"
    done
  done
}

case "${1:-all}" in
  fusion)   run_fusion ;;
  nets)     run_nets ;;
  finetune) run_finetune ;;
  all)      run_fusion; run_nets; run_finetune ;;
  *) echo "usage: $0 [fusion|nets|finetune|all]   (SEEDS env to override seed list)"; exit 1 ;;
esac
echo "DONE. aggregate e.g.:  python scripts/aggregate_seeds.py --glob 'results/track3_full/fusion_rnafm_eng*'"

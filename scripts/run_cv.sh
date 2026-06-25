#!/usr/bin/env bash
# ===========================================================================
# 5-fold leakage-safe cross-validation on the full-length core settings, for
# ALL models (k-mer, FM frozen+head, engineered, length, RNATracker, DM3Loc,
# fusion, AND RNA-FM end-to-end fine-tune). Nets/fusion/fine-tune ≤50 epochs +
# early stopping. NOTE: fine-tune × 5 folds × 4 settings is by far the dominant
# compute cost (full backprop over sliding windows); plan GPU time accordingly.
#
# Step 1 generates 5 group-aware fold CSVs from each setting's existing
# split_assignments.csv (re-partition GROUPS, never samples). Step 2 runs every
# model × 5 folds × 4 settings via --split-assignments. FM embeddings are cached
# per setting (fold-independent), so FM/fusion folds are cheap.
#
# Settings: track3_full, track3_full_isoform (binary) + fine_full_gene,
#           fine_full_isoform (multi-label).
#
# Usage:
#   bash scripts/run_cv.sh folds     # generate fold CSVs only
#   bash scripts/run_cv.sh run       # run all models × folds (needs folds first)
#   bash scripts/run_cv.sh all
# Aggregate per model:
#   python scripts/aggregate_seeds.py --glob "results/track3_full/cv/fusion_rnafm_eng_fold*"
# ===========================================================================
set -euo pipefail

PY=python
TRAIN=scripts/train_rnafm_multilabel.py
MKCV=scripts/make_cv_folds.py
IN="mixed_bulkgene_isoform_neuropil"
ORTHO="ortholog/human_mouse_rat_gene_to_ortholog_group.tsv"
SP=results/_frozen_splits
M_RNAFM=./rnafm
M_DNABERT=./DNABERT-2-117M
K=5
FOLDS=$(seq 0 $((K - 1)))

C_BIN=(--label-scheme soma_vs_neurite --label-agg soft --source-mask --classifier logistic
       --min-support 150 --seed 0 --ortholog-map "$ORTHO")
C_FINE=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
        --min-support 200 --seed 0 --ortholog-map "$ORTHO")
NETS_EP=(--ts-epochs 50)

# setting -> sample-level ; scheme picked by name
level_of() { case "$1" in *isoform*) echo isoform_sequence_union;; *) echo gene;; esac; }
sched_of() { case "$1" in fine_*) echo fine;; *) echo bin;; esac; }
# source split to derive folds from: each setting's OWN kmer run split_assignments
# (on that setting's exact universe, with isoform-level / fine-level split_group).
src_split() { echo "results/$1/kmer/split_assignments.csv"; }

SETTINGS=(track3_full track3_full_isoform fine_full_gene fine_full_isoform)

gen_folds() {
  for s in "${SETTINGS[@]}"; do
    local src; src=$(src_split "$s")
    [ -f "$src" ] || { echo "[skip] missing frozen split $src for $s"; continue; }
    $PY $MKCV --from-split "$src" --out-prefix "$SP/cv_${s}" --k "$K" --seed 0
  done
}

run_all() {
  for s in "${SETTINGS[@]}"; do
    local lvl; lvl=$(level_of "$s")
    local C; if [ "$(sched_of "$s")" = fine ]; then C=("${C_FINE[@]}"); else C=("${C_BIN[@]}"); fi
    local OUT="results/$s/cv"; mkdir -p "$OUT"
    for j in $FOLDS; do
      local SPL="$SP/cv_${s}_fold${j}.csv"
      [ -f "$SPL" ] || { echo "[skip] missing $SPL (run 'folds' first)"; continue; }
      local base=(--input-dir "$IN" --region full --sample-level "$lvl" "${C[@]}" --split-assignments "$SPL")
      echo "### $s fold$j : k-mer"
      $PY $TRAIN "${base[@]}" --baseline kmer --kmer-k 4                         --output-dir "$OUT/kmer_fold$j"
      echo "### $s fold$j : length"
      $PY $TRAIN "${base[@]}" --features length                                 --output-dir "$OUT/length_fold$j"
      echo "### $s fold$j : engineered"
      $PY $TRAIN "${base[@]}" --features engineered                             --output-dir "$OUT/engineered_fold$j"
      echo "### $s fold$j : rnafm"
      $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022         --output-dir "$OUT/rnafm_fold$j"
      echo "### $s fold$j : dnabert2"
      $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 3000         --output-dir "$OUT/dnabert2_fold$j"
      echo "### $s fold$j : rnatracker"
      $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 31000 "${NETS_EP[@]}" --output-dir "$OUT/rnatracker_fold$j"
      echo "### $s fold$j : dm3loc"
      $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 31000 "${NETS_EP[@]}" --output-dir "$OUT/dm3loc_fold$j"
      echo "### $s fold$j : fusion"
      $PY $TRAIN "${base[@]}" --fusion --features fm engineered --model-dir "$M_RNAFM" \
        --max-tokens 1022 "${NETS_EP[@]}"                                       --output-dir "$OUT/fusion_rnafm_eng_fold$j"
      # RNA-FM end-to-end fine-tune (EXPENSIVE: full backprop, sliding windows ×
      # every fold × setting). ≤50 epochs + early stopping; full-coverage policy.
      echo "### $s fold$j : finetune"
      $PY $TRAIN "${base[@]}" --finetune --model-dir "$M_RNAFM" \
        --ft-long-seq-policy sliding_mean --ft-epochs 50                        --output-dir "$OUT/rnafm_finetune_sliding_fold$j"
    done
  done
}

case "${1:-all}" in
  folds) gen_folds ;;
  run)   run_all ;;
  all)   gen_folds; run_all ;;
  *) echo "usage: $0 [folds|run|all]"; exit 1 ;;
esac
echo "DONE. e.g.:  python scripts/aggregate_seeds.py --glob 'results/track3_full/cv/fusion_rnafm_eng_fold*'"

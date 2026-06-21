#!/usr/bin/env bash
# ===========================================================================
# 神经元 mRNA 定位 benchmark —— 三条区域 track + 受控消融 + 多标签平行
# 设计见 docs/EXPERIMENT_DESIGN.md
#
#   Track 1 (utr3) : 3'UTR,全部数据(bulk提取+原生isoform),gene + isoform
#   Track 2 (cds)  : CDS,bulk 全长,codon 模型入场
#   Track 3 (full) : 全长,bulk 全长,长程模型(信息最多)
#   消融           : 单模型 × {utr3,cds,full} 同基因交集
#   fine           : 上述关键 track 的 5 隔室多标签版本
#
# 模型族(每条 track 内,同 split/同标签/同评估,只换 encoder):
#   k-mer | UTR-BERT/RNA-FM/mRNA-FM/DNABERT-2(基础模型) | rnatracker/dm3loc(定位专用)
#
# 用法: bash scripts/run_all.sh [splits|track1a|track1b|track2|track3|ablation|fine|all]
# ===========================================================================
set -euo pipefail

PY=python
TRAIN=scripts/train_rnafm_multilabel.py
INPUT_DIR="全长_合并多标签输入_plus_mart_plus_isoform_files"   # bulk + isoform 同目录
ORTHO="ortholog/human_mouse_rat_gene_to_ortholog_group.tsv"
GTF=(data_训练/Homo_sapiens.GRCh38.116.gtf.gz
     data_训练/Mus_musculus.GRCm39.116.gtf.gz
     data_训练/Rattus_norvegicus.GRCr8.116.gtf.gz)

M_RNAFM=./rnafm            # RNA-FM   (nt; utr3/cds/full)
M_MRNAFM=./rnafm_codon     # mRNA-FM  (codon; 仅 cds)
M_UTRBERT=./utrbert-3mer   # UTR-BERT (仅 utr3)
M_DNABERT=./DNABERT-2-117M # DNABERT-2(nt; utr3/cds/full;同脚本加载,名字含 "dnabert" 即走专用分支)
# 注:mRNABERT 未纳入——它需区域感知双 tokenization(UTR=nt/CDS=codon,空格分隔,需 CDS 边界),
#     与本受控单序列管线不兼容;若日后写专用 adapter 再加。

OUT=results
SPLIT_DIR=$OUT/_frozen_splits
# 物种范围:human 仅 ~289 基因、无法做有意义的 per-species 评估,默认只用 mouse+rat
# 做干净的"啮齿类"benchmark。想保留 human 就把这行设为空: SPECIES=()
SPECIES=(--species mouse rat)
COMMON=(--label-scheme soma_vs_neurite --label-agg soft --source-mask
        --classifier logistic --min-support 150 --seed 0 --ortholog-map "$ORTHO"
        "${SPECIES[@]}")

# 原生 3'UTR 源(已是 3'UTR,跑 --region utr3 时原样通过,不 GTF 提取)。
# 含 blank sequence_type 的 Andreassi 必须在此列出,否则会被当全长去提取。
# 子串大小写不敏感匹配 source 列;按你数据的 source 取值核对。
NATIVE=(andreassi taliaferro ciolli tushev mikl isodend)
mkdir -p "$SPLIT_DIR"

# ---------------------------------------------------------------------------
# 0) 冻结 split
# ---------------------------------------------------------------------------
make_splits() {
  echo "### split U1 (utr3, gene) — Track 1A/1B 共用"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region utr3 --sample-level gene \
    --gtf "${GTF[@]}" --native-region-sources "${NATIVE[@]}" \
    --model-dir "$M_RNAFM" "${COMMON[@]}" --output-dir "$SPLIT_DIR/U1_gene"
  cp "$SPLIT_DIR/U1_gene/split_assignments.csv" "$SPLIT_DIR/split_U1_gene.csv"

  echo "### split U2 (cds, gene) — Track 2 / 消融"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region cds --sample-level gene \
    --gtf "${GTF[@]}" --model-dir "$M_RNAFM" "${COMMON[@]}" --output-dir "$SPLIT_DIR/U2_gene"
  cp "$SPLIT_DIR/U2_gene/split_assignments.csv" "$SPLIT_DIR/split_U2_gene.csv"

  echo "### split U3 (full, gene) — Track 3"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region full --sample-level gene \
    --model-dir "$M_RNAFM" "${COMMON[@]}" --output-dir "$SPLIT_DIR/U3_gene"
  cp "$SPLIT_DIR/U3_gene/split_assignments.csv" "$SPLIT_DIR/split_U3_gene.csv"

  # 消融 split = cds 基因交集(cds⊆utr3⊆full,这批基因三区域都能跑)
  cp "$SPLIT_DIR/split_U2_gene.csv" "$SPLIT_DIR/split_Uablate_gene.csv"
}

# ---------------------------------------------------------------------------
# Track 1A —— 3'UTR × gene
# ---------------------------------------------------------------------------
track1a() {
  local SP="$SPLIT_DIR/split_U1_gene.csv" O="$OUT/track1a_gene"
  local base=(--input-dir "$INPUT_DIR" --region utr3 --sample-level gene
              --gtf "${GTF[@]}" --native-region-sources "${NATIVE[@]}"
              "${COMMON[@]}" --split-assignments "$SP")
  $PY $TRAIN "${base[@]}" --baseline kmer --kmer-k 4               --output-dir "$O/kmer"
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 2000      --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 2000      --output-dir "$O/dm3loc"
  $PY $TRAIN "${base[@]}" --model-dir "$M_UTRBERT" --max-tokens 510  --output-dir "$O/utrbert"
  $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O/rnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 2000 --output-dir "$O/dnabert2"
}

# ---------------------------------------------------------------------------
# Track 1B —— 3'UTR × isoform(复用 U1_gene split)
# ---------------------------------------------------------------------------
track1b() {
  local SP="$SPLIT_DIR/split_U1_gene.csv" O="$OUT/track1b_isoform"
  local base=(--input-dir "$INPUT_DIR" --region utr3 --sample-level isoform_sequence_union
              --gtf "${GTF[@]}" --native-region-sources "${NATIVE[@]}"
              "${COMMON[@]}" --split-assignments "$SP")
  $PY $TRAIN "${base[@]}" --baseline kmer --kmer-k 4               --output-dir "$O/kmer"
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 2000      --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 2000      --output-dir "$O/dm3loc"
  $PY $TRAIN "${base[@]}" --model-dir "$M_UTRBERT" --max-tokens 510  --output-dir "$O/utrbert"
  $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O/rnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 2000 --output-dir "$O/dnabert2"
}

# ---------------------------------------------------------------------------
# Track 2 —— CDS × gene(codon 入场)
# ---------------------------------------------------------------------------
track2() {
  local SP="$SPLIT_DIR/split_U2_gene.csv" O="$OUT/track2_gene"
  local base=(--input-dir "$INPUT_DIR" --region cds --sample-level gene
              --gtf "${GTF[@]}" "${COMMON[@]}" --split-assignments "$SP")
  $PY $TRAIN "${base[@]}" --baseline kmer --kmer-k 4               --output-dir "$O/kmer"
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 4000      --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 4000      --output-dir "$O/dm3loc"
  $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"  --max-tokens 1022 --output-dir "$O/rnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_MRNAFM" --max-tokens 1024 --output-dir "$O/mrnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O/dnabert2"
}

# ---------------------------------------------------------------------------
# Track 3 —— 全长 × gene(信息最多;不含 mRNA-FM/UTR-BERT)
# ---------------------------------------------------------------------------
track3() {
  local SP="$SPLIT_DIR/split_U3_gene.csv" O="$OUT/track3_full"
  local base=(--input-dir "$INPUT_DIR" --region full --sample-level gene
              "${COMMON[@]}" --split-assignments "$SP")
  $PY $TRAIN "${base[@]}" --baseline kmer --kmer-k 4               --output-dir "$O/kmer"
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 6000      --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 6000      --output-dir "$O/dm3loc"
  $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --window-pool mean --output-dir "$O/rnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O/dnabert2"
}

# ---------------------------------------------------------------------------
# 消融 —— RNA-FM × {utr3,cds,full},同一批 cds 基因 + 同 split
# ---------------------------------------------------------------------------
ablation() {
  # 同基因交集 = CDS 基因(cds⊆utr3⊆full)。--restrict-to-split 让 utr3/full 只保留
  # 这批基因,三臂同基因、同 split、只变区域 → 真正可比的 3'UTR vs CDS vs full。
  local SP="$SPLIT_DIR/split_Uablate_gene.csv"
  for R in utr3 cds full; do
    local extra=(); [ "$R" != "full" ] && extra=(--gtf "${GTF[@]}")
    local nat=(); [ "$R" = "utr3" ] && nat=(--native-region-sources "${NATIVE[@]}")
    $PY $TRAIN --input-dir "$INPUT_DIR" --region "$R" --sample-level gene \
      "${extra[@]}" "${nat[@]}" "${COMMON[@]}" --split-assignments "$SP" --restrict-to-split \
      --model-dir "$M_RNAFM" --max-tokens 1022 --output-dir "$OUT/ablation_region/rnafm_$R"
  done
}

# ---------------------------------------------------------------------------
# fine —— 5 隔室多标签(Cell_body/Dendrite/Neuropil/Axon/Neurite),关键 track 各一遍
#   多标签必须 --label-scheme fine --source-mask;min-support 提到 200
# ---------------------------------------------------------------------------
fine() {
  local C=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
           --min-support 200 --seed 0 --ortholog-map "$ORTHO")
  # 3'UTR fine(全部数据,gene)
  local SP1="$SPLIT_DIR/split_U1_gene.csv" O1="$OUT/fine_utr3_gene"
  local b1=(--input-dir "$INPUT_DIR" --region utr3 --sample-level gene --gtf "${GTF[@]}"
            --native-region-sources "${NATIVE[@]}" "${C[@]}" --split-assignments "$SP1")
  $PY $TRAIN "${b1[@]}" --baseline kmer --kmer-k 4               --output-dir "$O1/kmer"
  $PY $TRAIN "${b1[@]}" --arch dm3loc --ts-max-len 2000          --output-dir "$O1/dm3loc"
  $PY $TRAIN "${b1[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O1/rnafm"
  $PY $TRAIN "${b1[@]}" --model-dir "$M_DNABERT" --max-tokens 2000 --output-dir "$O1/dnabert2"
  # 全长 fine(bulk,gene)
  local SP3="$SPLIT_DIR/split_U3_gene.csv" O3="$OUT/fine_full_gene"
  local b3=(--input-dir "$INPUT_DIR" --region full --sample-level gene
            "${C[@]}" --split-assignments "$SP3")
  $PY $TRAIN "${b3[@]}" --baseline kmer --kmer-k 4               --output-dir "$O3/kmer"
  $PY $TRAIN "${b3[@]}" --arch dm3loc --ts-max-len 6000          --output-dir "$O3/dm3loc"
  $PY $TRAIN "${b3[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O3/rnafm"
  $PY $TRAIN "${b3[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O3/dnabert2"
}

case "${1:-all}" in
  splits)   make_splits ;;
  track1a)  track1a ;;
  track1b)  track1b ;;
  track2)   track2 ;;
  track3)   track3 ;;
  ablation) ablation ;;
  fine)     fine ;;
  all)      make_splits; track1a; track1b; track2; track3; ablation; fine ;;
  *) echo "usage: $0 [splits|track1a|track1b|track2|track3|ablation|fine|all]"; exit 1 ;;
esac

echo "DONE. 合并主表:  find $OUT -name overall_metrics.csv"

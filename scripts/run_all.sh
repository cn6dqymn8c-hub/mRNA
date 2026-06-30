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
INPUT_DIR="data_训练/mixed_bulkgene_isoform_neuropil"   # bulk + isoform 同目录(已修正物种的重抽数据)
ORTHO="ortholog/human_mouse_rat_gene_to_ortholog_group.tsv"
GTF=(data_训练/Homo_sapiens.GRCh38.116.gtf.gz
     data_训练/Mus_musculus.GRCm39.116.gtf.gz
     data_训练/Rattus_norvegicus.GRCr8.116.gtf.gz)

M_RNAFM=./rnafm            # RNA-FM   (nt; utr3/cds/full)
M_MRNAFM=./mrnafm          # mRNA-FM  (codon; 仅 cds)
M_UTRBERT=./utrbert-3mer   # UTR-BERT (仅 utr3)
M_DNABERT=./DNABERT-2-117M # DNABERT-2(nt; utr3/cds/full;同脚本加载,名字含 "dnabert" 即走专用分支)
# 注:mRNABERT 未纳入——它需区域感知双 tokenization(UTR=nt/CDS=codon,空格分隔,需 CDS 边界),
#     与本受控单序列管线不兼容;若日后写专用 adapter 再加。

OUT=results
SPLIT_DIR=$OUT/_frozen_splits
# 物种范围:统一【全物种】(mouse+rat+human,以 mixed_bulkgene_isoform_neuropil 为准),
# 与 run_cv.sh / run_seeds.sh 保持同一数据 universe。如需只跑啮齿类: SPECIES=(--species mouse rat)
SPECIES=()
COMMON=(--label-scheme soma_vs_neurite --label-agg soft --source-mask
        --classifier logistic --min-support 150 --seed 0 --ortholog-map "$ORTHO"
        "${SPECIES[@]}")

# 原生 3'UTR 源(已是 3'UTR,跑 --region utr3 时原样通过,不 GTF 提取)。
# 子串大小写不敏感匹配 source 列。新数据里需要显式标的只有 blank-seqtype 的
# Andreassi(source="AllIsoforms_coordinates"),用 isoform 一个词即可命中;
# 其余原生源(Tushev/Ciolli/Mikl/isoDend 的 *3utr*、Taliaferro 的 ALE)已由
# seqtype_is_native_utr3 自动识别。已核验无任何全长 cDNA 源 source 含 "isoform"。
NATIVE=(isoform)
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
# 0b) fine 专用冻结 split —— 必须与 fine 同一 universe 独立生成。
#   二分类(soma_vs_neurite)会因 --source-mask 删除一批 soma-only negative,
#   fine(多标签)不这样删,样本更多;复用二分类 split 会 "does not cover N samples"。
#   split 与模型无关(由 groups+labels+seed 决定),故用 --baseline kmer 生成——
#   不需要 FM 权重,只 utr3/cds 需要 GTF。
# ---------------------------------------------------------------------------
make_fine_splits() {
  local FC=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
            --min-support 200 --seed 0 --ortholog-map "$ORTHO" "${SPECIES[@]}" --baseline kmer)
  echo "### split fine-utr3-gene"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region utr3 --sample-level gene \
    --gtf "${GTF[@]}" --native-region-sources "${NATIVE[@]}" "${FC[@]}" \
    --output-dir "$SPLIT_DIR/fineU1_gene"
  cp "$SPLIT_DIR/fineU1_gene/split_assignments.csv" "$SPLIT_DIR/split_fine_utr3_gene.csv"

  echo "### split fine-cds-gene"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region cds --sample-level gene \
    --gtf "${GTF[@]}" "${FC[@]}" --output-dir "$SPLIT_DIR/fineU2_gene"
  cp "$SPLIT_DIR/fineU2_gene/split_assignments.csv" "$SPLIT_DIR/split_fine_cds_gene.csv"

  echo "### split fine-full-gene"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region full --sample-level gene \
    "${FC[@]}" --output-dir "$SPLIT_DIR/fineU3_gene"
  cp "$SPLIT_DIR/fineU3_gene/split_assignments.csv" "$SPLIT_DIR/split_fine_full_gene.csv"

  echo "### split fine-full-isoform (供 run_seeds / run_cv 使用)"
  $PY $TRAIN --input-dir "$INPUT_DIR" --region full --sample-level isoform_sequence_union \
    "${FC[@]}" --output-dir "$SPLIT_DIR/fineU3_isoform"
  cp "$SPLIT_DIR/fineU3_isoform/split_assignments.csv" "$SPLIT_DIR/split_fine_full_isoform.csv"
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
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 31000     --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 31000     --output-dir "$O/dm3loc"
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
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 31000     --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 31000     --output-dir "$O/dm3loc"
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
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 31000     --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 31000     --output-dir "$O/dm3loc"
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
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 31000     --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 31000     --output-dir "$O/dm3loc"
  $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --window-pool mean --output-dir "$O/rnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O/dnabert2"
}

# ---------------------------------------------------------------------------
# Track 3B —— 全长 × isoform_sequence_union(继承 U3 gene split,保留 isoform 序列)
# ---------------------------------------------------------------------------
track3isoform() {
  local SP="$SPLIT_DIR/split_U3_gene.csv" O="$OUT/track3_full_isoform"
  local base=(--input-dir "$INPUT_DIR" --region full --sample-level isoform_sequence_union
              "${COMMON[@]}" --split-assignments "$SP")
  $PY $TRAIN "${base[@]}" --baseline kmer --kmer-k 4               --output-dir "$O/kmer"
  $PY $TRAIN "${base[@]}" --features length                         --output-dir "$O/length"
  $PY $TRAIN "${base[@]}" --features engineered                     --output-dir "$O/engineered"
  $PY $TRAIN "${base[@]}" --arch rnatracker --ts-max-len 31000      --output-dir "$O/rnatracker"
  $PY $TRAIN "${base[@]}" --arch dm3loc     --ts-max-len 31000      --output-dir "$O/dm3loc"
  $PY $TRAIN "${base[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O/rnafm"
  $PY $TRAIN "${base[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O/dnabert2"
  $PY $TRAIN "${base[@]}" --fusion --features fm engineered --model-dir "$M_RNAFM" \
       --max-tokens 1022 --output-dir "$O/fusion_rnafm_eng"
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
# fine 区域消融 —— RNA-FM × {utr3,cds,full},同一批 fine-CDS 基因 + 同 split。
#   §3.2 区域比较的"可比"版本:--restrict-to-split 让三臂只保留 fine-CDS 基因
#   (cds⊆utr3⊆full),同基因、同 split、只变区域 → 区域差异才是唯一变量。
#   依赖 make_fine_splits 先产 split_fine_cds_gene.csv。
# ---------------------------------------------------------------------------
fineablation() {
  local FC=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
            --min-support 200 --seed 0 --ortholog-map "$ORTHO" "${SPECIES[@]}")
  local SP="$SPLIT_DIR/split_fine_cds_gene.csv"
  for R in utr3 cds full; do
    local extra=(); [ "$R" != "full" ] && extra=(--gtf "${GTF[@]}")
    local nat=(); [ "$R" = "utr3" ] && nat=(--native-region-sources "${NATIVE[@]}")
    $PY $TRAIN --input-dir "$INPUT_DIR" --region "$R" --sample-level gene \
      "${extra[@]}" "${nat[@]}" "${FC[@]}" --split-assignments "$SP" --restrict-to-split \
      --model-dir "$M_RNAFM" --max-tokens 1022 --output-dir "$OUT/fine_ablation_region/rnafm_$R"
  done
}

# ---------------------------------------------------------------------------
# fine —— 5 隔室多标签(Cell_body/Dendrite/Neuropil/Axon/Neurite),关键 track 各一遍
#   多标签必须 --label-scheme fine --source-mask;min-support 提到 200
# ---------------------------------------------------------------------------
fine() {
  local C=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
           --min-support 200 --seed 0 --ortholog-map "$ORTHO")

  # 3'UTR fine(全部数据,gene): UTR-BERT 在这里入场。
  local SP1="$SPLIT_DIR/split_fine_utr3_gene.csv" O1="$OUT/fine_utr3_gene"
  local b1=(--input-dir "$INPUT_DIR" --region utr3 --sample-level gene --gtf "${GTF[@]}"
            --native-region-sources "${NATIVE[@]}" "${C[@]}" --split-assignments "$SP1")
  $PY $TRAIN "${b1[@]}" --baseline kmer --kmer-k 4               --output-dir "$O1/kmer"
  $PY $TRAIN "${b1[@]}" --features length                         --output-dir "$O1/length"
  $PY $TRAIN "${b1[@]}" --features engineered                     --output-dir "$O1/engineered"
  $PY $TRAIN "${b1[@]}" --arch rnatracker --ts-max-len 31000      --output-dir "$O1/rnatracker"
  $PY $TRAIN "${b1[@]}" --arch dm3loc     --ts-max-len 31000      --output-dir "$O1/dm3loc"
  $PY $TRAIN "${b1[@]}" --model-dir "$M_UTRBERT" --max-tokens 510  --output-dir "$O1/utrbert"
  $PY $TRAIN "${b1[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O1/rnafm"
  $PY $TRAIN "${b1[@]}" --model-dir "$M_DNABERT" --max-tokens 2000 --output-dir "$O1/dnabert2"
  $PY $TRAIN "${b1[@]}" --fusion --features fm engineered --model-dir "$M_RNAFM" \
       --max-tokens 1022 --output-dir "$O1/fusion_rnafm_eng"

  # CDS fine(bulk,gene)— U2;cds 专属 codon 模型 mRNA-FM 在此入场(对齐二分类 track2)
  local SP2="$SPLIT_DIR/split_fine_cds_gene.csv" O2="$OUT/fine_cds_gene"
  local b2=(--input-dir "$INPUT_DIR" --region cds --sample-level gene --gtf "${GTF[@]}"
            "${C[@]}" --split-assignments "$SP2")
  $PY $TRAIN "${b2[@]}" --baseline kmer --kmer-k 4               --output-dir "$O2/kmer"
  $PY $TRAIN "${b2[@]}" --features length                         --output-dir "$O2/length"
  $PY $TRAIN "${b2[@]}" --features engineered                     --output-dir "$O2/engineered"
  $PY $TRAIN "${b2[@]}" --arch rnatracker --ts-max-len 31000      --output-dir "$O2/rnatracker"
  $PY $TRAIN "${b2[@]}" --arch dm3loc     --ts-max-len 31000      --output-dir "$O2/dm3loc"
  $PY $TRAIN "${b2[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O2/rnafm"
  $PY $TRAIN "${b2[@]}" --model-dir "$M_MRNAFM"  --max-tokens 1024 --output-dir "$O2/mrnafm"
  $PY $TRAIN "${b2[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O2/dnabert2"
  $PY $TRAIN "${b2[@]}" --fusion --features fm engineered --model-dir "$M_MRNAFM" \
       --max-tokens 1024 --output-dir "$O2/fusion_mrnafm_eng"

  # 全长 fine(bulk,gene)
  local SP3="$SPLIT_DIR/split_fine_full_gene.csv" O3="$OUT/fine_full_gene"
  local b3=(--input-dir "$INPUT_DIR" --region full --sample-level gene
            "${C[@]}" --split-assignments "$SP3")
  $PY $TRAIN "${b3[@]}" --baseline kmer --kmer-k 4               --output-dir "$O3/kmer"
  $PY $TRAIN "${b3[@]}" --features length                         --output-dir "$O3/length"
  $PY $TRAIN "${b3[@]}" --features engineered                     --output-dir "$O3/engineered"
  $PY $TRAIN "${b3[@]}" --arch rnatracker --ts-max-len 31000      --output-dir "$O3/rnatracker"
  $PY $TRAIN "${b3[@]}" --arch dm3loc     --ts-max-len 31000      --output-dir "$O3/dm3loc"
  $PY $TRAIN "${b3[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O3/rnafm"
  $PY $TRAIN "${b3[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O3/dnabert2"
  $PY $TRAIN "${b3[@]}" --fusion --features fm engineered --model-dir "$M_RNAFM" \
       --max-tokens 1022 --output-dir "$O3/fusion_rnafm_eng"

  # 全长 fine(isoform_sequence_union): 使用 fine isoform 自己的 split,供主表/CV/seed 统一引用。
  local SP4="$SPLIT_DIR/split_fine_full_isoform.csv" O4="$OUT/fine_full_isoform"
  local b4=(--input-dir "$INPUT_DIR" --region full --sample-level isoform_sequence_union
            "${C[@]}" --split-assignments "$SP4")
  $PY $TRAIN "${b4[@]}" --baseline kmer --kmer-k 4               --output-dir "$O4/kmer"
  $PY $TRAIN "${b4[@]}" --features length                         --output-dir "$O4/length"
  $PY $TRAIN "${b4[@]}" --features engineered                     --output-dir "$O4/engineered"
  $PY $TRAIN "${b4[@]}" --arch rnatracker --ts-max-len 31000      --output-dir "$O4/rnatracker"
  $PY $TRAIN "${b4[@]}" --arch dm3loc     --ts-max-len 31000      --output-dir "$O4/dm3loc"
  $PY $TRAIN "${b4[@]}" --model-dir "$M_RNAFM"   --max-tokens 1022 --output-dir "$O4/rnafm"
  $PY $TRAIN "${b4[@]}" --model-dir "$M_DNABERT" --max-tokens 3000 --output-dir "$O4/dnabert2"
  $PY $TRAIN "${b4[@]}" --fusion --features fm engineered --model-dir "$M_RNAFM" \
       --max-tokens 1022 --output-dir "$O4/fusion_rnafm_eng"
}

# ---------------------------------------------------------------------------
# fusion —— 提出模型:RNA-FM 嵌入 ⊕ 工程特征,注意力融合头。每个 track 跑一个,
#   之后用 bootstrap 比 fusion vs kmer,看能否成为"显著最优"。FM 块用 RNA-FM
#   (它在 Track1B 是唯一显著超 k-mer 的);Track2 想试 codon 可把 M_RNAFM 换 M_MRNAFM。
# ---------------------------------------------------------------------------
fusion() {
  # 1A: 3'UTR gene
  $PY $TRAIN --input-dir "$INPUT_DIR" --region utr3 --sample-level gene --gtf "${GTF[@]}" \
    --native-region-sources "${NATIVE[@]}" "${COMMON[@]}" --split-assignments "$SPLIT_DIR/split_U1_gene.csv" \
    --fusion --features fm engineered --model-dir "$M_RNAFM" \
    --max-tokens 1022 --output-dir "$OUT/track1a_gene/fusion_rnafm_eng"
  # 1B: 3'UTR isoform (重点战场)
  $PY $TRAIN --input-dir "$INPUT_DIR" --region utr3 --sample-level isoform_sequence_union --gtf "${GTF[@]}" \
    --native-region-sources "${NATIVE[@]}" "${COMMON[@]}" --split-assignments "$SPLIT_DIR/split_U1_gene.csv" \
    --fusion --features fm engineered --model-dir "$M_RNAFM" \
    --max-tokens 1022 --output-dir "$OUT/track1b_isoform/fusion_rnafm_eng"
  # 2: CDS gene
  $PY $TRAIN --input-dir "$INPUT_DIR" --region cds --sample-level gene --gtf "${GTF[@]}" \
    "${COMMON[@]}" --split-assignments "$SPLIT_DIR/split_U2_gene.csv" \
    --fusion --features fm engineered --model-dir "$M_MRNAFM" \
    --max-tokens 1024 --output-dir "$OUT/track2_gene/fusion_mrnafm_eng"
  # 3: full gene
  $PY $TRAIN --input-dir "$INPUT_DIR" --region full --sample-level gene \
    "${COMMON[@]}" --split-assignments "$SPLIT_DIR/split_U3_gene.csv" \
    --fusion --features fm engineered --model-dir "$M_RNAFM" \
    --max-tokens 1022 --output-dir "$OUT/track3_full/fusion_rnafm_eng"
  # 3B: full isoform
  $PY $TRAIN --input-dir "$INPUT_DIR" --region full --sample-level isoform_sequence_union \
    "${COMMON[@]}" --split-assignments "$SPLIT_DIR/split_U3_gene.csv" \
    --fusion --features fm engineered --model-dir "$M_RNAFM" \
    --max-tokens 1022 --output-dir "$OUT/track3_full_isoform/fusion_rnafm_eng"
}

# ---------------------------------------------------------------------------
# deepseek —— 通用 LLM(DeepSeekMoE)经 QLoRA 适配为序列编码器,作为受控对照接入
#   主 benchmark(全长×isoform,fine split)。保留原文本 BPE 分词器(跨模态迁移是被
#   检验而非假设)。重活,不进 all;单卡 24GB(4-bit)即可,用 DS_GPU 选空闲卡。
#   产物 results/fine_full_isoform/deepseek_moe → make_fine_report 自动纳入对比。
#   用法: DS_GPU=2 bash scripts/run_all.sh deepseek
# ---------------------------------------------------------------------------
deepseek() {
  local C=(--label-scheme fine --label-agg soft --source-mask --classifier logistic
           --min-support 200 --seed 0 --ortholog-map "$ORTHO" "${SPECIES[@]}")
  local SP4="$SPLIT_DIR/split_fine_full_isoform.csv" O4="$OUT/fine_full_isoform"
  CUDA_VISIBLE_DEVICES="${DS_GPU:-0}" $PY $TRAIN --input-dir "$INPUT_DIR" --region full \
    --sample-level isoform_sequence_union "${C[@]}" --split-assignments "$SP4" \
    --llm-encoder "${DS_MODEL:-deepseek-ai/deepseek-moe-16b-base}" --llm-4bit \
    --llm-max-tokens "${DS_MAXTOK:-1024}" --llm-batch "${DS_BATCH:-4}" \
    --output-dir "$O4/deepseek_moe"
}

# ---------------------------------------------------------------------------
# deploy —— 最终部署模型:全长 fusion(基准里最优,ROC-AUC 0.70),用 --train-on-all
#   在【全部样本】上重拟合(无 held-out;报告指标用 benchmark 的 fusion 那次,不是这次的
#   in-sample 数)。产出 fusion_model.joblib + run_config.json + label_thresholds.csv,
#   之后用 scripts/predict.py 对新序列打分。
# ---------------------------------------------------------------------------
deploy() {
  $PY $TRAIN --input-dir "$INPUT_DIR" --region full --sample-level gene \
    "${COMMON[@]}" --fusion --features fm engineered --model-dir "$M_RNAFM" \
    --max-tokens 1022 --train-on-all --output-dir "$OUT/final_deploy_full_fusion"
  echo "部署模型已保存到 $OUT/final_deploy_full_fusion"
  echo "打分新序列:"
  echo "  $PY scripts/predict.py --artifact-dir $OUT/final_deploy_full_fusion \\"
  echo "       --input new_transcripts.csv --output predictions.csv"
}

case "${1:-all}" in
  splits)     make_splits; make_fine_splits ;;
  finesplits) make_fine_splits ;;
  fusion)   fusion ;;
  deepseek) deepseek ;;
  deploy)   deploy ;;
  track1a)  track1a ;;
  track1b)  track1b ;;
  track2)   track2 ;;
  track3)   track3 ;;
  track3isoform) track3isoform ;;
  ablation) ablation ;;
  fineablation) fineablation ;;
  fine)     fine ;;
  fineonly) make_fine_splits; fine; fineablation ;;
  all)      make_splits; make_fine_splits; track1a; track1b; track2; track3; track3isoform; ablation; fine; fineablation ;;
  *) echo "usage: $0 [splits|finesplits|track1a|track1b|track2|track3|track3isoform|ablation|fineablation|fine|fineonly|fusion|deepseek|deploy|all]"; exit 1 ;;
esac

echo "DONE. 合并主表:  find $OUT -name overall_metrics.csv"

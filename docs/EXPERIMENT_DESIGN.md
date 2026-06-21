# 神经元 mRNA 亚细胞定位 —— Benchmark 实验设计（可执行版）

每个实验对应具体命令、数据子集、产物。所有结论基于对仓库真实数据的核查。

---

## 0. 数据现状（核查结果）

### 两类数据，由 `sequence_type` 区分

| 类型 | `sequence_type` | transcript_id | 中位长度 | 得到目标区域 |
|---|---|---|---|---|
| **全长 / bulk** | `cdna` | ✅ | 1400–2800 nt | **GTF 提取 3'UTR / CDS** |
| **原生 isoform** | `isoform_3utr` / `isoform_spliced_3utr` / `reported_3utr_..._fragment` / `isoform_3utr_TableS7` / `ALE_genomic_interval` / **空** | 视源 | ~600–1000 nt | **本身就是 3'UTR，原样用** |

**分流规则（`apply_region` 已实现并验证）**：跑 `--region utr3` 时——
(1) `--native-region-sources` 命中的源、或 seqtype 含 `3utr/utr3/ale/genomic_interval` 标记 → **原样通过**；
(2) `cdna/transcript/mrna` → **GTF 提取 3'UTR**；
(3) **空 `sequence_type` → 默认当全长去 GTF 提取**(不再 passthrough)。
`--region cds` 时全长行提 CDS,其余(含 3'UTR/ALE/空)无 CDS → 丢弃。

> ⚠️ **空 seqtype 有歧义**:它既可能是原生 3'UTR(Andreassi)，也可能是合并文件里的全长 bulk 代表转录本。
> 若默认 passthrough,后者会被**当全长喂进 3'UTR 实验、silently 污染**。故默认改为"空→当全长提取"(misclassify
> 时会变成"不在 GTF→丢弃",loud 可见),并用 `--native-region-sources <子串...>` 显式标出真正的原生 3'UTR 源
> (含 blank 的 Andreassi 必须列出)。被路由到 blank-提取的源会打印出来供核对。
> 已用单元测试验证:blank bulk→提取、native 源(含 blank Ciolli/Andreassi)→通过、cdna→提取。

### isoform 水平数据（完整 6 源）

| 源 | 行数 | 物种 | within-gene soma↔neurite 对立基因 |
|---|---|---|---|
| **Ciolli** | 1173 | mouse | **580**（配对金标准） |
| **Tushev** | 2782 | mouse | 13 |
| **Mikl** | 536 | mouse | 36 |
| **isoDend** | 298 | mouse | 0（全 Dendrite） |
| **Andreassi** | 8290 | **rat** | 3（98% Cell_body） |
| **Taliaferro ALE** | 412 | mouse | 0（全 Neurite） |
| 合计去重 | 13491 | mouse+rat | **~625** |

判别核心 ~625 基因，Ciolli(580) 主导，Andreassi 带来 rat。物种极偏(mouse≫rat≫human)，不做跨物种强声称。

### ⚠️ Ciolli 长度混淆
soma 3'UTR 中位 1699nt、neurite 599nt；90.2% 配对 neurite 更短 → 纯长度分类器即 90.2% 配对准确率。
isoform 判别结论**必须先扣长度**(§5)。

---

## 1. 三个正交的轴

| 轴 | 取值 | 控制 |
|---|---|---|
| **区域** | utr3 / cds / full | `--region` |
| **模型** | 见下表 | `--baseline` / `--arch` / `--model-dir` |
| **粒度** | gene / isoform | `--sample-level` |

**铁律**：同一对比内，数据子集 + split + 头 + 标签方案一致，唯一变量是模型。

### 模型族 → 调用方式 → 可吃区域（**全部走同一个 `train_rnafm_multilabel.py`,同 split/标签/评估**）

| 族 | 模型 | 调用 | tokenizer | 可吃区域 |
|---|---|---|---|---|
| 传统基线 | k-mer | `--baseline kmer` | — | 任意 |
| **定位专用(重训)** | RNATracker 式 | `--arch rnatracker` | one-hot | 任意 |
| **定位专用(重训)** | DM3Loc 式 | `--arch dm3loc` | one-hot | 任意 |
| RNA 基础模型 | UTR-BERT | `--model-dir <utrbert>` | 3-mer nt(≈508) | utr3 |
| RNA 基础模型 | RNA-FM | `--model-dir <rnafm>` | nt(≈1022) | utr3 / cds / full |
| RNA 基础模型 | mRNA-FM | `--model-dir <rnafm_codon>` | **codon（需阅读框）** | **仅 cds** |
| DNA 基础模型 | DNABERT-2 | `--model-dir <DNABERT-2-117M>` | BPE nt(ALiBi 长程) | utr3 / cds / full |

- **DNABERT-2 不再单独跑**：`load_model` 检测目录名含 `dnabert` 即走非-multimolecule 分支(nt/token=1、
  按核苷酸滑窗、ALiBi 无硬上限)，与其它模型同一冻结嵌入 pipeline → 对比只差 encoder。
  独立 `train_dnbert.py` 仅用于 DNABERT-2 端到端 fine-tune。
- **定位专用架构**(`--arch rnatracker|dm3loc`)从零训练于 one-hot，用与基础模型**完全相同**的
  split/标签/masked-BCE/评估，作"FM vs 专用 vs k-mer"对照。长度由 `--ts-max-len` 控制。
  注：是 RNATracker/DM3Loc 的**紧凑复现**(非原版权重)，作受控基线。
- **`--features` 特征堆叠**(可选,默认不启用,对其它实验零影响):把多组特征拼成一个矩阵喂 head,
  如 `--features fm engineered`。块:`fm`(嵌入)/`kmer`/`engineered`(长度+GC+二核苷酸+定位 motif)/
  `length`(仅长度=**混淆基线**)/`structure`(MFE,需 ViennaRNA)。**冲效果时只在最优配置上加一两个增强 run**;
  含长度的块会涨分但部分来自长度混淆,务必同时跑 `--features length` 量化其贡献。
  公平性:若做对比,要么 engineered 单独成列、要么对所有模型同等加,别只偏加给一个。
- **mRNA-FM 仅 cds，不进 full**：codon tokenizer 需阅读框；`apply_region` 提取的 CDS 起点是起始密码子
  (in-frame)，token=真实密码子。而 full 含任意长 5'UTR，从位置 0 切三联体会**与真实帧错位**、且 UTR 无
  密码子结构，codon 解释不干净。故 Track 3(full)不含 mRNA-FM；只有 nt/BPE 模型(RNA-FM/DNABERT-2)与
  one-hot/k-mer 可吃 full。

---

## 2. gene-level vs isoform-level

同一份全量数据的两种折叠规则，两个实验都用全部 source。

- **gene-level**（`--sample-level gene`）：一基因一行、标签并集。行少、样本独立。**头条对比。**
- **isoform-level**（`--sample-level isoform_sequence_union`）：每个不同 3'UTR 各一行(无 transcript_id
  的源**用序列做身份**)，相同序列合并。**分辨率分析。**

isoform-level 是 gene-level 的**超集**(行更多、阳性更多)，不是更少；无第三个"混合集"(拼接=复制+泄漏)；
两者**不 ensemble**(成员相关、增益小且易泄漏)。

---

## 3. 实验清单（三条区域 track + 受控消融 + fine 多标签）

二分类统一 `--label-scheme soma_vs_neurite --label-agg soft --source-mask`。

**区域是头条轴**：full 含信息最多，但 mean-pool 可能稀释 3'UTR 信号 → "full 最强"是待检验假设。

| 实验 | region | sample-level | 数据宇宙 | 模型 | 冻结 split |
|---|---|---|---|---|---|
| **1A** | utr3 | gene | 全部 3'UTR(bulk+isoform) | k-mer/rnatracker/dm3loc/UTR-BERT/RNA-FM/DNABERT-2 | `split_U1_gene` |
| **1B** | utr3 | isoform_sequence_union | 同上 | 同上 | **复用** `split_U1_gene` |
| **2A** | cds | gene | bulk 全长 | k-mer/rnatracker/dm3loc/RNA-FM/**mRNA-FM**/DNABERT-2 | `split_U2_gene` |
| **3A** | full | gene | bulk 全长 | k-mer/rnatracker/dm3loc/RNA-FM/DNABERT-2 | `split_U3_gene` |
| **消融** | utr3/cds/full | gene | 三区域基因交集 | RNA-FM 单模型 | `split_Uablate_gene` |
| **fine** | utr3 & full | gene | 同 1A/3A | k-mer/dm3loc/RNA-FM/DNABERT-2 | 复用对应 split |

- **1A/1B 复用 gene-level split**：split 按 `(species,gene)` 键，isoform 每条落到该基因同侧 → 可比、无泄漏。
- **Track 2/3 只 gene**；**Track 3 不含 mRNA-FM/UTR-BERT**(codon 无阅读框、UTR-BERT 专 3'UTR)。
- **fine = 5 隔室多标签**(Cell_body/Dendrite/Neuropil/Axon/Neurite)，必须 `--label-scheme fine --source-mask`，
  作为二分类头条的补充视角。
- **`--keep-assay-labels`**:额外把 **Ribosome/Cytoplasm** 纳入 fine 目标类。它们是**翻译/分馏轴**(非解剖
  定位轴)——Ribosome 序列可预测性强(实测 ROC≈0.86),Cytoplasm 弱(ROC≈0.58)。走 source-mask(只有做过
  该 assay 的源贡献负样本)。**单独成节报告**,不要混进定位隔室的 macro,以免糊掉两条不同的生物学轴。
- **full vs 3'UTR 结论**：Track 3 vs Track 1(趋势) + 消融在同基因交集严格比(claim)。两走向都可发表。

---

## 4. 区域处理：两阶段

阶段 1 得到区域(`apply_region` 按 seqtype 分流)；阶段 2 喂模型(tokenize/截断/滑窗/pooling 完全一致)。
到达 3'UTR 的方式不同(提取 vs 原生)，到达后喂模型方式相同。

---

## 5. isoform 杀招分析（主图候选，必须控长度）

~625 配对基因(Ciolli 580)。命题="扣长度后还分不分得对"：
1. 配对内评估(每基因 soma/neurite 两条 3'UTR 排序)；
2. **必加长度基线**(Ciolli=90.2%)，FM/k-mer/专用必须显著超过；
3. 长度残差化 / 长度匹配子集；
4. 分源报告 + Tushev 反衬(278 多异构体仅 13 对立=干净生物结论)。
（脚本 `analyze_isoform_discordance.py` 待实现。）

---

## 6. 评估与诚实声明

主指标 macro ROC-AUC / macro PR-AUC / PR-lift + all_zero/prior 基线(自动产)；
isoform-level 报 effective N = #group；模型间差异在 group 层 bootstrap；human n 太小不做泛化声称。

---

## 7. 执行

`scripts/run_all.sh [splits|track1a|track1b|track2|track3|ablation|fine|all]`。
代码：`scripts/train_rnafm_multilabel.py`(区域分流 + 冻结 split + DNABERT-2 同脚本 + `--arch`)
和 `scripts/task_specific_models.py`(RNATracker/DM3Loc)，均已单元/烟雾测试。

> 数据放置：6 个 isoform CSV 与 bulk CSV 放进**同一** `--input-dir`，Track 1 自动合并
> (bulk 提 3'UTR + isoform 原生通过)。

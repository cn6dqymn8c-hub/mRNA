# mRNALocator-imb: An Imbalance-Tolerant Ensemble Learning Framework for mRNA Subcellular Localization
# mRNALocator-imb：一种用于mRNA亚细胞定位的不平衡容忍集成学习框架

**Source:** E:\mRNA\Doc\mRNALocator-imb.pdf  
**Journal:** IET Systems Biology  
**Manuscript ID:** SYB-2026-02-0033  
**Authors:** Jinbo Hu, Haibin Liu, Hao Wu  

---

## Abstract / 摘要

<a id="S001"></a>
**Source:** p.1 S001

**Original:** The subcellular localization of mRNAs determines the spatial site of protein translation and is essential for the precise regulation of protein function. However, existing computational methods for predicting eukaryotic mRNA localization remain limited by insufficient utilization of sequence information and poor performance on highly imbalanced datasets. To overcome these limitations, we propose mRNALocator-imb, an imbalance-tolerant ensemble learning framework for mRNA subcellular localization. The model integrates physicochemical pattern features with distributed nucleic acid representations and employs a hybrid architecture that combines Random Forest and Gated Recurrent Unit (GRU) networks. To explicitly address class imbalance, the Labeled Distribution-Aware Margin (LDAM) loss is employed during GRU training, while adaptive synthetic sampling (ADASYN) is applied to balance the data for Random Forest learning. Experimental results demonstrate that mRNALocator-imb consistently outperforms conventional machine learning approaches and existing predictors, particularly in imbalanced classification settings. Overall, this work presents a robust and generalizable framework for sequence-based localization prediction, with potential applicability to a wide range of bioinformatics tasks.

**中文:** mRNA的亚细胞定位决定了蛋白质翻译的空间位点，对于蛋白质功能的精确调控至关重要。然而，现有的预测真核mRNA定位的计算方法仍受到序列信息利用不足和在高度不平衡数据集上性能不佳的限制。为了克服这些限制，我们提出了mRNALocator-imb，一种用于mRNA亚细胞定位的不平衡容忍集成学习框架。该模型整合了物理化学模式特征与分布式核酸表示，并采用结合随机森林和门控循环单元（GRU）网络的混合架构。为了明确解决类别不平衡问题，在GRU训练期间采用标签分布感知边界（LDAM）损失，而自适应合成采样（ADASYN）应用于平衡随机森林学习的数据。实验结果表明，mRNALocator-imb始终优于传统机器学习方法和现有预测器，特别是在不平衡分类设置中。总体而言，这项工作为基于序列的定位预测提供了一个鲁棒且可泛化的框架，具有在广泛的生物信息学任务中的潜在适用性。

**Keywords:** mRNA Subcellular Localization, Imbalanced Data Set, Deep Learning Framework, Labeled Distribution-Aware Margin, Adaptive Synthetic Sampling  
**关键词:** mRNA亚细胞定位，不平衡数据集，深度学习框架，标签分布感知边界，自适应合成采样

---

## Introduction / 引言

<a id="S002"></a>
**Source:** p.2 S002

**Original:** Messenger RNA (mRNA) is a single-stranded RNA molecule that conveys genetic information from DNA to the ribosome, where it serves as the template for protein synthesis. In eukaryotic cells, mRNAs are not randomly distributed; instead, their intracellular localization follows highly regulated and selective programs. Such spatial organization enables mRNAs to be translated at specific subcellular sites, which is essential for precise temporal and spatial control of protein production and for appropriate cellular responses to internal and external stimuli [1–3].

**中文:** 信使RNA（mRNA）是一种单链RNA分子，将遗传信息从DNA传递到核糖体，在那里它作为蛋白质合成的模板。在真核细胞中，mRNA并非随机分布；相反，它们的细胞内定位遵循高度调控和选择性程序。这种空间组织使mRNA能够在特定的亚细胞位点翻译，这对于蛋白质产生的精确时空控制以及对内外刺激的适当细胞反应至关重要[1-3]。

<a id="S003"></a>
**Source:** p.2 S003

**Original:** Traditionally, mRNA localization and translational regulation were considered specialized mechanisms affecting only a limited subset of transcripts, primarily to restrict gene expression to polarized or asymmetric regions within the cell [4]. However, accumulating evidence has demonstrated that mRNA subcellular localization is a widespread regulatory phenomenon implicated in diverse biological processes, including cell polarity, cell migration, embryonic development, and asymmetric cell division, as well as in the pathogenesis of complex diseases such as Alzheimer's disease and cancer [5–8]. Therefore, a comprehensive understanding of mRNA subcellular localization mechanisms is crucial for elucidating intracellular information flow, uncovering disease-related molecular mechanisms, and facilitating the development of novel therapeutic strategies.

**中文:** 传统上，mRNA定位和翻译调控被认为是仅影响有限转录本子集的特殊机制，主要为了将基因表达限制在细胞内的极化或不对称区域[4]。然而，越来越多的证据表明，mRNA亚细胞定位是一个广泛存在的调控现象，涉及多种生物过程，包括细胞极性、细胞迁移、胚胎发育和非对称细胞分裂，以及阿尔茨海默病和癌症等复杂疾病的发病机制[5-8]。因此，全面理解mRNA亚细胞定位机制对于阐明细胞内信息流、揭示疾病相关分子机制以及促进新治疗策略的开发至关重要。

<a id="S004"></a>
**Source:** p.3 S004

**Original:** Early studies of mRNA subcellular localization primarily relied on experimental techniques. Methods such as RNA fluorescence in situ hybridization (RNA-FISH) [9], CRISPR LiveFISH [10], and INSIGHT [11] enabled direct visualization and localization of RNA molecules within cells. With the rapid advancement of high-throughput sequencing technologies, RNA-Seq has provided comprehensive transcriptomic profiles, greatly expanding our understanding of RNA function and spatial organization. In particular, sequencing-based approaches, including subRNA-seq [12], FISSEQ [13], and APEX-seq [14], have enabled large-scale characterization of RNA subcellular localization.

**中文:** 早期mRNA亚细胞定位研究主要依赖于实验技术。RNA荧光原位杂交（RNA-FISH）[9]、CRISPR LiveFISH[10]和INSIGHT[11]等方法使得能够直接可视化和定位细胞内的RNA分子。随着高通量测序技术的快速发展，RNA-Seq提供了全面的转录组谱，大大扩展了我们对RNA功能和空间组织的理解。特别是，基于测序的方法，包括subRNA-seq[12]、FISSEQ[13]和APEX-seq[14]，已经实现了RNA亚细胞定位的大规模表征。

<a id="S005"></a>
**Source:** p.3 S005

**Original:** These experimental approaches have generated extensive datasets describing RNA spatial distribution, offering valuable resources for elucidating RNA functions across different cellular compartments. However, wet-lab methods for RNA localization analysis suffer from several inherent limitations. They typically require specialized instrumentation and technical expertise, involve complex and labor-intensive experimental procedures, and incur substantial costs due to expensive reagents, consumables, and time investment. These constraints limit their scalability and widespread application in large-scale studies.

**中文:** 这些实验方法产生了描述RNA空间分布的大量数据集，为阐明不同细胞区室中RNA功能提供了宝贵资源。然而，RNA定位分析的湿实验方法存在几个固有的局限性。它们通常需要专门的仪器和技术专业知识，涉及复杂且劳动密集的实验程序，并且由于昂贵的试剂、耗材和时间投入而产生大量成本。这些限制限制了它们在大规模研究中的可扩展性和广泛应用。

<a id="S006"></a>
**Source:** p.4 S006

**Original:** Consequently, despite their indispensable role in uncovering RNA localization mechanisms, experimental approaches alone are insufficient to meet the growing demand for efficient and scalable analysis. This has made the accurate prediction of RNA subcellular localization using bioinformatics and computational methods a critical and challenging research problem.

**中文:** 因此，尽管实验方法在揭示RNA定位机制方面发挥着不可或缺的作用，但仅靠实验方法无法满足对高效和可扩展分析日益增长的需求。这使得使用生物信息学和计算方法准确预测RNA亚细胞定位成为一个关键且具有挑战性的研究问题。

<a id="S007"></a>
**Source:** p.4 S007

**Original:** In recent years, advances in bioinformatics have led to significant progress in the study of RNA subcellular localization. By integrating computational analysis with machine learning techniques, bioinformatics approaches effectively alleviate the time-consuming and cost-intensive limitations of traditional experimental methods, while enabling the extraction of biologically meaningful patterns from large-scale datasets to predict RNA localization and function.

**中文:** 近年来，生物信息学的进展在RNA亚细胞定位研究中取得了重大进展。通过将计算分析与机器学习技术相结合，生物信息学方法有效地缓解了传统实验方法耗时和成本密集的限制，同时能够从大规模数据集中提取生物学上有意义的模式来预测RNA定位和功能。

<a id="S008"></a>
**Source:** p.4 S008

**Original:** RNATracker [15], the first computational predictor for mRNA subcellular localization, employs convolutional neural networks in combination with a bidirectional long short-term memory (BiLSTM) architecture and a self-attention mechanism to enhance predictive accuracy and efficiency. Subsequently, Zhang et al. proposed iLoc-mRNA [16], which adopts a support vector machine (SVM) classifier to provide an alternative prediction strategy. However, both RNATracker and iLoc-mRNA are primarily restricted to human mRNA localization prediction. To enable broader applicability across species, Garg et al. developed mRNALoc [17], a eukaryote-oriented predictor that extracts sequence features using pseudo K-tuple nucleotide composition (PseKNC) and constructs classification models based on support vector machines.

**中文:** RNATracker[15]是第一个用于mRNA亚细胞定位的计算预测器，采用卷积神经网络结合双向长短期记忆（BiLSTM）架构和自注意力机制来提高预测准确性和效率。随后，Zhang等人提出了iLoc-mRNA[16]，采用支持向量机（SVM）分类器提供替代预测策略。然而，RNATracker和iLoc-mRNA都主要局限于人类mRNA定位预测。为了实现跨物种的更广泛应用，Garg等人开发了mRNALoc[17]，这是一个面向真核生物的预测器，使用伪K元核苷酸组成（PseKNC）提取序列特征并基于支持向量机构建分类模型。

<a id="S009"></a>
**Source:** p.4 S009

**Original:** More recently, Li et al. introduced SubLocEP [18], a LightGBM-based framework [19] that represents the current state-of-the-art in mRNA subcellular localization prediction. By combining gradient boosting decision trees with optimized feature engineering strategies, SubLocEP achieves improved prediction accuracy and generalization performance, providing a robust and effective tool for large-scale mRNA localization analysis.

**中文:** 最近，Li等人介绍了SubLocEP[18]，这是一个基于LightGBM的框架[19]，代表了当前mRNA亚细胞定位预测的最先进水平。通过将梯度提升决策树与优化的特征工程策略相结合，SubLocEP实现了改进的预测准确性和泛化性能，为大规模mRNA定位分析提供了一个鲁棒且有效的工具。

<a id="S010"></a>
**Source:** p.4 S010

**Original:** Although existing mRNA subcellular localization prediction models have achieved encouraging performance, several challenges remain. First, predictive accuracy can be further improved. Second, most current algorithms exhibit limited robustness when handling highly imbalanced datasets. Third, many studies focus primarily on predictive performance while lacking in-depth biological interpretation of the learned features. To address the issues of prediction accuracy and data imbalance, this study employs a rigorous feature selection strategy that integrates multiple complementary feature extraction methods to comprehensively characterize mRNA sequences. In addition, advanced machine learning and deep learning models are systematically investigated in conjunction with imbalance-aware optimization strategies to construct an accurate and robust prediction framework. To overcome the limited biological interpretability of existing approaches, we further incorporate multiple feature importance and interpretability analysis techniques to identify key sequence-derived biomarkers and elucidate their biological relevance from both the deep learning model perspective and the mRNA sequence level.

**中文:** 尽管现有的mRNA亚细胞定位预测模型取得了令人鼓舞的性能，但仍存在几个挑战。首先，预测准确性可以进一步提高。其次，大多数当前算法在处理高度不平衡数据集时表现出有限的鲁棒性。第三，许多研究主要关注预测性能，而缺乏对学习特征的深入生物学解释。为了解决预测准确性和数据不平衡问题，本研究采用了严格的特征选择策略，整合多种互补的特征提取方法来全面表征mRNA序列。此外，系统地研究了先进的机器学习和深度学习模型，并结合不平衡感知优化策略来构建准确且鲁棒的预测框架。为了克服现有方法的有限生物学可解释性，我们进一步整合了多种特征重要性和可解释性分析技术，以识别关键的序列衍生生物标志物，并从深度学习模型角度和mRNA序列水平阐明其生物学相关性。

<a id="S011"></a>
**Source:** p.5 S011

**Original:** By jointly improving predictive performance, imbalance tolerance, and biological interpretability, this work provides deeper insights into the functional mechanisms underlying mRNA subcellular localization. The proposed framework not only facilitates a more comprehensive understanding of mRNA localization and function but also offers valuable tools and biological insights for future bioinformatics and biomedical research.

**中文:** 通过共同改善预测性能、不平衡容忍度和生物学可解释性，这项工作为mRNA亚细胞定位背后的功能机制提供了更深入的见解。提出的框架不仅促进了对mRNA定位和功能的更全面理解，还为未来的生物信息学和生物医学研究提供了有价值的工具和生物学见解。

---

## Materials and Methods / 材料与方法

<a id="S012"></a>
**Source:** p.5 S012

**Original:** The dataset used in this study was obtained from mRNALoc, a curated database of mRNA subcellular localization information [17]. In total, 14,909 mRNA sequences were collected, distributed across five subcellular compartments: cytoplasm (6,376), endoplasmic reticulum (1,426), extracellular region (855), mitochondria (421), and nucleus (5,831). For model development and evaluation, the dataset was partitioned into a training set, a weight set, and an independent test set using a stratified sampling strategy at an approximate ratio of 5:1:1, ensuring consistent class distributions across subsets. The detailed data splitting procedure and resulting sample distributions are illustrated in Figure 1.

**中文:** 本研究使用的数据集来自mRNALoc，这是一个经过整理的mRNA亚细胞定位信息数据库[17]。总共收集了14,909个mRNA序列，分布在五个亚细胞区室中：细胞质（6,376）、内质网（1,426）、细胞外区域（855）、线粒体（421）和细胞核（5,831）。为了模型开发和评估，使用分层抽样策略将数据集划分为训练集、权重集和独立测试集，比例约为5:1:1，确保各子集间类别分布一致。详细的数据分割过程和结果样本分布在图1中说明。

<a id="F001"></a>
### Fig. 1. Subcellular location distribution of mRNA sequences in the datasets
### 图1. 数据集中mRNA序列的亚细胞定位分布

**Placed near:** p.5 S012  
**Source:** p.6 C001  

![Fig. 1](assets/fig_page6_img1.jpeg)

**Original caption:** Subcellular location distribution of mRNA sequences in the datasets  
**中文图注:** 数据集中mRNA序列的亚细胞定位分布

**Reading note:** 此图展示了五个亚细胞区室中mRNA序列的分布情况，显示了数据集的类别不平衡特征。

<a id="S013"></a>
**Source:** p.6 S013

**Original:** Previous studies have demonstrated that feature fusion strategies can substantially improve model performance in sequence-based prediction tasks [20–22]. In this study, five complementary feature encoding schemes, including NMBACC, TPCP, DACC, MMI, and Word2Vec, are adopted to characterize mRNA sequences. Specifically, NMBACC, TPCP, DACC, and MMI features are used as inputs to the Random Forest classifier, while Word2Vec embeddings are employed to represent sequence information for the GRU-based model. Detailed descriptions, computational procedures, and biological interpretations of these feature encoding schemes are provided in Supplementary Text S1.

**中文:** 先前的研究表明，特征融合策略可以显著提高基于序列的预测任务中的模型性能[20-22]。在本研究中，采用了五种互补的特征编码方案，包括NMBACC、TPCP、DACC、MMI和Word2Vec来表征mRNA序列。具体而言，NMBACC、TPCP、DACC和MMI特征用作随机森林分类器的输入，而Word2Vec嵌入用于表示基于GRU模型的序列信息。这些特征编码方案的详细描述、计算程序和生物学解释在补充文本S1中提供。

<a id="S014"></a>
**Source:** p.6 S014

**Original:** As illustrated in Figure 2, mRNALocator-imb is a deep ensemble learning framework specifically designed to predict mRNA subcellular localization directly from nucleotide sequences using a multi-scale feature fusion strategy. The proposed framework performs an end-to-end mapping from raw mRNA sequences to high-confidence subcellular localization predictions, thereby eliminating the need for manual post-processing of intermediate representations.

**中文:** 如图2所示，mRNALocator-imb是一个深度集成学习框架，专门设计用于使用多尺度特征融合策略直接从核苷酸序列预测mRNA亚细胞定位。提出的框架执行从原始mRNA序列到高置信度亚细胞定位预测的端到端映射，从而消除了对中间表示进行手动后处理的需要。

<a id="F002"></a>
### Fig. 2. Overview of the mRNALocator-imb framework
### 图2. mRNALocator-imb框架概述

**Placed near:** p.6 S014  
**Source:** p.7 C002  

![Fig. 2](assets/fig_page7_img1.jpeg)

**Original caption:** Overview of the mRNALocator-imb framework  
**中文图注:** mRNALocator-imb框架概述

**Reading note:** 此图展示了mRNALocator-imb的整体架构，包括特征编码、模型训练和集成预测的流程。

<a id="S015"></a>
**Source:** p.7 S015

**Original:** Dataset processing begins with raw mRNA sequences in FASTA format, which are fed into two parallel feature encoding pipelines to overcome the limitations of single-modal sequence representation. In the handcrafted feature branch, each input sequence is transformed into four numerical feature vectors with explicit biological relevance, including NMBAC, TPCP, MMI, and DACC, followed by feature fusion to support subsequent learning by base classifiers. These handcrafted descriptors enable interpretable, white-box quantification of mRNA subcellular localization by capturing diverse physicochemical and statistical properties of nucleotide sequences. In particular, the extracted features characterize informative sequence components and motifs, such as guanine (G) nucleotides, dinucleotide patterns (e.g., GG), cytosine (C) nucleotides, and trinucleotide motifs (e.g., CCT and CCC), among others.

**中文:** 数据处理从FASTA格式的原始mRNA序列开始，这些序列被输入到两个并行的特征编码管道中以克服单模态序列表示的局限性。在手工特征分支中，每个输入序列被转换为四个具有明确生物学相关性的数值特征向量，包括NMBAC、TPCP、MMI和DACC，然后进行特征融合以支持基分类器的后续学习。这些手工描述符通过捕获核苷酸序列的多种物理化学和统计性质，实现了对mRNA亚细胞定位的可解释、白箱量化。特别是，提取的特征表征了信息丰富的序列组件和基序，如鸟嘌呤（G）核苷酸、二核苷酸模式（如GG）、胞嘧啶（C）核苷酸和三核苷酸基序（如CCT和CCC）等。

<a id="S016"></a>
**Source:** p.7 S016

**Original:** In parallel, the sequence embedding branch employs the Word2Vec algorithm to encode mRNA sequences into high-dimensional dense vectors, thereby modeling the contextual dependencies of nucleotides and the co-occurrence patterns of functional sequence motifs. By jointly integrating biologically interpretable handcrafted features with data-driven distributed representations, the proposed dual-pipeline architecture provides a comprehensive and complementary characterization of mRNA sequences, effectively mitigating the representational limitations of single-feature encoding schemes.

**中文:** 与此同时，序列嵌入分支采用Word2Vec算法将mRNA序列编码为高维密集向量，从而对核苷酸的上下文依赖性和功能序列基序的共现模式进行建模。通过联合整合具有生物学可解释性的手工特征与数据驱动的分布式表示，提出的双管道架构提供了对mRNA序列的全面且互补的表征，有效缓解了单特征编码方案的表示局限性。

<a id="S017"></a>
**Source:** p.7 S017

**Original:** To address the severe class imbalance inherent in mRNA subcellular localization datasets, imbalance mitigation strategies are applied to both handcrafted and sequence-derived features prior to model training. Specifically, Label-Distribution-Aware Margin (LDAM) loss is incorporated during the training of the GRU network, while Adaptive Synthetic Sampling (ADASYN) is employed to rebalance the handcrafted feature space used by the Random Forest classifier. These complementary strategies effectively alleviate the negative impact of class imbalance on model optimization.

**中文:** 为了解决mRNA亚细胞定位数据集中固有的严重类别不平衡问题，在模型训练之前对手工特征和序列衍生特征都应用了不平衡缓解策略。具体而言，在GRU网络训练期间引入标签分布感知边界（LDAM）损失，而采用自适应合成采样（ADASYN）来重新平衡随机森林分类器使用的手工特征空间。这些互补策略有效缓解了类别不平衡对模型优化的负面影响。

<a id="S018"></a>
**Source:** p.8 S018

**Original:** In the final decision stage, mRNALocator-imb adopts a heterogeneous ensemble learning strategy to integrate predictions from multiple base learners. To leverage the robustness of traditional machine learning in few-shot and noise-resistant settings, an independent Random Forest (RF) classifier is trained on the fused handcrafted features, producing a set of class probability outputs denoted as P_RF. In parallel, to capture long-range sequential dependencies in mRNA sequences, a Gated Recurrent Unit (GRU) network with early stopping is trained on the high-dimensional sequence embeddings, yielding an additional set of prediction probabilities P_GRU.

**中文:** 在最终决策阶段，mRNALocator-imb采用异构集成学习策略来整合来自多个基学习器的预测。为了利用传统机器学习在少样本和抗噪声设置中的鲁棒性，在融合的手工特征上训练独立的随机森林（RF）分类器，产生一组表示为P_RF的类别概率输出。与此同时，为了捕获mRNA序列中的长程序列依赖性，在高维序列嵌入上训练具有早停机制的门控循环单元（GRU）网络，产生另一组预测概率P_GRU。

<a id="S019"></a>
**Source:** p.8 S019

**Original:** The final prediction of the mRNALocator-imb framework is produced by a weighted ensemble module that adaptively integrates the outputs of the Random Forest classifier and the GRU network via learnable weighting parameters. This end-to-end ensemble architecture, a spanning multi-modal feature encoding at the representation level and heterogeneous model fusion at the decision level, enables the framework to achieve high predictive accuracy and strong generalization across diverse mRNA subcellular localization categories.

**中文:** mRNALocator-imb框架的最终预测由加权集成模块产生，该模块通过可学习的权重参数自适应地整合随机森林分类器和GRU网络的输出。这种端到端集成架构，在表示层跨越多模态特征编码，在决策层实现异构模型融合，使框架能够在不同的mRNA亚细胞定位类别中实现高预测准确性和强泛化能力。

<a id="S020"></a>
**Source:** p.8 S020

**Original:** Random Forests have been widely adopted in bioinformatics applications [23,24] and have demonstrated strong predictive performance in sequence-based classification tasks using handcrafted biological features [25]. Accordingly, this study constructs a Random Forest classifier based on the handcrafted mRNA sequence features described in the preceding subsection.

**中文:** 随机森林在生物信息学应用中被广泛采用[23,24]，并在使用手工生物学特征的基于序列的分类任务中表现出强大的预测性能[25]。因此，本研究基于前一小节描述的手工mRNA序列特征构建随机森林分类器。

<a id="S021"></a>
**Source:** p.8 S021

**Original:** However, models relying solely on handcrafted biological features inevitably lose explicit sequence-order information inherent to mRNA sequences. To address this limitation and effectively capture contextual dependencies, we incorporate distributed sequence representations learned via sequence embedding techniques. Specifically, the Word2Vec framework is employed to generate dense vector representations of mRNA sequences. Originally developed for natural language processing, Word2Vec has been successfully applied to biological sequence analysis, including proteins [26] and non-coding RNAs [27].

**中文:** 然而，仅依赖手工生物学特征的模型不可避免地会丢失mRNA序列固有的明确序列顺序信息。为了解决这一限制并有效捕获上下文依赖性，我们整合了通过序列嵌入技术学习的分布式序列表示。具体而言，采用Word2Vec框架生成mRNA序列的密集向量表示。Word2Vec最初是为自然语言处理开发的，已成功应用于生物序列分析，包括蛋白质[26]和非编码RNA[27]。

<a id="S022"></a>
**Source:** p.9 S022

**Original:** To construct the embedding corpus, all mRNA sequences in the training set are segmented into overlapping "sentences" composed of trinucleotides using a sliding window of size 3 and a step size of 1. The resulting corpus is then used to train word embeddings with the skip-gram model of Word2Vec. To further model long-range contextual dependencies within mRNA sequences, the learned trinucleotide embeddings are subsequently processed by a Gated Recurrent Unit (GRU) network, enabling effective utilization of sequential information for downstream classification.

**中文:** 为了构建嵌入语料库，训练集中的所有mRNA序列使用大小为3、步长为1的滑动窗口分割成由三核苷酸组成的重叠"句子"。然后使用Word2Vec的skip-gram模型在结果语料库上训练词嵌入。为了进一步建模mRNA序列内的长程上下文依赖性，学习到的三核苷酸嵌入随后由门控循环单元（GRU）网络处理，实现序列信息在下游分类中的有效利用。

<a id="S023"></a>
**Source:** p.9 S023

**Original:** The mRNALocator-imb framework integrates the Random Forest and GRU models using a weighted averaging ensemble strategy. To prevent information leakage and ensure unbiased weight estimation, an independent weight set is reserved exclusively for learning the ensemble weights.

**中文:** mRNALocator-imb框架使用加权平均集成策略整合随机森林和GRU模型。为了防止信息泄漏并确保无偏权重估计，保留独立的权重集专门用于学习集成权重。

<a id="S024"></a>
**Source:** p.9 S024

**Original:** Specifically, the Random Forest and GRU models are first trained independently using the training dataset. During GRU training, the original training set is further partitioned into a training subset and a validation subset at a ratio of 5:1 to mitigate overfitting. A ten-fold cross-validation scheme combined with an early-stopping mechanism is employed, allowing training to terminate automatically when the validation loss ceases to decrease.

**中文:** 具体而言，随机森林和GRU模型首先使用训练数据集独立训练。在GRU训练期间，原始训练集以5:1的比例进一步划分为训练子集和验证子集以减轻过拟合。采用十折交叉验证方案结合早停机制，允许在验证损失停止下降时自动终止训练。

<a id="S025"></a>
**Source:** p.9 S025

**Original:** After base model training, the validation (weight) set is used to optimize the ensemble weighting parameters within the integrated learning framework. This procedure enables adaptive calibration of the contributions from the Random Forest and GRU models, thereby enhancing the overall predictive performance of the ensemble.

**中文:** 在基模型训练后，验证（权重）集用于优化集成学习框架内的集成权重参数。该程序实现了对随机森林和GRU模型贡献的自适应校准，从而提高集成的整体预测性能。

<a id="S026"></a>
**Source:** p.9 S026

**Original:** Machine learning methods have been increasingly applied in bioinformatics; however, their performance is often challenged by the prevalence of imbalanced data distributions. Most learning algorithms implicitly assume relatively balanced class proportions, and their predictive accuracy and reliability can degrade substantially when this assumption is violated. In highly imbalanced settings, minority classes are frequently underrepresented during training and may be treated as noise, leading to biased models that favor majority classes. This bias typically manifests as inflated false-positive rates and reduced true-positive rates for minority categories.

**中文:** 机器学习方法在生物信息学中得到了越来越多的应用；然而，其性能经常受到不平衡数据分布普遍存在的挑战。大多数学习算法隐含地假设相对平衡的类别比例，当这一假设被违反时，其预测准确性和可靠性会大幅下降。在高度不平衡的设置中，少数类在训练期间经常代表不足，可能被视为噪声，导致偏向多数类的有偏模型。这种偏见通常表现为少数类的假阳性率升高和真阳性率降低。

<a id="S027"></a>
**Source:** p.10 S027

**Original:** Data imbalance is particularly pronounced in mRNA subcellular localization datasets, where the number of samples associated with different cellular compartments varies considerably. Such class imbalance can severely impair model performance, especially for subcellular localizations with limited training samples, and may even prevent effective prediction altogether. This challenge is widely recognized in machine learning and bioinformatics research, motivating the development of specialized strategies to enhance model robustness and predictive performance under imbalanced data conditions.

**中文:** 数据不平衡在mRNA亚细胞定位数据集中尤为明显，其中与不同细胞区室相关的样本数量差异很大。这种类别不平衡会严重损害模型性能，特别是对于训练样本有限的亚细胞定位，甚至可能完全阻止有效预测。这一挑战在机器学习和生物信息学研究中被广泛认识，促使开发专门策略以提高不平衡数据条件下的模型鲁棒性和预测性能。

<a id="S028"></a>
**Source:** p.10 S028

**Original:** In traditional machine learning, data-level resampling techniques are commonly employed to address class imbalance, including oversampling and undersampling strategies. Oversampling aims to increase the representation of minority classes by replicating existing samples or generating synthetic samples, such as through the Synthetic Minority Over-sampling Technique (SMOTE) [28], thereby promoting a more balanced class distribution. In contrast, undersampling reduces the number of majority-class samples, either through random removal or via clustering-based selection methods.

**中文:** 在传统机器学习中，数据级重采样技术通常用于解决类别不平衡，包括过采样和欠采样策略。过采样旨在通过复制现有样本或生成合成样本来增加少数类的表示，例如通过合成少数过采样技术（SMOTE）[28]，从而促进更平衡的类别分布。相比之下，欠采样通过随机删除或基于聚类的选择方法减少多数类样本的数量。

<a id="S029"></a>
**Source:** p.10 S029

**Original:** Although both approaches can mitigate class imbalance, they present inherent trade-offs. Oversampling may increase the risk of overfitting by introducing redundant or highly similar samples, while undersampling can lead to information loss and hinder the model's ability to fully capture the characteristics of the majority class.

**中文:** 虽然这两种方法都可以缓解类别不平衡，但它们存在固有的权衡。过采样可能通过引入冗余或高度相似的样本增加过拟合风险，而欠采样可能导致信息丢失并阻碍模型完全捕获多数类特征的能力。

<a id="S030"></a>
**Source:** p.10 S030

**Original:** In deep learning, class imbalance is commonly addressed through the design of customized loss functions that explicitly account for uneven class distributions. These loss functions are formulated to target challenges arising during imbalanced training, such as improving the recognition of minority-class samples and enhancing the model's sensitivity to hard-to-classify instances. By assigning class-dependent weights or margins, customized loss functions effectively rebalance the optimization objective. As a result, models trained with such loss functions impose higher penalties on misclassification of minority classes, encouraging the network to allocate greater attention to these underrepresented categories. Importantly, this approach improves predictive performance on minority classes without altering the original data distribution.

**中文:** 在深度学习中，类别不平衡通常通过设计明确考虑不均匀类别分布的定制损失函数来解决。这些损失函数的制定旨在解决不平衡训练期间出现的挑战，例如提高少数类样本的识别能力和增强模型对难以分类实例的敏感性。通过分配类别相关的权重或边界，定制损失函数有效地重新平衡优化目标。因此，使用此类损失函数训练的模型对少数类的错误分类施加更高的惩罚，鼓励网络对这些代表不足的类别给予更多关注。重要的是，这种方法在不改变原始数据分布的情况下提高了少数类的预测性能。

<a id="S031"></a>
**Source:** p.11 S031

**Original:** In the integrated framework proposed in this study, distinct imbalance-handling strategies are applied to different sub-models to enhance overall predictive performance. For the Random Forest sub-model, we adopt Adaptive Synthetic Sampling (ADASYN) to address class imbalance at the data level. ADASYN generates synthetic minority-class samples in proportion to their learning difficulty, producing more samples for hard-to-classify instances and fewer for relatively easy ones. This adaptive sampling strategy effectively reduces classification bias caused by imbalanced class distributions and enables the decision boundary to better accommodate minority classes, thereby improving classification accuracy.

**中文:** 在本研究提出的集成框架中，对不同的子模型应用不同的不平衡处理策略以提高整体预测性能。对于随机森林子模型，我们采用自适应合成采样（ADASYN）在数据级别解决类别不平衡。ADASYN根据学习难度按比例生成合成少数类样本，为难以分类的实例生成更多样本，为相对容易的实例生成较少样本。这种自适应采样策略有效减少了不平衡类别分布引起的分类偏差，使决策边界能够更好地适应少数类，从而提高分类准确性。

<a id="S032"></a>
**Source:** p.11 S032

**Original:** For the Gated Recurrent Unit (GRU) sub-model, class imbalance is addressed at the optimization level by incorporating the Label-Distribution-Aware Margin (LDAM) loss function during training. LDAM introduces class-dependent margins that assign larger decision margins to minority classes, encouraging the model to place greater emphasis on underrepresented categories and resulting in more balanced predictive performance. The margin for each class is controlled by a tunable hyperparameter, allowing flexible adjustment of the imbalance-aware optimization process.

**中文:** 对于门控循环单元（GRU）子模型，通过在训练期间引入标签分布感知边界（LDAM）损失函数在优化级别解决类别不平衡。LDAM引入类别相关边界，为少数类分配更大的决策边界，鼓励模型更加重视代表不足的类别，从而产生更平衡的预测性能。每个类的边界由可调超参数控制，允许灵活调整不平衡感知优化过程。

<a id="S033"></a>
**Source:** p.11 S033

**Original:** Formally, the LDAM loss is defined as: ℒLDAM((x,y);f) = -log [exp(z_y-Δ_y) / (exp(z_y-Δ_y) + Σ_{j≠y} exp(z_j))] (1), where (x,y) denotes a training sample with label y, f represents the GRU model, z_y is the logit corresponding to the ground-truth class, and z_j denotes the logit of class j. The class-dependent margin Δ_y is defined as: Δ_y = C * (n_y)^(-1/4), y∈{1,⋯,K} (2), where n_y is the number of training samples in class y, K is the total number of classes, and C is a hyperparameter that controls the overall scale of the margins.

**中文:** 形式上，LDAM损失定义为：ℒLDAM((x,y);f) = -log [exp(z_y-Δ_y) / (exp(z_y-Δ_y) + Σ_{j≠y} exp(z_j))] (1)，其中(x,y)表示标签为y的训练样本，f表示GRU模型，z_y是对应于真实类别的logit，z_j表示类别j的logit。类别相关边界Δ_y定义为：Δ_y = C * (n_y)^(-1/4), y∈{1,⋯,K} (2)，其中n_y是类别y中的训练样本数，K是类别总数，C是控制边界整体尺度的超参数。

<a id="S034"></a>
**Source:** p.10 S034

**Original:** Suitable hyperparameter selection is critical for enhancing model performance, accelerating training convergence, and mitigating the risks of overfitting or underfitting. In this study, we employed a grid search strategy combined with ten-fold cross-validation on the training set. This procedure reduces evaluation bias by iteratively training and validating the model on different data subsets, ensuring that each sample is used for both training and validation. The Macro F-measure was adopted as the optimization criterion to promote balanced performance across classes. We focused on tuning hyperparameters with a substantial impact on model effectiveness, including the lag value used in biometric feature extraction, the window size of the Word2Vec context, and the learning rate. The optimal hyperparameter settings are summarized in Table 1.

**中文:** 合适的超参数选择对于提高模型性能、加速训练收敛以及缓解过拟合或欠拟合的风险至关重要。在本研究中，我们采用网格搜索策略结合训练集上的十折交叉验证。该程序通过在不同数据子集上迭代训练和验证模型来减少评估偏差，确保每个样本都用于训练和验证。采用宏F度量作为优化标准以促进跨类别的平衡性能。我们专注于调优对模型有效性有重大影响的超参数，包括生物特征提取中使用的滞后值、Word2Vec上下文的窗口大小以及学习率。最佳超参数设置总结在表1中。

<a id="T001"></a>
### Table 1. Hyperparameter optimization results
### 表1. 超参数优化结果

**Placed near:** p.10 S034  
**Source:** p.12 C003  

| Hyperparameters | Search range | Step length | Optimal value |
|---|---|---|---|
| Number of trees in a random forest | [100,1000] | 100 | 500 |
| Lagged values of NMBAC | [1, 6] | 1 | 2 |
| Lag values for DACC | [1, 6] | 1 | 1 |
| Window size of Word2Vec splitter | [2, 6] | 1 | 3 |
| Learning rate | 0.1, 0.01, 0.001, 0.0001,0.00001, 0.000001 | - | 0.00001 |

**Original caption:** Hyperparameter optimization results  
**中文图注:** 超参数优化结果

**Reading note:** 此表展示了通过网格搜索和十折交叉验证确定的最佳超参数设置。

---

## Results / 结果

<a id="S035"></a>
**Source:** p.11 S035

**Original:** Ten-fold cross-validation is a widely adopted strategy in machine learning and statistical modeling for evaluating a model's generalization performance, stability, and reliability. In this approach, the training dataset is partitioned into ten equal subsets; in each iteration, nine subsets are used for model training, and the remaining subset is reserved for validation. This procedure provides a robust estimate of model performance across different data partitions, enabling a more accurate assessment of generalization capability. Moreover, when data are limited, ten-fold cross-validation ensures that all available samples contribute to both training and validation, thereby maximizing data utilization. Finally, by averaging results across multiple folds, the variance introduced by a single data split is reduced, leading to a more reliable and unbiased evaluation.

**中文:** 十折交叉验证是机器学习和统计建模中广泛采用的策略，用于评估模型的泛化性能、稳定性和可靠性。在这种方法中，训练数据集被划分为十个相等的子集；在每次迭代中，九个子集用于模型训练，剩余子集保留用于验证。该程序提供了模型在不同数据划分上性能的鲁棒估计，能够更准确地评估泛化能力。此外，当数据有限时，十折交叉验证确保所有可用样本都用于训练和验证，从而最大化数据利用。最后，通过对多个折叠的结果求平均，减少了单次数据划分引入的方差，导致更可靠和无偏的评估。

<a id="S036"></a>
**Source:** p.11 S036

**Original:** To validate the effectiveness of the mRNALocator-imb model, we conducted ten-fold cross-validation on the training set to evaluate its predictive performance. Macro F-measure, balanced accuracy (BACC), and the area under the ROC curve (AUC) were used as the primary evaluation metrics. Detailed definitions of these metrics are provided in Supplementary Text S2. Table 2 summarizes the experimental results obtained from ten-fold cross-validation on the training set, reporting the performance of each fold as well as the corresponding average values. As shown in the table, mRNALocator-imb demonstrates robust and competitive performance in predicting mRNA subcellular localization.

**中文:** 为了验证mRNALocator-imb模型的有效性，我们在训练集上进行了十折交叉验证以评估其预测性能。宏F度量、平衡准确率（BACC）和ROC曲线下面积（AUC）被用作主要评估指标。这些指标的详细定义在补充文本S2中提供。表2总结了在训练集上进行的十折交叉验证获得的实验结果，报告了每个折叠的性能以及相应的平均值。如表所示，mRNALocator-imb在预测mRNA亚细胞定位方面表现出鲁棒且具有竞争力的性能。

<a id="T002"></a>
### Table 2. Ten-fold cross-validation results on the training set
### 表2. 训练集上的十折交叉验证结果

**Placed near:** p.11 S036  
**Source:** p.13 C004  

| Ten-Fold Cross Validation | Macro F Measure | BACC | AUC |
|---|---|---|---|
| Fold-1 | 0.6355 | 0.6932 | 0.7210 |
| Fold-2 | 0.6353 | 0.6086 | 0.8437 |
| Fold-3 | 0.6707 | 0.6233 | 0.7504 |
| Fold-4 | 0.5630 | 0.5487 | 0.6982 |
| Fold-5 | 0.5724 | 0.5927 | 0.7922 |
| Fold-6 | 0.6305 | 0.6254 | 0.6901 |
| Fold-7 | 0.6080 | 0.5623 | 0.7615 |
| Fold-8 | 0.6743 | 0.6387 | 0.6531 |
| Fold-9 | 0.6132 | 0.5899 | 0.6847 |
| Fold-10 | 0.5880 | 0.6053 | 0.7609 |
| Mean | 0.6191 | 0.6088 | 0.7356 |

**Original caption:** Ten-fold cross-validation results on the training set  
**中文图注:** 训练集上的十折交叉验证结果

**Reading note:** 此表展示了mRNALocator-imb在训练集上十折交叉验证的详细结果，包括每个折叠和平均值的性能指标。

<a id="S037"></a>
**Source:** p.12 S037

**Original:** To validate the effectiveness of the proposed data imbalance handling strategy, we systematically investigated its impact on model performance from two perspectives: (i) the influence of different resampling techniques on traditional machine learning models, and (ii) the effect of alternative loss functions on deep learning models. Through comparative experiments, we demonstrate that the imbalance mitigation strategies adopted in this study lead to consistent performance improvements.

**中文:** 为了验证所提出的数据不平衡处理策略的有效性，我们从两个角度系统地调查了其对模型性能的影响：（i）不同重采样技术对传统机器学习模型的影响，以及（ii）替代损失函数对深度学习模型的影响。通过比较实验，我们证明本研究采用的不平衡缓解策略带来了一致的性能改进。

<a id="S038"></a>
**Source:** p.12 S038

**Original:** For resampling-based methods, we selected five representative techniques derived from the stochastic minority oversampling paradigm for comparative evaluation, including the Synthetic Minority Over-sampling Technique (SMOTE) [28], the Adaptive Synthetic Sampling (ADASYN) algorithm [29], random oversampling, random undersampling, and the NearMiss algorithm [30]. SMOTE addresses class imbalance by generating synthetic minority samples through linear interpolation between a minority instance and its nearest neighbors. ADASYN further extends this idea by adaptively generating synthetic samples according to the learning difficulty of individual minority instances, allocating more synthetic data to harder-to-learn samples and fewer to easier ones. Random oversampling mitigates imbalance by randomly replicating minority class samples to increase their representation in the dataset, whereas random undersampling reduces imbalance by randomly removing samples from the majority class. In contrast, the NearMiss algorithm selectively removes the majority class samples based on their distance to minority class instances, retaining those that are most informative for defining class boundaries, thereby achieving class balance while minimizing information loss.

**中文:** 对于基于重采样的方法，我们选择了从随机少数过采样范式衍生的五种代表性技术进行比较评估，包括合成少数过采样技术（SMOTE）[28]、自适应合成采样（ADASYN）算法[29]、随机过采样、随机欠采样和NearMiss算法[30]。SMOTE通过在少数实例及其最近邻之间进行线性插值来生成合成少数样本，从而解决类别不平衡。ADASYN通过根据单个少数实例的学习难度自适应地生成合成样本进一步扩展了这一思想，为难以学习的样本分配更多合成数据，为容易学习的样本分配较少数据。随机过采样通过随机复制少数类样本来增加其在数据集中的表示来缓解不平衡，而随机欠采样通过从多数类中随机删除样本来减少不平衡。相比之下，NearMiss算法根据多数类样本与少数类实例的距离选择性删除多数类样本，保留那些对定义类别边界最有信息的样本，从而在最小化信息损失的同时实现类别平衡。

<a id="S039"></a>
**Source:** p.13 S039

**Original:** We subsequently conducted comparative performance evaluations using the five aforementioned resampling techniques. All Random Forest models were trained with identical hyperparameter settings and employed the same fused feature representations as inputs, ensuring a fair comparison across resampling strategies. Ten-fold cross-validation was again performed exclusively on the training set, not only to obtain a comprehensive and robust performance estimate, but also to prevent test-set data leakage and avoid bias in selecting resampling techniques based on test-set outcomes.

**中文:** 随后，我们使用上述五种重采样技术进行了比较性能评估。所有随机森林模型都使用相同的超参数设置训练，并采用相同的融合特征表示作为输入，确保重采样策略之间的公平比较。再次仅在训练集上执行十折交叉验证，不仅是为了获得全面且鲁棒的性能估计，也是为了防止测试集数据泄漏并避免基于测试集结果选择重采样技术时的偏差。

<a id="S040"></a>
**Source:** p.13 S040

**Original:** As shown by the averaged ten-fold cross-validation results in Figure 3, the choice of resampling method has a substantial impact on model performance. Among the evaluated approaches, the Random Forest model trained with the ADASYN algorithm achieved the best overall performance, followed by SMOTE, random oversampling, NearMiss, and random downsampling. These results confirm the effectiveness of the proposed data imbalance handling strategy for the Random Forest sub-model.

**中文:** 如图3中平均十折交叉验证结果所示，重采样方法的选择对模型性能有重大影响。在评估的方法中，使用ADASYN算法训练的随机森林模型获得了最佳整体性能，其次是SMOTE、随机过采样、NearMiss和随机下采样。这些结果证实了所提出的数据不平衡处理策略对随机森林子模型的有效性。

<a id="F003"></a>
### Fig. 3. Performance comparison of different resampling techniques
### 图3. 不同重采样技术的性能比较

**Placed near:** p.13 S040  
**Source:** p.14 C005  

![Fig. 3](assets/fig_page14_img1.jpeg)

**Original caption:** Performance comparison of different resampling techniques  
**中文图注:** 不同重采样技术的性能比较

**Reading note:** 此图比较了五种不同重采样技术在随机森林模型上的性能表现。

<a id="S041"></a>
**Source:** p.14 S041

**Original:** From a methodological perspective, undersampling-based methods (i.e., NearMiss and random downsampling) generally underperformed oversampling-based approaches. This is primarily because undersampling balances class distributions by discarding majority-class samples, which may inadvertently remove informative instances and thereby degrade predictive performance. In contrast, oversampling methods such as SMOTE and ADASYN address class imbalance by generating synthetic minority-class samples, preserving majority-class information while enhancing minority-class representation and improving the model's discriminative capability.

**中文:** 从方法论角度来看，基于欠采样的方法（即NearMiss和随机下采样）通常表现不如基于过采样的方法。这主要是因为欠采样通过丢弃多数类样本来平衡类别分布，这可能会无意中删除信息丰富的实例，从而降低预测性能。相比之下，SMOTE和ADASYN等过采样方法通过生成合成少数类样本来解决类别不平衡，保留多数类信息的同时增强少数类表示并提高模型的判别能力。

<a id="S042"></a>
**Source:** p.14 S042

**Original:** The superior performance of ADASYN can be attributed to its adaptive sampling mechanism, which focuses on generating synthetic samples in regions where minority-class instances are harder to learn—particularly near class boundaries. By emphasizing these challenging regions, ADASYN improves the classifier's generalization ability, especially in datasets characterized by overlapping classes or ambiguous decision boundaries.

**中文:** ADASYN的优越性能可归因于其自适应采样机制，该机制专注于在少数类实例难以学习的区域生成合成样本——特别是在类别边界附近。通过强调这些具有挑战性的区域，ADASYN提高了分类器的泛化能力，特别是在具有重叠类别或模糊决策边界特征的数据集中。

<a id="S043"></a>
**Source:** p.14 S043

**Original:** Finally, we evaluated the impact of different loss functions on the performance of deep learning models. Five representative loss functions were selected for a systematic and comprehensive comparison, including cross-entropy (CE) loss [31], focal loss [32], weighted cross-entropy loss [33], the Lovász-softmax loss [34], and the Label-Distribution-Aware Margin (LDAM) loss [35]. All GRU-based models were trained using identical hyperparameter configurations and the same fused feature representations to ensure a fair comparison. Performance was assessed using a ten-fold cross-validation conducted on the training set.

**中文:** 最后，我们评估了不同损失函数对深度学习模型性能的影响。选择了五种代表性损失函数进行系统全面的比较，包括交叉熵（CE）损失[31]、焦点损失[32]、加权交叉熵损失[33]、Lovász-softmax损失[34]和标签分布感知边界（LDAM）损失[35]。所有基于GRU的模型都使用相同的超参数配置和相同的融合特征表示进行训练，以确保公平比较。使用在训练集上进行的十折交叉验证来评估性能。

<a id="S044"></a>
**Source:** p.15 S044

**Original:** As shown by the averaged ten-fold cross-validation results in Figure 4, the GRU model trained with the LDAM loss function achieved the best overall performance among all evaluated loss functions. These results highlight the effectiveness of LDAM in handling class imbalance and improving the predictive capability of GRU-based deep learning models for mRNA subcellular localization.

**中文:** 如图4中平均十折交叉验证结果所示，使用LDAM损失函数训练的GRU模型在所有评估的损失函数中获得了最佳整体性能。这些结果突显了LDAM在处理类别不平衡和提高基于GRU的深度学习模型对mRNA亚细胞定位的预测能力方面的有效性。

<a id="F004"></a>
### Fig. 4. Performance comparison of different loss functions
### 图4. 不同损失函数的性能比较

**Placed near:** p.15 S044  
**Source:** p.15 C006  

![Fig. 4](assets/fig_page15_img1.jpeg)

**Original caption:** Performance comparison of different loss functions  
**中文图注:** 不同损失函数的性能比较

**Reading note:** 此图比较了五种不同损失函数在GRU模型上的性能表现。

<a id="S045"></a>
**Source:** p.15 S045

**Original:** In addition, although resampling techniques and specialized loss functions are effective approaches for addressing class imbalance, the optimal imbalance-handling strategy must be determined in accordance with the specific characteristics of the dataset, model architecture, and application scenario. Thorough and systematic experimental evaluation is therefore essential prior to selecting a particular strategy, as such analyses can substantially enhance both the performance and practical applicability of models in imbalanced-data settings.

**中文:** 此外，虽然重采样技术和专用损失函数是解决类别不平衡的有效方法，但最佳的不平衡处理策略必须根据数据集、模型架构和应用场景的具体特征来确定。因此，在选择特定策略之前，进行全面系统的实验评估是必不可少的，因为此类分析可以显著提高模型在不平衡数据设置中的性能和实际适用性。

<a id="S046"></a>
**Source:** p.15 S046

**Original:** To comprehensively assess the effectiveness of the proposed mRNALocator-imb framework, we compared its performance with four representative existing methods, including RNATracker, mRNALoc, iLocmRNA, and SubLocEP, on an independent test dataset. RNATracker employs one-hot encoded sequence features and secondary structure information as inputs, and integrates a convolutional neural network (CNN), a long short-term memory (LSTM) network, and an attention mechanism to predict mRNA subcellular localization. mRNALoc utilizes pseudo k-tuple nucleotide composition (PseKNC) features and trains a support vector machine (SVM) classifier for localization prediction. iLocmRNA also adopts an SVM-based framework, in which features derived from binary encoding and k-mer composition are first selected using binary distribution and ANOVA-based feature selection before model training. SubLocEP extracts nine complementary feature representations, including electron–ion interaction pseudopotential (PseEIIP), trinucleotide composition (TNC), dinucleotide composition (DNC), composition of k-spacer nucleic acid pairs (CKSNAP), parallel correlation pseudo-DNC (PCPseDNC), parallel correlation pseudo-TNC (PCPseTNC), sequence correlation pseudo-DNC (SCPseDNC), sequence correlation pseudo-TNC (SCPseTNC), and dinucleotide autocorrelation covariance (DACC), and constructs a two-layer weighted ensemble model based on LightGBM.

**中文:** 为了全面评估所提出的mRNALocator-imb框架的有效性，我们在独立测试数据集上将其性能与四种代表性的现有方法进行了比较，包括RNATracker、mRNALoc、iLocmRNA和SubLocEP。RNATracker使用独热编码序列特征和二级结构信息作为输入，并整合卷积神经网络（CNN）、长短期记忆（LSTM）网络和注意力机制来预测mRNA亚细胞定位。mRNALoc利用伪k元核苷酸组成（PseKNC）特征并训练支持向量机（SVM）分类器进行定位预测。iLocmRNA也采用基于SVM的框架，其中从二进制编码和k-mer组成衍生的特征在模型训练之前首先使用二进制分布和基于方差分析的特征选择进行选择。SubLocEP提取九种互补的特征表示，包括电子-离子相互作用伪势（PseEIIP）、三核苷酸组成（TNC）、二核苷酸组成（DNC）、k间隔核酸对组成（CKSNAP）、并行相关伪DNC（PCPseDNC）、并行相关伪TNC（PCPseTNC）、序列相关伪DNC（SCPseDNC）、序列相关伪TNC（SCPseTNC）和二核苷酸自相关协方差（DACC），并基于LightGBM构建两层加权集成模型。

<a id="F005"></a>
### Fig. 5. Performance comparison of mRNALocator-imb and existing methods
### 图5. mRNALocator-imb与现有方法的性能比较

**Placed near:** p.15 S046  
**Source:** p.16 C007  

![Fig. 5](assets/fig_page16_img1.jpeg)

**Original caption:** Performance comparison of mRNALocator-imb and existing methods  
**中文图注:** mRNALocator-imb与现有方法的性能比较

**Reading note:** 此图展示了mRNALocator-imb与四种现有方法在独立测试集上的性能比较。

<a id="S047"></a>
**Source:** p.16 S047

**Original:** The detailed performance comparison results of the evaluated prediction models are presented in Figure 5. As shown in the figure, mRNALocator-imb consistently outperforms the four existing mRNA subcellular localization prediction methods across all evaluation metrics. Specifically, mRNALocator-imb achieves the highest Macro F-measure of 0.618. In terms of balanced accuracy (BACC), mRNALocator-imb attains a value of 0.612, exceeding those of SubLocEP (0.601), RNATracker (0.519), mRNALoc (0.491), and iLocmRNA (0.425). Similarly, on the AUC metric, mRNALocator-imb yields the best performance with a value of 0.701, surpassing all competing methods.

**中文:** 评估预测模型的详细性能比较结果如图5所示。如图所示，mRNALocator-imb在所有评估指标上始终优于四种现有的mRNA亚细胞定位预测方法。具体而言，mRNALocator-imb实现了最高的宏F度量0.618。在平衡准确率（BACC）方面，mRNALocator-imb达到0.612，超过了SubLocEP（0.601）、RNATracker（0.519）、mRNALoc（0.491）和iLocmRNA（0.425）。同样，在AUC指标上，mRNALocator-imb以0.701的值获得最佳性能，超越了所有竞争方法。

<a id="S048"></a>
**Source:** p.16 S048

**Original:** These results demonstrate the superior discriminative capability of mRNALocator-imb for distinguishing among different mRNA subcellular localization categories, as well as its robustness and accuracy across classes. The observed performance gains can be attributed to the comprehensive feature encoding scheme, the integrated model architecture, and the effective data imbalance handling strategy employed in this study, all of which contribute to enhanced predictive performance.

**中文:** 这些结果证明了mRNALocator-imb在区分不同mRNA亚细胞定位类别方面的卓越判别能力，以及其在类别间的鲁棒性和准确性。观察到的性能增益可归因于本研究采用的综合特征编码方案、集成模型架构和有效的数据不平衡处理策略，所有这些都促进了增强的预测性能。

<a id="S049"></a>
**Source:** p.16 S049

**Original:** To gain deeper insight into the decision-making mechanism of the Random Forest model for the five-class classification task and to identify key features influencing mRNA subcellular localization prediction, we employed the SHapley Additive exPlanations (SHAP) framework to quantify both the importance and directional effects of individual features on model outputs. SHAP provides a unified measure of feature contribution by attributing each prediction to its constituent features in a theoretically grounded manner.

**中文:** 为了深入了解随机森林模型对五类分类任务的决策机制并识别影响mRNA亚细胞定位预测的关键特征，我们采用SHapley加性解释（SHAP）框架来量化单个特征对模型输出的重要性和方向性影响。SHAP通过以理论 grounded的方式将每个预测归因于其构成特征，提供了特征贡献的统一度量。

<a id="S050"></a>
**Source:** p.17 S050

**Original:** In this study, SHAP was used to summarize and analyze global model behavior by estimating feature importance and feature effects. Feature importance was quantified as the mean absolute SHAP value of each feature across all samples, with larger values indicating greater influence on model predictions. The feature importance score for feature j is defined as follows: FI_j = (1/n) * Σ_{i=1}^n |φ^(i)_j| (3), where n denotes the total number of samples in the dataset and φ^(i)_j represents the SHAP value of feature j for the i-th sample.

**中文:** 在本研究中，SHAP用于通过估计特征重要性和特征效应来总结和分析全局模型行为。特征重要性量化为每个特征在所有样本上的平均绝对SHAP值，较大的值表示对模型预测的影响更大。特征j的特征重要性得分定义如下：FI_j = (1/n) * Σ_{i=1}^n |φ^(i)_j| (3)，其中n表示数据集中的样本总数，φ^(i)_j表示第i个样本的特征j的SHAP值。

<a id="S051"></a>
**Source:** p.17 S051

**Original:** Figure 6 displays the top 30 features with the greatest impact on the model. In the figure, the x-axis represents SHAP values, where positive values indicate a positive contribution to the predicted class and negative values indicate a negative contribution. The y-axis lists features ranked by their mean absolute SHAP values across all samples. As shown in Figure 6, features such as Cytosine content (lag1), Guanine content (lag1), CCT_Nucleosome-Rigid, CCC_MW-Daltons, CCC_Consensus_roll, MMI_GG, and CCC trinucleotide GC content exhibit particularly strong influence on model predictions, indicating that the Random Forest model relies heavily on these features during decision making.

**中文:** 图6显示了对模型影响最大的前30个特征。图中，x轴表示SHAP值，正值表示对预测类别的正贡献，负值表示负贡献。y轴列出了按所有样本上的平均绝对SHAP值排序的特征。如图6所示，胞嘧啶含量（lag1）、鸟嘌呤含量（lag1）、CCT_Nucleosome-Rigid、CCC_MW-Daltons、CCC_Consensus_roll、MMI_GG和CCC三核苷酸GC含量等特征对模型预测表现出特别强的影响，表明随机森林模型在决策过程中严重依赖这些特征。

<a id="F006"></a>
### Fig. 6. SHAP values for input features of the Random Forest model
### 图6. 随机森林模型输入特征的SHAP值

**Placed near:** p.17 S051  
**Source:** p.18 C008  

![Fig. 6](assets/fig_page18_img1.jpeg)

**Original caption:** SHAP values for input features of the Random Forest model  
**中文图注:** 随机森林模型输入特征的SHAP值

**Reading note:** 此图展示了随机森林模型中最重要的30个输入特征的SHAP值分析。

<a id="S052"></a>
**Source:** p.18 S052

**Original:** Further analysis of the underlying feature encoding schemes reveals that these influential features are derived primarily from nucleotide-level and k-mer-based representations, involving guanine (G), cytosine (C), the dinucleotide group GG, and trinucleotide groups such as CCT and CCC. This observation suggests that specific nucleotide compositions and local sequence patterns play a critical role in determining mRNA subcellular localization, highlighting the biological relevance of the selected feature encodings.

**中文:** 对潜在特征编码方案的进一步分析表明，这些有影响力的特征主要来源于核苷酸级别和基于k-mer的表示，涉及鸟嘌呤（G）、胞嘧啶（C）、二核苷酸组GG以及CCT和CCC等三核苷酸组。这一观察表明，特定的核苷酸组成和局部序列模式在决定mRNA亚细胞定位方面起着关键作用，突显了所选特征编码的生物学相关性。

---

## Conclusion and Discussion / 结论与讨论

<a id="S053"></a>
**Source:** p.18 S053

**Original:** In this study, we proposed mRNALocator-imb, a weighted ensemble learning framework that integrates a Random Forest submodel and a gated recurrent unit (GRU) submodel for accurate prediction of mRNA subcellular localization. To construct this framework, five complementary feature encoding strategies were selected as model inputs. Among them, four fused feature representations excluding Word2Vec were used to train the Random Forest submodel, while Word2Vec embeddings were specifically employed as inputs to the GRU submodel, enabling each component model to fully exploit its respective strengths. Correspondingly, tailored data imbalance handling strategies were designed for the Random Forest and GRU submodels, and optimal hyperparameters of the integrated framework were determined via grid search combined with ten-fold cross-validation.

**中文:** 在本研究中，我们提出了mRNALocator-imb，这是一个加权集成学习框架，整合了随机森林子模型和门控循环单元（GRU）子模型，用于准确预测mRNA亚细胞定位。为了构建这个框架，选择了五种互补的特征编码策略作为模型输入。其中，四种不包括Word2Vec的融合特征表示用于训练随机森林子模型，而Word2Vec嵌入专门用作GRU子模型的输入，使每个组件模型能够充分利用其各自的优势。相应地，为随机森林和GRU子模型设计了定制的数据不平衡处理策略，并通过网格搜索结合十折交叉验证确定了集成框架的最佳超参数。

<a id="S054"></a>
**Source:** p.18 S054

**Original:** To comprehensively evaluate the proposed model and associated strategies, extensive experiments were conducted, including ten-fold cross-validation on the training set, systematic validation of data imbalance handling strategies, performance comparisons with conventional machine learning models, and benchmarking against existing state-of-the-art methods on an independent test dataset. Experimental results consistently demonstrate that mRNALocator-imb achieves superior and robust performance across multiple evaluation metrics, highlighting the effectiveness of the integrated architecture, feature fusion strategy, and imbalance-aware optimization.

**中文:** 为了全面评估所提出的模型和相关策略，进行了广泛的实验，包括训练集上的十折交叉验证、数据不平衡处理策略的系统验证、与传统机器学习模型的性能比较以及在独立测试数据集上与现有最先进方法的基准测试。实验结果一致表明，mRNALocator-imb在多个评估指标上实现了优越且鲁棒的性能，突显了集成架构、特征融合策略和不平衡感知优化的有效性。

<a id="S055"></a>
**Source:** p.18 S055

**Original:** Furthermore, the decision-making process of the Random Forest submodel was interpreted using the SHAP framework to quantify feature contributions and dependencies. This analysis not only enhances the interpretability of the proposed model but also reveals that specific nucleotides and nucleotide groups—such as cytosine- and guanine-related patterns—play a prominent role in mRNA subcellular localization prediction. These biologically meaningful findings provide additional evidence supporting the rationality of the selected feature encoding schemes and offer new insights into the potential sequence determinants of mRNA localization.

**中文:** 此外，使用SHAP框架解释了随机森林子模型的决策过程，以量化特征贡献和依赖关系。这一分析不仅增强了所提出模型的可解释性，还揭示了特定核苷酸和核苷酸组——如胞嘧啶和鸟嘌呤相关模式——在mRNA亚细胞定位预测中起着突出作用。这些具有生物学意义的发现为所选特征编码方案的合理性提供了额外证据，并为mRNA定位的潜在序列决定因素提供了新见解。

<a id="S056"></a>
**Source:** p.18 S056

**Original:** Overall, the proposed mRNALocator-imb framework demonstrates strong predictive performance, interpretability, and robustness under class imbalance, and we anticipate that the methodological insights and biological interpretations presented in this study will contribute valuable perspectives and inspiration for future research on mRNA subcellular localization.

**中文:** 总体而言，所提出的mRNALocator-imb框架在类别不平衡下表现出强大的预测性能、可解释性和鲁棒性，我们预期本研究展示的方法见解和生物学解释将为未来的mRNA亚细胞定位研究提供有价值的视角和灵感。

---

## Data and Code Availability / 数据和代码可用性

<a id="S057"></a>
**Source:** p.19 S057

**Original:** The raw data utilized in this study, along with comprehensive details of the mRNALocator-imb code, are available for access at https://github.com/HaoWuLab Bioinformatics/mRNALocator-imb.

**中文:** 本研究使用的原始数据以及mRNALocator-imb代码的详细信息可在https://github.com/HaoWuLab Bioinformatics/mRNALocator-imb访问。

---

## Acknowledgments / 致谢

<a id="S058"></a>
**Source:** p.19 S058

**Original:** We thank members of the group for their valuable discussions and comments. The scientific calculations in this study have been done on the HPC Cloud Platform of Shandong University.

**中文:** 我们感谢小组成员的有益讨论和评论。本研究的科学计算已在山东大学HPC云平台上完成。

---

## Funding / 资助

<a id="S059"></a>
**Source:** p.19 S059

**Original:** This work is supported by the Guangdong Basic and Applied Basic Research Foundation (Grant No.2024A1515012775) and the National Natural Science Foundation of China (Grant Nos. 62272278 & 61972322). The funders did not play any role in the design of the study, the collection, analysis, and interpretation of data, or the writing of the manuscript.

**中文:** 本工作由广东省基础与应用基础研究基金会（批准号2024A1515012775）和国家自然科学基金（批准号62272278和61972322）支持。资助者在研究设计、数据收集、分析和解释或手稿撰写方面未发挥任何作用。

---

## References / 参考文献

<a id="S060"></a>
**Source:** p.19-22 S060

**Original:** [References 1-35 listed in the paper]

**中文:** [论文中列出的参考文献1-35]

---

## 阅读提示 / Critical Reading Notes

本文提出了一种名为mRNALocator-imb的不平衡容忍集成学习框架，用于预测mRNA的亚细胞定位。该研究的主要贡献包括：

1. **混合架构设计**：结合了随机森林和门控循环单元（GRU）网络，充分利用传统机器学习的鲁棒性和深度学习的序列建模能力。

2. **多尺度特征融合**：采用了五种互补的特征编码方案（NMBACC、TPCP、DACC、MMI和Word2Vec），全面表征mRNA序列信息。

3. **不平衡处理策略**：针对不同子模型采用定制的不平衡处理方法——随机森林使用ADASYN数据重采样，GRU使用LDAM损失函数。

4. **性能优势**：在独立测试集上优于现有的四种方法（RNATracker、mRNALoc、iLocmRNA、SubLocEP），特别是在不平衡分类设置中表现突出。

5. **可解释性分析**：使用SHAP框架分析特征重要性，发现胞嘧啶和鸟嘌呤相关模式在mRNA亚细胞定位预测中起关键作用。

该研究为mRNA亚细胞定位预测提供了一个鲁棒、可泛化且具有生物学可解释性的框架，对生物信息学和生物医学研究具有重要价值。

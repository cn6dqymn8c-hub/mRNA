# A BERT-based model for the prediction of lncRNA subcellular localization in Homo sapiens
# 基于BERT的人类长链非编码RNA亚细胞定位预测模型

**Source:** E:\mRNA\Doc\Bert.pdf  
**Journal:** International Journal of Biological Macromolecules  
**Volume:** 265 (2024)  
**Article Number:** 130659  
**Available online:** 10 March 2024  
**DOI:** https://doi.org/10.1016/j.ijbiomac.2024.130659  
**Received:** 11 January 2024  
**Received in revised form:** 19 February 2024  
**Accepted:** 4 March 2024  
**Authors:** Zhao-Yue Zhang, Zheng Zhang, Xiucai Ye, Tetsuya Sakurai, Hao Lin  
**Corresponding authors:** Xiucai Ye (yexiucai@cs.tsukuba.ac.jp), Hao Lin (hlin@uestc.edu.cn)  

---

## Abstract / 摘要

<a id="S001"></a>
**Source:** p.1 S001

**Original:** Understanding the subcellular localization of lncRNAs is crucial for comprehending their regulation activities. The conventional detection of lncRNA subcellular location usually uses in situ detection techniques, which are resource intensive. Some machine learning-based algorithms have been proposed for lncRNA subcellular location prediction in mammals. However, due to the low level of conservation of lncRNA sequence, the performance of cross-species models remains unsatisfactory. In this study, we curated a novel dataset containing subcellular location information of lncRNAs in Homo sapiens. Subsequently, based on the BERT pre-trained language algorithm, we developed a model for lncRNA subcellular location prediction. Our model achieved a micro-average area under the receiver operating characteristic (AUROC) of 0.791 on the training set and an AUROC of 0.700 on the testing nucleus set. Additionally, we conducted cross-species validation and motif discovery to further investigate underlying patterns. In summary, our study provides valuable guidance and computational analysis tools for exploring the mechanisms of lncRNA subcellular localization and the dynamic spatial changes of RNA in abnormal physiological states.

**中文:** 理解lncRNA的亚细胞定位对于理解其调控活动至关重要。lncRNA亚细胞定位的传统检测通常使用原位检测技术，这些技术资源密集。已经提出了一些基于机器学习的算法用于哺乳动物lncRNA亚细胞定位预测。然而，由于lncRNA序列保守性水平较低，跨物种模型的性能仍然不令人满意。在本研究中，我们整理了一个包含人类lncRNA亚细胞定位信息的新数据集。随后，基于BERT预训练语言算法，我们开发了一个lncRNA亚细胞定位预测模型。我们的模型在训练集上实现了0.791的接收者操作特征曲线下微平均面积（AUROC），在测试核集上实现了0.700的AUROC。此外，我们进行了跨物种验证和基序发现以进一步研究潜在模式。总之，我们的研究为探索lncRNA亚细胞定位机制和RNA在异常生理状态下的动态空间变化提供了有价值的指导和计算分析工具。

**Keywords:** lncRNA, Subcellular localization, Multi-label classification, BERT  
**关键词:** 长链非编码RNA，亚细胞定位，多标签分类，BERT

---

## 1. Introduction / 引言

<a id="S002"></a>
**Source:** p.1 S002

**Original:** Long non-coding RNA (lncRNA) is a type of RNA without protein-coding function. Using the genome as a template, lncRNAs were transcribed by RNA polymerase II and general transcription factors. Although most of them lack protein-coding potential, this large population of RNAs is functionally associated with transcription or translation regulation [1,2]. For instance, the transcription of ThymoD inhibits the methylation of the CCCTC-binding transcription factors (CTCF) binding site, thereby enabling Bcl11b gene expression [3]. The lncRNA LAST binds to the 5′UTR of CCND1 mRNA to protect against possible nuclease targeting stabilizes, resulting in an increase of CCND1 expression [4]. Although lncRNAs have received considerable attention in recent years, their various regulatory mechanisms have not been explored to a large extent.

**中文:** 长链非编码RNA（lncRNA）是一种没有蛋白质编码功能的RNA类型。以基因组为模板，lncRNA由RNA聚合酶II和通用转录因子转录。尽管大多数缺乏蛋白质编码潜力，但这大量RNA在功能上与转录或翻译调控相关[1,2]。例如，ThymoD的转录抑制CCCTC结合转录因子（CTCF）结合位点的甲基化，从而实现Bcl11b基因表达[3]。lncRNA LAST结合到CCND1 mRNA的5′UTR以防止可能的核酸酶靶向稳定，导致CCND1表达增加[4]。尽管近年来lncRNA受到相当关注，但其各种调控机制尚未得到充分探索。

<a id="S003"></a>
**Source:** p.1 S003

**Original:** The intracellular positioning of lncRNAs plays a crucial role in determining their regulatory activities and functional roles [5]. Different subcellular compartments provide distinct microenvironments and interaction partners, which influence lncRNA function and their ability to participate in various cellular processes. Studying the subcellular location of lncRNA is crucial to understand their roles in cellular processes, gene regulation, and disease mechanisms. Techniques like fluorescence in situ hybridization (FISH) [6] and subcellular fractionation [7] offer valuable insights into the intracellular localization of lncRNAs in various cellular compartments. However, the resource-intensive experimental processes, with their time, money, and labor demands, may impede the research progress in RNA localization. To address this challenge, integrating artificial intelligence algorithms [8–11] can provide valuable assistance in detecting lncRNA subcellular localization [12–15]. This approach enables researchers to gain insights into the subcellular positioning and functional roles of lncRNAs in cellular processes and disease mechanisms.

**中文:** lncRNA的细胞内定位在决定其调控活动和功能角色方面起着关键作用[5]。不同的亚细胞区室提供不同的微环境和相互作用伙伴，这影响lncRNA功能及其参与各种细胞过程的能力。研究lncRNA的亚细胞定位对于理解它们在细胞过程、基因调控和疾病机制中的作用至关重要。荧光原位杂交（FISH）[6]和亚细胞分级分离[7]等技术为各种亚细胞区室中lncRNA的细胞内定位提供了有价值的见解。然而，资源密集的实验过程，其时间、金钱和劳动力需求，可能会阻碍RNA定位的研究进展。为了应对这一挑战，整合人工智能算法[8-11]可以为检测lncRNA亚细胞定位提供有价值的帮助[12-15]。这种方法使研究人员能够深入了解lncRNA在细胞过程和疾病机制中的亚细胞定位和功能角色。

<a id="S004"></a>
**Source:** p.1 S004

**Original:** Although many machine learning methods have been proposed for the prediction of lncRNA subcellular location in mammals, most of them transform the original multi-label classification task into a multi-class classification task [12–28]. In this study, we developed a multi-label prediction framework using Bidirectional Encoder Representations from Transformers (BERT) [29] to identify lncRNA subcellular locations (Fig. 1). Through a series of experiments, we have demonstrated the effectiveness of the predictive model. The following is the detailed process of model construction.

**中文:** 尽管已经提出了许多机器学习方法用于预测哺乳动物lncRNA的亚细胞定位，但大多数方法将原始多标签分类任务转换为多类分类任务[12-28]。在本研究中，我们开发了一个使用双向编码器表示来自变换器（BERT）[29]的多标签预测框架来识别lncRNA亚细胞定位（图1）。通过一系列实验，我们证明了预测模型的有效性。以下是模型构建的详细过程。

<a id="F001"></a>
### Fig. 1. The flowchart of multi-label lncRNA subcellular location prediction
### 图1. 多标签lncRNA亚细胞定位预测流程图

**Placed near:** p.1 S004  
**Source:** p.2 C001  

![Fig. 1](bert_assets/page2_img1.jpeg)

**Original caption:** The flowchart of multi-label lncRNA subcellular location prediction.  
**中文图注:** 多标签lncRNA亚细胞定位预测流程图。

**Reading note:** 此图展示了多标签lncRNA亚细胞定位预测的整体流程，包括数据输入、BERT模型处理和多标签输出。

---

## 2. Materials and methods / 材料与方法

<a id="S005"></a>
**Source:** p.2 S005

**Original:** Previously, the majority of lncRNA subcellular location prediction efforts utilized a dataset of 655 mammalian lncRNA sequences collected by Su et al. [22]. However, the dataset excluded the lncRNAs that are simultaneously present in multiple subcellular locations. In this work, we focused on the identification of lncRNAs that were found to be presented in multiple locations. We retrieved the lncRNA subcellular location information in Homo sapiens (H. sapiens) and Mus musculus (M. musculus) with experimental evidence from RNALocate [30] and lncRNA nucleotide sequences from GenBank [31] with the annotation "long non-coding RNA". After removing duplicated lncRNA entries, deleting data entry errors, and excluding lncRNAs with unmatched sequences, we obtained a total of 219 distinct lncRNA sequences in H. sapiens. As the small samples may lead to the distortion of the prediction model, we categorized the lncRNAs in the nucleus, cytoplasm, and extracellular vesicle. Among these, 134 lncRNAs were found in the nucleus, 143 in the cytoplasm, and 18 in the extracellular vesicle. The number of samples and the length distribution of each subcellular location in the training set have been depicted in Fig. 2.

**中文:** 以前，大多数lncRNA亚细胞定位预测工作使用了Su等人[22]收集的655个哺乳动物lncRNA序列数据集。然而，该数据集排除了同时存在于多个亚细胞位置的lncRNA。在这项工作中，我们专注于识别发现存在于多个位置的lncRNA。我们从RNALocate [30]检索了具有实验证据的人类（H. sapiens）和小鼠（M. musculus）lncRNA亚细胞定位信息，并从GenBank [31]检索了标注为"长链非编码RNA"的lncRNA核苷酸序列。在删除重复的lncRNA条目、删除数据输入错误并排除序列不匹配的lncRNA后，我们在人类中总共获得了219个不同的lncRNA序列。由于小样本可能导致预测模型的失真，我们将lncRNA分类为细胞核、细胞质和细胞外囊泡。其中，134个lncRNA在细胞核中发现，143个在细胞质中发现，18个在细胞外囊泡中发现。训练集中每个亚细胞位置的样本数量和长度分布如图2所示。

<a id="F002"></a>
### Fig. 2. Length distribution of lncRNA sequences and the number of samples in each subcellular location
### 图2. lncRNA序列长度分布和每个亚细胞位置的样本数量

**Placed near:** p.2 S005  
**Source:** p.2 C002  

![Fig. 2](bert_assets/page2_img2.jpeg)

**Original caption:** Length distribution of lncRNA sequences and the number of samples in each subcellular location.  
**中文图注:** lncRNA序列长度分布和每个亚细胞位置的样本数量。

**Reading note:** 此图展示了lncRNA序列的长度分布以及不同亚细胞位置的样本数量，为数据集特征提供了可视化。

<a id="T001"></a>
### Table 1. The detailed information of the dataset
### 表1. 数据集的详细信息

**Placed near:** p.2 S005  
**Source:** p.2 T001  

| Distinct lncRNA | Subcellular location | Nucleus | Cytoplasm | Extracellular vesicle |
|---|---|---|---|---|
| **Training set** | | | | |
| | H. sapiens | 219 | 134 | 143 | 18 |
| **Testing set** | | | | |
| | H. sapiens | 623 | 619 | 35 | 0 |
| **Mus musculus** | 65 | 31 | 20 | 27 |

**中文表注:** 训练集、测试集和小鼠数据集中不同亚细胞位置的lncRNA数量分布。

**Reading note:** 此表详细列出了训练集、测试集和小鼠数据集中不同亚细胞位置的lncRNA数量。

<a id="S006"></a>
**Source:** p.3 S006

**Original:** For the testing set, we utilized the H. sapiens dataset employed in LncLocFormer [15]. LncLocFormer excluded exosome-localized entries in their study. Thus, our testing set comprised 623 lncRNAs, with 619 localized in the nucleus and 35 in the cytoplasm. Using the same data processing approach as the training set, we obtained subcellular localization data for lncRNAs in M. musculus, encompassing 65 lncRNAs. Among these, 31 lncRNAs were identified in the nucleus, 20 in the cytoplasm, and 27 in the extracellular vesicle. Detailed information regarding the dataset was provided in Table 1. Notably, the H. sapiens training set was used to train the deep learning model, LncLocFormer testing set was employed to assess the model's performance, while the M. musculus dataset was used to evaluate the mode's species transferability.

**中文:** 对于测试集，我们使用了LncLocFormer [15]中采用的人类数据集。LncLocFormer在其研究中排除了外泌体定位的条目。因此，我们的测试集包含623个lncRNA，其中619个定位于细胞核，35个定位于细胞质。使用与训练集相同的数据处理方法，我们获得了小鼠lncRNA的亚细胞定位数据，涵盖65个lncRNA。其中，31个lncRNA在细胞核中识别，20个在细胞质中，27个在细胞外囊泡中。关于数据集的详细信息在表1中提供。值得注意的是，人类训练集用于训练深度学习模型，LncLocFormer测试集用于评估模型性能，而小鼠数据集用于评估模型的物种可转移性。

<a id="S007"></a>
**Source:** p.3 S007

**Original:** Deep learning methods have been extensively applied in the prediction of sequence modification sites [32–36], disease risk [37], drug discovery [34,38–41], and other biological aspects [42,43]. In recent years, the remarkable success of BERT in natural language processing tasks made it a promising candidate for sequence-based predictions [44,45]. The standard BERT framework consists of multiple stacked encoder layers, and each includes self-attention mechanisms and feed-forward neural networks. The self-attention mechanism allows the model to weigh the importance of different elements in a sequence, while the feed-forward networks capture non-linear relationships between features. These layers work collaboratively to generate rich contextual embeddings for the input sequences, which are then used for downstream prediction tasks. Due to the robust capabilities of BERT, it could deliver accurate predictions for the subcellular location of lncRNAs.

**中文:** 深度学习方法已广泛应用于序列修饰位点预测[32-36]、疾病风险[37]、药物发现[34,38-41]和其他生物学方面[42,43]。近年来，BERT在自然语言处理任务中的显著成功使其成为基于序列预测的有力候选者[44,45]。标准BERT框架由多个堆叠的编码器层组成，每层包括自注意力机制和前馈神经网络。自注意力机制允许模型权衡序列中不同元素的重要性，而前馈网络捕获特征之间的非线性关系。这些层协同工作为输入序列生成丰富的上下文嵌入，然后用于下游预测任务。由于BERT的强大能力，它可以为lncRNA的亚细胞定位提供准确的预测。

<a id="S008"></a>
**Source:** p.3 S008

**Original:** In the framework of lncRNA subcellular location prediction, as illustrated in Fig. 1, we input the truncated lncRNA sequences into the deep learning model. The "DNABERT-2" [46] was used to integrate pre-trained representation of DNA sequences, which can automatically handle sequences longer than 512. Subsequently, two linear layers were appended to the "DNABERT-2" architecture, each preceded by a dropout layer for regularization. The first linear layer reduced the dimensionality from 768 to a hidden size of 64. Following this, another dropout layer is employed before the second linear layer, which further reduces the dimensions from 64 to 3. Two-way Multi-Label Loss [47] and Adaptive Moment Estimation (Adam) algorithm were used for the hyper-parameter tuning. The Sigmoid activation function was applied to generate the prediction probabilities for each location. In the hyper-parameter optimization process, the model was trained and evaluated by 10-fold stratified cross-validation. To counteract overfitting, within the cross-validation process, training for a specific fold was halted when the training set's loss exceeded the testing set's loss by more than 0.1 for three occurrences. The deep-learning model was implemented using Pytorch 2.1.0. Model training and testing were performed with NVIDIA Tesla V100 SXM2.

**中文:** 在lncRNA亚细胞定位预测框架中，如图1所示，我们将截断的lncRNA序列输入深度学习模型。"DNABERT-2" [46]用于整合DNA序列的预训练表示，它可以自动处理超过512的序列。随后，两个线性层附加到"DNABERT-2"架构，每个线性层前面都有一个dropout层用于正则化。第一个线性层将维度从768减少到隐藏大小64。在此之后，在第二个线性层之前使用另一个dropout层，进一步将维度从64减少到3。双向多标签损失[47]和自适应矩估计（Adam）算法用于超参数调整。Sigmoid激活函数应用于生成每个位置的预测概率。在超参数优化过程中，模型通过10折分层交叉验证进行训练和评估。为了对抗过拟合，在交叉验证过程中，当训练集的损失超过测试集的损失超过0.1三次时，特定折的训练停止。深度学习模型使用Pytorch 2.1.0实现。模型训练和测试使用NVIDIA Tesla V100 SXM2进行。

<a id="S009"></a>
**Source:** p.3 S009

**Original:** Utilizing the optimal combinations of hyperparameters that yielded the highest micro-average area under the receiver operating characteristic (AUROC), we constructed a model to predict lncRNA subcellular locations and subjected it to a 10-fold cross-validation assessment. The evaluation of the multi-label classification was calculated by the macro/micro-average precision (macro/micro-Pre), macro/micro-average recall (macro/micro-Recall), macro/micro-average F1-measure (macro/micro-F1), macro/micro-average AUROC, the area under the precision-recall curve (AUPRC), hamming loss (HL) and Jaccard index [48].

**中文:** 利用产生最高接收者操作特征曲线下微平均面积（AUROC）的超参数最佳组合，我们构建了一个模型来预测lncRNA亚细胞定位并对其进行10折交叉验证评估。多标签分类的评估通过宏/微平均精度（macro/micro-Pre）、宏/微平均召回率（macro/micro-Recall）、宏/微平均F1度量（macro/micro-F1）、宏/微平均AUROC、精确召回曲线下面积（AUPRC）、汉明损失（HL）和Jaccard指数[48]计算。

---

## 3. Results / 结果

<a id="S010"></a>
**Source:** p.3 S010

**Original:** Given the species specificity of lncRNAs, we restricted training solely to the H. sapiens data. The length of lncRNA sequences varies from 192 nucleotides (nt) to 19,296 nt, with a median length of 1897 nt. Padding all lncRNA sequences to maximum length would lead to an increase in unnecessary computing load. Therefore, the initial stage of model development involves homogenizing lncRNA sequences by truncating their lengths. Studies have indicated that short sequence elements within the RNA sequence, predominantly suited in the 3′ UTR, can function in cis to regulate RNA localization. In this primary phase, we randomly trimmed sequences 200 times within the range of 512 nt to 2560 nt at 3 end of lncRNA sequences. An appropriate truncation length was discerned by evaluating the macro-average AUROC values attained during model construction. The influence of various input sequence lengths on the model was depicted in Fig. 3A, and additional details were provided in Supplementary Table 1. Finally, employing sequences with a 3′ end length of 2292 nt as model input culminated in optimal performance.

**中文:** 鉴于lncRNA的物种特异性，我们将训练仅限于人类数据。lncRNA序列的长度从192个核苷酸（nt）到19,296 nt不等，中位长度为1897 nt。将所有lncRNA序列填充到最大长度会导致不必要的计算负荷增加。因此，模型开发的初始阶段涉及通过截断长度来均质化lncRNA序列。研究表明，RNA序列中的短序列元件，主要适合在3′UTR中，可以顺式作用调节RNA定位。在这个初级阶段，我们在lncRNA序列的3端随机修剪序列200次，范围从512 nt到2560 nt。通过评估模型构建期间获得的宏平均AUROC值来识别适当的截断长度。各种输入序列长度对模型的影响如图3A所示，补充表1中提供了更多细节。最后，使用3′端长度为2292 nt的序列作为模型输入实现了最佳性能。

<a id="F003"></a>
### Fig. 3. Prediction results on training data
### 图3. 训练数据上的预测结果

**Placed near:** p.3 S010  
**Source:** p.3 C003  

![Fig. 3](bert_assets/page3_img1.jpeg)

**Original caption:** Prediction results on training data. (A) The predictive performance based on different truncation lengths. (B) ROC and (C) PR curves for iLoc-lncRNA-BERT on H. sapiens training dataset.  
**中文图注:** 训练数据上的预测结果。（A）基于不同截断长度的预测性能。（B）iLoc-lncRNA-BERT在人类训练数据集上的ROC和（C）PR曲线。

**Reading note:** 此图展示了不同截断长度对预测性能的影响，以及模型在训练数据集上的ROC和PR曲线。

<a id="S011"></a>
**Source:** p.3 S011

**Original:** Once the ideal truncation length was established, the hyper-parameter search was conducted systematically to identify the combination that yielded the optimal model performance for the given task. A grid search was conducted to investigate the impact of batch size, dropout values, and learning rates on the model's performance. Specifically, batch sizes ranging from 8 to 32 were evaluated, encompassing values of 8, 16 and 32. Dropout values of 0.1, 0.2, 0.3, 0.4, and 0.5 were examined to assess their influence on regularization. Additionally, learning rates spanning 1e−5 and 3e−5 were tested to determine their effect on model convergence and overall accuracy. All other hyper-parameters were retained at their default settings. As our data is highly imbalanced, we employed hierarchical classification with 10-fold cross-validation during the training process. The obtained output probabilities were presented as estimations of specific subcellular locations.

**中文:** 一旦建立了理想的截断长度，系统地进行超参数搜索以识别为给定任务产生最佳模型性能的组合。进行了网格搜索以研究批量大小、dropout值和学习率对模型性能的影响。具体而言，评估了从8到32的批量大小，包括8、16和32的值。检查了0.1、0.2、0.3、0.4和0.5的dropout值以评估其对正则化的影响。此外，测试了跨越1e−5和3e−5的学习率以确定其对模型收敛和整体准确性的影响。所有其他超参数保留在其默认设置。由于我们的数据高度不平衡，我们在训练过程中采用分层分类和10折交叉验证。获得的输出概率作为特定亚细胞位置的估计。

<a id="S012"></a>
**Source:** p.3 S012

**Original:** When the batch size was set to 16, a dropout value of 0.1 for both two dropout layers, and a learning rate of 1e−5, the model achieved its best macro-AUROC of 0.6701 (see Supplementary Table 2). Accordingly, we adopted this model as our predictor for lncRNA subcellular location.

**中文:** 当批量大小设置为16，两个dropout层的dropout值均为0.1，学习率为1e−5时，模型实现了其最佳宏AUROC为0.6701（见补充表2）。因此，我们采用此模型作为lncRNA亚细胞定位的预测器。

<a id="T002"></a>
### Table 2. The performance of the iLoc-lncRNA-BERT
### 表2. iLoc-lncRNA-BERT的性能

**Placed near:** p.3 S012  
**Source:** p.3 T002  

| iLoc-lncRNA-BERT | H. sapiens training set | | H. sapiens testing set | | Mus musculus | |
|---|---|---|---|---|---|---|
| | Macro | Micro | Macro | Micro | Macro | Micro |
| AUROC | 0.639 | 0.791 | 0.592 | 0.306 | 0.555 | 0.527 |
| AUPR | 0.575 | 0.696 | 0.53 | 0.423 | 0.481 | 0.454 |
| Pre | 0.658 | 0.655 | 0.529 | 0.396 | 0.264 | 0.384 |
| Recall | 0.64 | 0.857 | 0.733 | 0.545 | 0.511 | 0.494 |
| F1 | 0.572 | 0.742 | 0.397 | 0.459 | 0.34 | 0.432 |
| HL | 0.268 | 0.675 | 0.521 | | | |
| Jaccard | 0.59 | | 0.298 | 0.275 | | |

**中文表注:** iLoc-lncRNA-BERT在人类训练集、测试集和小鼠数据集上的宏平均和微平均性能指标。

**Reading note:** 此表展示了iLoc-lncRNA-BERT模型在不同数据集上的详细性能指标，包括AUROC、AUPR、精确率、召回率、F1分数、汉明损失和Jaccard指数。

<a id="S013"></a>
**Source:** p.3 S013

**Original:** Continuing our previous models for lncRNA subcellular location prediction, namely iLoc-lncRNA and iLoc-lncRNA 2.0, the current model was named iLoc-lncRNA-BERT. Detailed results of iLoc-lncRNA-BERT were outlined in Table 2. The AUROC for the nucleus, cytoplasm, and extracellular vesicle are 0.635, 0.547, and 0.735, respectively. The AUPRC for the nucleus, cytoplasm, and extracellular vesicle are 0.740, 0.691, and 0.293, respectively. Moreover, we produced receiver operating characteristic (ROC) curves and precision-recall (PR) curves to visually illustrate the predictive capabilities of our model across diverse subcellular regions, as depicted in Fig. 3B and C. When evaluating the model on the testing set, the AUROC values obtained for the nucleus and cytoplasm were 0.700 and 0.484, respectively.

**中文:** 继续我们之前的lncRNA亚细胞定位预测模型，即iLoc-lncRNA和iLoc-lncRNA 2.0，当前模型命名为iLoc-lncRNA-BERT。iLoc-lncRNA-BERT的详细结果在表2中概述。细胞核、细胞质和细胞外囊泡的AUROC分别为0.635、0.547和0.735。细胞核、细胞质和细胞外囊泡的AUPRC分别为0.740、0.691和0.293。此外，我们产生了接收者操作特征（ROC）曲线和精确召回（PR）曲线以可视化说明我们模型在不同亚细胞区域的预测能力，如图3B和C所示。在测试集上评估模型时，细胞核和细胞质获得的AUROC值分别为0.700和0.484。

<a id="S014"></a>
**Source:** p.3 S014

**Original:** There are several machine learning-based models for lncRNA subcellular location prediction [12–14,16–24]. Some of them only reserved one label of a sample, and three works reserved multi-labels of lncRNA [14,15,24]. Here, we only compared our model with the three multi-label classification works. Wang et al. [14] collected 588 H. sapiens lncRNA with 771 location labels from the RNALocate database. They extracted k-mer nucleotide composition and k-mer variant features from lncRNA sequences. Converting the multi-label classification problem to a multi-class classification problem through a one-vs-rest strategy, they built 5 binary Hilbert-Schmidt independence criterion-multiple kernel support vector machine classifiers (MKSVM-HSIC) [49–51] for the ribosome, cytosol, nucleus, cytoplasm, and exosome, respectively. The average precision (AP), accuracy (Acc), and HL achieved 0.754, 0.418, and 0.069, respectively. Muhammad et al. [24] used the same data and presented a novel graph-based approach GeneticSeq2Vec to generate a rich statistical representation of lncRNA sequences. By combining an explainable Long Short-Term Memory (LSTM) [52] network with attention, their model, named HoEL-RMLocNet, achieved average precision, accuracy, and AUROC of 0.85, 0.55, and 0.766, respectively. Zeng et al. [15] compiled 811 H. sapiens lncRNA with various subcellular localizations, including nucleus, cytoplasm, chromatin, and insoluble cytoplasm, from the RNALocate database. They utilized eight transformer blocks to model long-range dependencies within the lncRNA sequence. Utilizing a localization-specific attention mechanism, their model, LncLocFormer, aimed at predicting multiple subcellular localizations simultaneously. The reported performance of LncLocFormer showcased a micro-Recall of 0.721 and a micro-AUROC of 0.648. As the dataset and code of MKSVM-HSIC and HoEL-RMLocNet are inaccessible, our comparison was only based on the reported model performance provided by them. Table 3 listed a performance comparison between our proposed model iLoc-lncRNA-BERT with these three published methods. It is obvious that our model performs better in micro-AUROC compared to the other methods.

**中文:** 有几种基于机器学习的lncRNA亚细胞定位预测模型[12-14,16-24]。其中一些仅保留样本的一个标签，三项工作保留了lncRNA的多标签[14,15,24]。在这里，我们仅将我们的模型与这三项多标签分类工作进行比较。Wang等人[14]从RNALocate数据库收集了588个具有771个位置标签的人类lncRNA。他们从lncRNA序列中提取了k-mer核苷酸组成和k-mer变异特征。通过一对一剩余策略将多标签分类问题转换为多类分类问题，他们分别为核糖体、细胞溶质、细胞核、细胞质和外泌体构建了5个二元希尔伯特-施密特独立性准则-多核支持向量机分类器（MKSVM-HSIC）[49-51]。平均精度（AP）、准确度（Acc）和HL分别达到0.754、0.418和0.069。Muhammad等人[24]使用相同的数据并提出了一种新颖的基于图的方法GeneticSeq2Vec来生成lncRNA序列的丰富统计表示。通过结合可解释的长短期记忆（LSTM）[52]网络与注意力，他们的模型HoEL-RMLocNet实现了0.85、0.55和0.766的平均精度、准确度和AUROC。Zeng等人[15]从RNALocate数据库编译了811个具有各种亚细胞定位的人类lncRNA，包括细胞核、细胞质、染色质和不溶性细胞质。他们利用八个变换器块来建模lncRNA序列内的长程依赖性。利用定位特异性注意力机制，他们的模型LncLocFormer旨在同时预测多个亚细胞定位。LncLocFormer的报告性能展示了0.721的微召回率和0.648的微AUROC。由于MKSVM-HSIC和HoEL-RMLocNet的数据集和代码不可访问，我们的比较仅基于他们提供的报告模型性能。表3列出了我们提出的模型iLoc-lncRNA-BERT与这三种已发表方法之间的性能比较。显然，与其他方法相比，我们的模型在微AUROC方面表现更好。

<a id="T003"></a>
### Table 3. Compare our model with existing tools
### 表3. 将我们的模型与现有工具进行比较

**Placed near:** p.3 S014  
**Source:** p.3 T003  

| Model | Acc | HL | micro-AUROC |
|---|---|---|---|
| iLoc-lncRNA-BERT | 0.353 | 0.267 | 0.791 |
| MKSVM-HSIC | 0.418 | 0.069 | / |
| HoEL-RMLocNet | 0.550 | / | 0.766 |
| LncLocFormer | 0.612 | / | 0.648 |

**中文表注:** iLoc-lncRNA-BERT与其他多标签分类模型的性能比较。

**Reading note:** 此表比较了iLoc-lncRNA-BERT与其他三种多标签分类模型的性能指标，显示我们的模型在微AUROC方面表现最佳。

<a id="S015"></a>
**Source:** p.3 S015

**Original:** Due to the low conservation of lncRNA between species, it is difficult to achieve cross-species prediction of subcellular localization of lncRNA between H. sapiens and M. musculus. To assess the generalization ability of our model constructed on H. sapiens data, we used M. musculus data to examine the proposed model's performance. Results in Table 2 showed that the macro-AUROC is 0.555. The AUROC for the nucleus, cytoplasm, and extracellular vesicle are 0.673, 0.409, and 0.585, respectively. The results imply that a certain disparity exists in subcellular localization mechanisms between H. sapiens and M. musculus. This necessitates the construction of separate subcellular localization prediction models for different species.

**中文:** 由于lncRNA在物种之间的保守性较低，很难实现人类和小鼠之间lncRNA亚细胞定位的跨物种预测。为了评估我们在人类数据上构建的模型的泛化能力，我们使用小鼠数据来检查提出模型的性能。表2中的结果显示宏AUROC为0.555。细胞核、细胞质和细胞外囊泡的AUROC分别为0.673、0.409和0.585。结果表明人类和小鼠之间的亚细胞定位机制存在一定差异。这需要为不同物种构建单独的亚细胞定位预测模型。

<a id="S016"></a>
**Source:** p.3 S016

**Original:** The Sensitive Thorough Rapid Enriched Motif Elicitation (STREME) algorithm was employed to identify known motifs enriched in inputted lncRNA sequences within the different subcellular regions [53]. It could discover ungapped motifs that are enriched in one type of sample. In this study, we performed differential motif enrichment analysis between lncRNAs located in nucleus/cytoplasm/extracellular vesicle and those located in non-nucleus/cytoplasm/extracellular vesicle regions. Due to the truncation of the 3′ end of lncRNA, right alignment was set. We got 3 motifs for the nucleus region, 3 motifs for the cytoplasm region, and 5 motifs for the extracellular region. The output motifs identified by STREME were shown in Fig. 4 and Table 4. Specifically, CAUGUUUUU, GUUUUCUCA, and GGGAAACAA were identified as RNA-binding motifs associated with the nucleus, while UGGCAGGA, GCCUCGGCC, and [C/T]GUCCA were linked to the cytoplasm. Additionally, GCUGGUCUUG, UCACCUGGGA, AGACAUGAG, GUCUUGUCU, and UCACAGAAUU were recognized as RNA-binding motifs associated with the extracellular vesicle. These findings offer insights into potential interactions between lncRNA and other RNA molecules.

**中文:** 敏感彻底快速富集基序提取（STREME）算法用于识别不同亚细胞区域内输入的lncRNA序列中富集的已知基序[53]。它可以发现富集在一种样本类型中的无间隙基序。在这项研究中，我们对定位于细胞核/细胞质/细胞外囊泡的lncRNA与定位于非细胞核/细胞质/细胞外囊泡区域的lncRNA进行了差异基序富集分析。由于lncRNA的3′端截断，设置了右对齐。我们为细胞核区域获得了3个基序，为细胞质区域获得了3个基序，为细胞外区域获得了5个基序。STREME识别的输出基序如图4和表4所示。具体而言，CAUGUUUUU、GUUUUCUCA和GGGAAACAA被识别为与细胞核相关的RNA结合基序，而UGGCAGGA、GCCUCGGCC和[C/T]GUCCA与细胞质相关。此外，GCUGGUCUUG、UCACCUGGGA、AGACAUGAG、GUCUUGUCU和UCACAGAAUU被识别为与细胞外囊泡相关的RNA结合基序。这些发现为lncRNA与其他RNA分子之间的潜在相互作用提供了见解。

<a id="F004"></a>
### Fig. 4. Visualization of the RNA-binding motifs generated from STREME
### 图4. STREME生成的RNA结合基序的可视化

**Placed near:** p.3 S016  
**Source:** p.4 C004  

![Fig. 4](bert_assets/page4_img1.jpeg)

**Original caption:** Visualization of the RNA-binding motifs generated from STREME.  
**中文图注:** STREME生成的RNA结合基序的可视化。

**Reading note:** 此图展示了通过STREME算法识别的不同亚细胞区域的RNA结合基序。

<a id="T004"></a>
### Table 4. The details of STREME motifs
### 表4. STREME基序的详细信息

**Placed near:** p.3 S016  
**Source:** p.3 T004  

| CONSENSUS | WIDTH | SITES | SEA_PVALUE | EVALUE | Score |
|---|---|---|---|---|---|
| **Nucleus** | | | | | |
| CAUGUUUUU | 9 | 26 | 6.3e−001 | 1.9e+000 | / |
| GUUUUCUCA | 9 | 29 | 7.7e−001 | 2.3e+000 | / |
| GGGAAACAA | 9 | 48 | 1.0e+000 | 3.0e+000 | / |
| **Cytoplasm** | | | | | |
| UGGCAGGA | 8 | 43 | 3.1e−001 | 9.4e−001 | / |
| GCCUCGGCC | 9 | 63 | 3.1e−001 | 9.4e−001 | / |
| YGUCCA | 6 | 98 | 7.9e−001 | 2.4e+000 | / |
| **Extracellular vesicle** | | | | | |
| GCUGGUCUUG | 10 | 6 | / | / | 1.3e−007 |
| UCACCUGGGA | 10 | 7 | / | / | 2.4e−007 |
| AGACAUGAG | 9 | 5 | / | / | 2.1e−006 |
| GUCUUGUCU | 9 | 5 | / | / | 2.1e−006 |
| UCACAGAAUU | 10 | 5 | / | / | 2.1e−006 |

**中文表注:** 不同亚细胞区域识别的STREME基序的详细信息，包括共识序列、宽度、位点数、P值和E值。

**Reading note:** 此表详细列出了不同亚细胞区域识别的RNA结合基序及其统计显著性。

---

## 4. Discussion and conclusion / 讨论与结论

<a id="S017"></a>
**Source:** p.3 S017

**Original:** Different subcellular locations enable lncRNAs to carry out specific functions, contributing to the complex regulatory networks within the cell. Artificial intelligence methodologies can assist in constructing models to address biological inquiries and explore the mechanisms of biological phenomena. This study aims to develop a computational model for predicting lncRNA subcellular localization. We collected data on H. sapiens lncRNA subcellular localization and utilized the widely acclaimed BERT pre-trained algorithm to establish the predictive model. Although this model has achieved promising predictive performance, achieving a micro-AUROC of 0.791, due to limited experimentally validated data for model training, there is still room for improvement in predicting the multi-label subcellular locations of lncRNAs. Moreover, the lower recognition rate of the model on M. musculus data suggests the importance of accounting for species-specific variations when investigating subcellular localization. Researchers employing predictive models for mammalian lncRNA subcellular localization should interpret outcomes with caution. The challenge of unraveling the biological and disease-related functions of the vast number of lncRNAs introduces an acute need for techniques to elucidate their subcellular localization. To facilitate peer comparison and improve lncRNA subcellular localization prediction models, we have made available the data and code for iLoc-lncRNA-BERT at https://github.com/ZhaoyueZhang/iLoc-lncRNA-BERT. Our work contributes a dependable dataset and an innovative model to advance the study of lncRNA subcellular localization, providing valuable insights for future research considerations.

**中文:** 不同的亚细胞定位使lncRNA能够执行特定功能，有助于细胞内复杂的调控网络。人工智能方法可以帮助构建模型来解决生物学探究并探索生物学现象的机制。本研究旨在开发一个计算模型来预测lncRNA亚细胞定位。我们收集了人类lncRNA亚细胞定位数据，并利用广泛赞誉的BERT预训练算法建立预测模型。尽管该模型已取得有希望的预测性能，实现了0.791的微AUROC，但由于用于模型训练的实验验证数据有限，在预测lncRNA的多标签亚细胞定位方面仍有改进空间。此外，模型在小鼠数据上的较低识别率表明在研究亚细胞定位时考虑物种特异性变异的重要性。使用预测模型进行哺乳动物lncRNA亚细胞定位的研究人员应谨慎解释结果。揭示大量lncRNA的生物学和疾病相关功能的挑战引入了对阐明其亚细胞定位技术的迫切需求。为了促进同行比较并改进lncRNA亚细胞定位预测模型，我们在https://github.com/ZhaoyueZhang/iLoc-lncRNA-BERT提供了iLoc-lncRNA-BERT的数据和代码。我们的工作为推进lncRNA亚细胞定位研究贡献了可靠的数据集和创新模型，为未来的研究考虑提供了有价值的见解。

---

## Supplementary Information / 补充信息

<a id="S018"></a>
**Source:** p.3 S018

**Original:** Supplementary data to this article can be found online at https://doi.org/10.1016/j.ijbiomac.2024.130659.

**中文:** 本文的补充数据可在https://doi.org/10.1016/j.ijbiomac.2024.130659在线找到。

---

## Author Contributions / 作者贡献

<a id="S019"></a>
**Source:** p.3 S019

**Original:** Zheng Zhang: Validation, Software, Methodology. Xiucai Ye: Writing – original draft, Supervision, Methodology, Conceptualization. Tetsuya Sakurai: Validation, Supervision, Formal analysis. Hao Lin: Writing – review & editing, Supervision, Project administration, Conceptualization.

**中文:** Zheng Zhang：验证、软件、方法学。Xiucai Ye：写作-初稿、监督、方法学、概念化。Tetsuya Sakurai：验证、监督、形式分析。Hao Lin：写作-审查与编辑、监督、项目管理、概念化。

---

## Declaration of Competing Interest / 利益冲突声明

<a id="S020"></a>
**Source:** p.3 S020

**Original:** The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**中文:** 作者声明他们没有已知的竞争性经济利益或个人关系可能影响本文报告的工作。

---

## Acknowledgment / 致谢

<a id="S021"></a>
**Source:** p.3 S021

**Original:** This work was supported by the grant from National Natural Science Foundation of China [62102067] and [62250028]; JST SPRING [Grant Number JPMJSP2124]; the JSPS KAKENHI [Grant Number JP23H03411] and [Grant Number JP22K12144]; and the JST [Grant Number JPMJPF2017].

**中文:** 本研究得到了国家自然科学基金[62102067]和[62250028]的资助；JST SPRING[资助号JPMJSP2124]；JSPS KAKENHI[资助号JP23H03411]和[资助号JP22K12144]；以及JST[资助号JPMJPF2017]的支持。

---

## References / 参考文献

<a id="S022"></a>
**Source:** p.3 S022

**Original:** [1] T. Ali, P. Grote, Beyond the RNA-dependent function of LncRNA genes, Elife (2020) 9.  
[2] R. Wang, et al., DeepBIO: an automated and interpretable deep-learning platform for high-throughput biological sequence prediction, functional annotation and visualization analysis, Nucleic Acids Res. 51 (7) (2023) 3017–3029.  
[3] T. Isoda, et al., Non-coding transcription instructs chromatin folding and compartmentalization to dictate enhancer-promoter communication and T cell fate, Cell 171 (1) (2017) 103–119 e18.  
[4] L. Cao, et al., LAST, a c-Myc-inducible long noncoding RNA, cooperates with CNBP to promote CCND1 mRNA stability in human cells, Elife (2017) 6.  
[5] J. Carlevaro-Fita, R. Johnson, Global positioning system: understanding long noncoding RNAs through subcellular localization, Mol. Cell 73 (5) (2019) 869–883.  
[6] W.P. Kloosterman, et al., In situ detection of miRNAs in animal embryos using LNA-modified oligonucleotide probes, Nat. Methods 3 (1) (2006) 27–29.  
[7] J. Ye, et al., Research advances in the detection of miRNA, J. Pharm. Anal. 9 (4) (2019) 217–226.  
[8] X. Zou, et al., Accurately identifying hemagglutinin using sequence information and machine learning methods, Front. Med. (Lausanne) 10 (2023) 1281880.  
[9] W. Zhu, et al., A first computational frame for recognizing heparin-binding protein, Diagnostics (Basel) 13 (14) (2023).  
[10] J. Jin, et al., iDNA-ABF: multi-scale deep biological language learning model for the interpretable prediction of DNA methylations, Genome Biol. 23 (1) (2022) 1–23.  
[11] H. Li, Y. Pang, B. Liu, BioSeq-BLM: a platform for analyzing DNA, RNA, and protein sequences based on biological language models, Nucleic Acids Res. 49 (22) (2021) e129.  
[12] Z. Cao, et al., The lncLocator: a subcellular localization predictor for long non-coding RNAs based on a stacked ensemble classifier, Bioinformatics 34 (13) (2018) 2185–2194.  
[13] M. Zeng, et al., DeepLncLoc: a deep learning framework for long non-coding RNA subcellular localization prediction based on subsequence embedding, Brief. Bioinform. 23 (1) (2022).  
[14] H. Wang, et al., Identify RNA-associated subcellular localizations based on multi-label learning using Chou's 5-steps rule, BMC Genomics 22 (1) (2021).  
[15] M. Zeng, et al., LncLocFormer: a transformer-based deep learning model for multi-label lncRNA subcellular localization prediction by using localization-specific attention mechanism, Bioinformatics 39 (12) (2023).  
[16] B.L. Gudenas, L. Wang, Prediction of LncRNA subcellular localization with deep learning from sequence features, Sci. Rep. 8 (1) (2018) 16385.  
[17] A. Ahmad, H. Lin, S. Shatabda, Locate-R: subcellular localization of long non-coding RNAs using nucleotide compositions, Genomics 112 (3) (2020) 2583–2589.  
[18] X.F. Yang, et al., Predicting LncRNA subcellular localization using unbalanced pseudo-k nucleotide compositions, Curr. Bioinforma. 15 (6) (2020) 554–562.  
[19] M. Li, et al., GraphLncLoc: long non-coding RNA subcellular localization prediction using graph convolutional networks based on sequence to graph transformation, Brief. Bioinform. 24 (1) (2023).  
[20] Y.X. Fan, M.J. Chen, Q.Q. Zhu, lncLocPred: predicting LncRNA subcellular localization using multiple sequence feature information, Ieee Access 8 (2020) 124702–124711.  
[21] S. Zhang, H. Qiao, KD-KLNMF: identification of lncRNAs subcellular localization with multiple features and nonnegative matrix factorization, Anal. Biochem. 610 (2020) 113995.  
[22] Z.D. Su, et al., iLoc-lncRNA: predict the subcellular location of lncRNAs by incorporating octamer composition into general PseKNC, Bioinformatics 34 (24) (2018) 4196–4204.  
[23] Z.Y. Zhang, et al., Towards a better prediction of subcellular location of long non-coding RNA, Front. Comput. Sci. 16 (5) (2022).  
[24] M.N. Asim, et al., EL-RMLocNet: an explainable LSTM network for RNA-associated multi-compartment localization prediction, Comput. Struct. Biotechnol. J. 20 (2022) 3986–4002.  
[25] J. Ding, et al., A multi-scale multi-model deep neural network via ensemble strategy on high-throughput microscopy image for protein subcellular localization, Expert Syst. Appl. 212 (2023).  
[26] H. Zhou, et al., Identify ncRNA subcellular localization via graph regularized k-local hyperplane distance nearest neighbor model on multi-kernel learning, IEEE/ACM Trans. Comput. Biol. Bioinform. 19 (6) (2022) 3517–3529.  
[27] Q. Zou, et al., Gene2vec: gene subsequence embedding for prediction of mammalian N6-methyladenosine sites from mRNA, Rna 25 (2) (2019) 205–218.  
[28] B. Liu, X. Gao, H. Zhang, BioSeq-Analysis2.0: an updated platform for analyzing DNA, RNA and protein sequences at sequence level and residue level based on machine learning approaches, Nucleic Acids Res. 47 (20) (2019) e127.  
[29] H.V. Tran, Q.H. Nguyen, iAnt: combination of convolutional neural network and random forest models using PSSM and BERT features to identify antioxidant proteins, Curr. Bioinforma. 17 (2) (2022) 184–195.  
[30] T. Cui, et al., RNALocate v2.0: an updated resource for RNA subcellular localization with increased coverage and annotation, Nucleic Acids Res. 50 (D1) (2022) D333–D339.  
[31] E.W. Sayers, et al., GenBank 2023 update, Nucleic Acids Res. 51 (D1) (2023) D141–D144.  
[32] Y.H. Yang, et al., i2OM: toward a better prediction of 2′-O-methylation in human RNA, Int. J. Biol. Macromol. 239 (2023).  
[33] W. Su, et al., iRNA-ac4C: a novel computational method for effectively detecting N4-acetylcytidine sites in human mRNA, Int. J. Biol. Macromol. 227 (2023) 1174–1181.  
[34] L. Chen, L. Yu, L. Gao, Potent antibiotic design via guided search from antibacterial activity evaluations, Bioinformatics 39 (2) (2023) btad059.  
[35] Y. Tang, Y. Pang, B. Liu, IDP-Seq2Seq: identification of intrinsically disordered regions based on sequence to sequence learning, Bioinformatics 36 (21) (2021) 5177–5186.  
[36] K. Yan, et al., sAMPpred-GAT: prediction of antimicrobial peptide by graph attention network and predicted peptide structure, Bioinformatics 39 (1) (2023) btac715.  
[37] H. Yang, et al., A gender specific risk assessment of coronary heart disease based on physical examination data, NPJ Digit. Med. 6 (1) (2023) 136.  
[38] X.W. Liu, et al., iPADD: a computational tool for predicting potential antidiabetic drugs using machine learning algorithms, J. Chem. Inf. Model. 63 (15) (2023) 4960–4969.  
[39] Y.H. Yang, et al., DeepIDC: a prediction framework of injectable drug combination based on heterogeneous information and deep learning, Clin. Pharmacokinet. 61 (12) (2022) 1749–1759.  
[40] Y.Y. Chen, et al., Deep generative model for drug design from protein target sequence, J. Cheminf. 15 (1) (2023).  
[41] X. Zeng, et al., Accurate prediction of molecular properties and drug targets using a self-supervised image representation learning framework, Nat. Mach. Intell. 4 (11) (2022) 1004–1016.  
[42] J. Xu, et al., Graph embedding and Gaussian mixture variational autoencoder network for end-to-end analysis of single-cell RNA sequencing data, Cell Rep. Methods 3 (1) (2023).  
[43] X. Pan, et al., Deep learning for drug repurposing: methods, databases, and applications, Wiley Interdiscip. Rev.: Comput. Mol. Sci. 12 (4) (2022) e1597.  
[44] S. Zhao, et al., AP-BERT: enhanced pre-trained model through average pooling, Appl. Intell. 52 (14) (2022) 15929–15937.  
[45] S. Zhao, et al., Augment BERT with average pooling layer for Chinese summary generation, J. Intell. Fuzzy Syst. 42 (3) (2022) 1859–1868.  
[46] Y. Ji, et al., DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome, Bioinformatics 37 (15) (2021) 2112–2120.  
[47] T. Kobayashi, Two-way multi-label loss, in: 2023 Ieee/Cvf Conference on Computer Vision and Pattern Recognition, Cvpr, 2023, pp. 7476–7485.  
[48] Z.Y. Zhang, et al., iLoc-miRNA: extracellular/intracellular miRNA prediction using deep BiLSTM with attention mechanism, Brief. Bioinform. 23 (5) (2022).  
[49] R. Qi, F. Guo, Q. Zou, String kernels construction and fusion: a survey with bioinformatics application, Front. Comp. Sci. 16 (6) (2022) 166904.  
[50] Y. Zou, et al., FTWSVM-SR: DNA-binding proteins identification via fuzzy twin support vector machines on self-representation, Interdiscip. Sci. Comput. Life Sci. 14 (2) (2022) 372–384.  
[51] Y. Wang, Y. Zhai, Y. Ding, Q. Zou, SBSM-Pro: Support Bio-sequence Machine for Proteins, 2023 (p. arXiv:2308.10275).  
[52] J. Chen, Q. Zou, J. Li, DeepM6ASeq-EL: prediction of human N6-methyladenosine (m6A) sites with LSTM and ensemble learning, Front. Comput. Sci. 16 (2) (2022) 162302.  
[53] T.L. Bailey, STREME: accurate and versatile sequence motif discovery, Bioinformatics (2021).

**中文:** [1] T. Ali, P. Grote, 超越LncRNA基因的RNA依赖功能, Elife (2020) 9.  
[2] R. Wang, 等, DeepBIO: 用于高通量生物序列预测、功能注释和可视化分析的自动化可解释深度学习平台, Nucleic Acids Res. 51 (7) (2023) 3017–3029.  
[3] T. Isoda, 等, 非编码转录指导染色质折叠和区室化以决定增强子-启动子通讯和T细胞命运, Cell 171 (1) (2017) 103–119 e18.  
[4] L. Cao, 等, LAST, 一种c-Myc诱导的长链非编码RNA, 与CNBP合作促进人类细胞中CCND1 mRNA稳定性, Elife (2017) 6.  
[5] J. Carlevaro-Fita, R. Johnson, 全球定位系统: 通过亚细胞定位理解长链非编码RNA, Mol. Cell 73 (5) (2019) 869–883.  
[6] W.P. Kloosterman, 等, 使用LNA修饰寡核苷酸探针原位检测动物胚胎中的miRNA, Nat. Methods 3 (1) (2006) 27–29.  
[7] J. Ye, 等, miRNA检测研究进展, J. Pharm. Anal. 9 (4) (2019) 217–226.  
[8] X. Zou, 等, 使用序列信息和机器学习方法准确识别血凝素, Front. Med. (Lausanne) 10 (2023) 1281880.  
[9] W. Zhu, 等, 识别肝素结合蛋白的第一个计算框架, Diagnostics (Basel) 13 (14) (2023).  
[10] J. Jin, 等, iDNA-ABF: 用于DNA甲基化可解释预测的多尺度深度生物语言学习模型, Genome Biol. 23 (1) (2022) 1–23.  
[11] H. Li, Y. Pang, B. Liu, BioSeq-BLM: 基于生物语言模型分析DNA、RNA和蛋白质序列的平台, Nucleic Acids Res. 49 (22) (2021) e129.  
[12] Z. Cao, 等, lncLocator: 基于堆叠集成分类器的长链非编码RNA亚细胞定位预测器, Bioinformatics 34 (13) (2018) 2185–2194.  
[13] M. Zeng, 等, DeepLncLoc: 基于子序列嵌入的长链非编码RNA亚细胞定位预测深度学习框架, Brief. Bioinform. 23 (1) (2022).  
[14] H. Wang, 等, 使用Chou五步规则基于多标签学习识别RNA相关亚细胞定位, BMC Genomics 22 (1) (2021).  
[15] M. Zeng, 等, LncLocFormer: 使用定位特异性注意力机制的多标签lncRNA亚细胞定位预测基于变换器的深度学习模型, Bioinformatics 39 (12) (2023).  
[16] B.L. Gudenas, L. Wang, 使用序列特征深度学习预测LncRNA亚细胞定位, Sci. Rep. 8 (1) (2018) 16385.  
[17] A. Ahmad, H. Lin, S. Shatabda, Locate-R: 使用核苷酸组成的长链非编码RNA亚细胞定位, Genomics 112 (3) (2020) 2583–2589.  
[18] X.F. Yang, 等, 使用不平衡伪k核苷酸组成预测LncRNA亚细胞定位, Curr. Bioinforma. 15 (6) (2020) 554–562.  
[19] M. Li, 等, GraphLncLoc: 基于序列到图转换的图卷积网络长链非编码RNA亚细胞定位预测, Brief. Bioinform. 24 (1) (2023).  
[20] Y.X. Fan, M.J. Chen, Q.Q. Zhu, lncLocPred: 使用多种序列特征信息预测LncRNA亚细胞定位, Ieee Access 8 (2020) 124702–124711.  
[21] S. Zhang, H. Qiao, KD-KLNMF: 使用多种特征和非负矩阵分解识别lncRNA亚细胞定位, Anal. Biochem. 610 (2020) 113995.  
[22] Z.D. Su, 等, iLoc-lncRNA: 通过将八聚体组成纳入一般PseKNC预测lncRNA亚细胞定位, Bioinformatics 34 (24) (2018) 4196–4204.  
[23] Z.Y. Zhang, 等, 更好地预测长链非编码RNA的亚细胞定位, Front. Comput. Sci. 16 (5) (2022).  
[24] M.N. Asim, 等, EL-RMLocNet: 用于RNA相关多区室定位预测的可解释LSTM网络, Comput. Struct. Biotechnol. J. 20 (2022) 3986–4002.  
[25] J. Ding, 等, 通过高通量显微镜图像上的集成策略的多尺度多模型深度神经网络用于蛋白质亚细胞定位, Expert Syst. Appl. 212 (2023).  
[26] H. Zhou, 等, 通过多核学习上的图正则化k局部超平面距离最近邻模型识别ncRNA亚细胞定位, IEEE/ACM Trans. Comput. Biol. Bioinform. 19 (6) (2022) 3517–3529.  
[27] Q. Zou, 等, Gene2vec: 用于从mRNA预测哺乳动物N6-甲基腺苷位点的基因子序列嵌入, Rna 25 (2) (2019) 205–218.  
[28] B. Liu, X. Gao, H. Zhang, BioSeq-Analysis2.0: 基于机器学习方法在序列水平和残基水平分析DNA、RNA和蛋白质序列的更新平台, Nucleic Acids Res. 47 (20) (2019) e127.  
[29] H.V. Tran, Q.H. Nguyen, iAnt: 使用PSSM和BERT特征的卷积神经网络和随机森林模型组合识别抗氧化蛋白, Curr. Bioinforma. 17 (2) (2022) 184–195.  
[30] T. Cui, 等, RNALocate v2.0: 具有增加覆盖率和注释的RNA亚细胞定位更新资源, Nucleic Acids Res. 50 (D1) (2022) D333–D339.  
[31] E.W. Sayers, 等, GenBank 2023更新, Nucleic Acids Res. 51 (D1) (2023) D141–D144.  
[32] Y.H. Yang, 等, i2OM: 更好地预测人类RNA中的2′-O-甲基化, Int. J. Biol. Macromol. 239 (2023).  
[33] W. Su, 等, iRNA-ac4C: 有效检测人类mRNA中N4-乙酰胞苷位点的新计算方法, Int. J. Biol. Macromol. 227 (2023) 1174–1181.  
[34] L. Chen, L. Yu, L. Gao, 通过从抗菌活性评估指导搜索设计强效抗生素, Bioinformatics 39 (2) (2023) btad059.  
[35] Y. Tang, Y. Pang, B. Liu, IDP-Seq2Seq: 基于序列到序列学习识别内在无序区域, Bioinformatics 36 (21) (2021) 5177–5186.  
[36] K. Yan, 等, sAMPpred-GAT: 通过图注意力网络和预测肽结构预测抗菌肽, Bioinformatics 39 (1) (2023) btac715.  
[37] H. Yang, 等, 基于体检数据的冠心病性别特异性风险评估, NPJ Digit. Med. 6 (1) (2023) 136.  
[38] X.W. Liu, 等, iPADD: 使用机器学习算法预测潜在抗糖尿病药物的计算工具, J. Chem. Inf. Model. 63 (15) (2023) 4960–4969.  
[39] Y.H. Yang, 等, DeepIDC: 基于异构信息和深度学习的可注射药物组合预测框架, Clin. Pharmacokinet. 61 (12) (2022) 1749–1759.  
[40] Y.Y. Chen, 等, 从蛋白质靶标序列设计药物的深度生成模型, J. Cheminf. 15 (1) (2023).  
[41] X. Zeng, 等, 使用自监督图像表示学习框架准确预测分子性质和药物靶标, Nat. Mach. Intell. 4 (11) (2022) 1004–1016.  
[42] J. Xu, 等, 图嵌入和高斯混合变分自编码器网络用于单细胞RNA测序数据的端到端分析, Cell Rep. Methods 3 (1) (2023).  
[43] X. Pan, 等, 深度学习用于药物重定位: 方法、数据库和应用, Wiley Interdiscip. Rev.: Comput. Mol. Sci. 12 (4) (2022) e1597.  
[44] S. Zhao, 等, AP-BERT: 通过平均池化增强的预训练模型, Appl. Intell. 52 (14) (2022) 15929–15937.  
[45] S. Zhao, 等, 用平均池化层增强BERT用于中文摘要生成, J. Intell. Fuzzy Syst. 42 (3) (2022) 1859–1868.  
[46] Y. Ji, 等, DNABERT: 用于基因组中DNA语言的预训练双向编码器表示来自变换器模型, Bioinformatics 37 (15) (2021) 2112–2120.  
[47] T. Kobayashi, 双向多标签损失, in: 2023 Ieee/Cvf Conference on Computer Vision and Pattern Recognition, Cvpr, 2023, pp. 7476–7485.  
[48] Z.Y. Zhang, 等, iLoc-miRNA: 使用深度BiLSTM和注意力机制预测细胞外/细胞内miRNA, Brief. Bioinform. 23 (5) (2022).  
[49] R. Qi, F. Guo, Q. Zou, 字符串核构建和融合: 生物信息学应用的综述, Front. Comp. Sci. 16 (6) (2022) 166904.  
[50] Y. Zou, 等, FTWSVM-SR: 通过自表示上的模糊双支持向量机识别DNA结合蛋白, Interdiscip. Sci. Comput. Life Sci. 14 (2) (2022) 372–384.  
[51] Y. Wang, Y. Zhai, Y. Ding, Q. Zou, SBSM-Pro: 蛋白质支持生物序列机器, 2023 (p. arXiv:2308.10275).  
[52] J. Chen, Q. Zou, J. Li, DeepM6ASeq-EL: 使用LSTM和集成学习预测人类N6-甲基腺苷（m6A）位点, Front. Comput. Sci. 16 (2) (2022) 162302.  
[53] T.L. Bailey, STREME: 准确通用的序列基序发现, Bioinformatics (2021).

---

**Translation completed successfully.**  
**翻译成功完成。**

**Generated files:**  
**生成的文件:**  
- `E:\mRNA\Doc\bert_paper.md` - Bilingual Markdown document  
- `E:\mRNA\Doc\bert_assets/` - Extracted figures (7 images)  
- `E:\mRNA\Doc\extracted_text_bert_new.txt` - Extracted text content  

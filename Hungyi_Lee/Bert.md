# 【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介

[link](https://www.youtube.com/watch?v=gh0hewYkjgo)<br>

一堆文章，进行情感分析：如果有标注，有很多方法

self-supervised learning: 自己想办法做supervised，在没有label的情况下

一堆文章，一部分作为输入，一部分作为标注
可以把self-supervised learning看作unsupervised的一部嗯
### Masking input

- Bert使用乐transformer的encoder架构
- 输入一排向量，输出一排向量,即输入一段sequence，输出一段sequence,输入多长，输出多长
- 输入一段文字，Randomly masking some tokens<br>
> mask的方法（随机地不变/选择1或2）：<br>
1. 换成一个不存在的special token,  自造字
2. 随机将其换成另一个字符 复旦大学 -> 复旦小学
- 对每个被 mask 的位置，取对应输出向量，经过线性层（通常映射到词表大小） + softmax，计算与真实 token 的交叉熵；只对这些位置累加 MLM 损失。因此 MLM 在该意义上是对词表的分类任务。

- linear transform 需要linear model 也有参数需要训练
>什么是 Encoder（在 Transformer / BERT 中）: Transformer 的 encoder 是由若干相同层堆叠而成，每层包含：<br>
   - 多头自注意力（multi-head self-attention） + -残差连接 + LayerNorm，<br>
- 前馈网络（position-wise feed-forward） + 残差连接 + LayerNorm。<br>
- Encoder 输入是 token 的 embedding（加上位置编码），输出是一组与输入长度相同的“上下文化”向量（每个位置的向量都编码了该位置相对于整段序列的上下文）。BERT 就是只使用这种 encoder 堆栈（双向/非自回归地同时看到左右上下文），不是包含 decoder 的完整 Transformer。



## BERT and Linear Transform for NLP Tasks

### [CLS] and [SEP] Token Usage in BERT

* BERT模型通常在每个句子的开头加上一个 `[CLS]` 标记，在句子之间加上 `[SEP]` 标记。例如：

  ```
  [CLS] sentence 1 [SEP] sentence 2
  ```
* 经过 BERT 处理后，`[CLS]` 部分的输出用于后续任务（如分类等）。

### SOP (Sentence Order Prediction) - Used in ALBERT

* **任务描述**：判断两个句子的顺序是否正确。

  * 训练时，输入的句子对 `sentence1` 和 `sentence2` 是相邻的，需要判断它们的顺序。
  * 在预训练阶段被称为填空题（Masked Language Modeling）。
  * 在 Downstream Tasks 中：

    * 我们有少量标注数据进行微调（Fine-tuning）。

### Fine-tuning and Linear Transformation

* **BERT + Linear Transformation**：对 BERT 和线性部分进行微调。

  * **Case 1: Sentiment Analysis**

    * 使用预训练的 BERT，Linear 层随机初始化。
    * 输入：`[CLS] sentence1 sentence2`
    * 输出：通过 `[CLS]` 部分来判断类别，使用梯度下降进行优化。

  * **Case 2: POS Tagging (Part of Speech Tagging)**

    * 特点：输入序列与输出序列长度应相同。
    * 输入：`[CLS] w1 w2 w3`
    * 每个单词通过 BERT 后，再通过 Linear 层转化，最后通过 softmax 输出标签。

  * **Case 3: Natural Language Inference (NLI)**

    * 输入两个句子，输出一个分类（Contradiction, Entailment, Neutral）。
    * 输入：`[CLS] w1 w2 [SEP] w3 w4`
    * 经过 BERT 后，使用 Linear 层生成分类。
    * 最终，我们只使用 `[CLS]` 对应的输出作为类别判断。

  * **Case 4: Extraction-based Question Answering**

    * **任务描述**：在给定的文档中，回答基于查询的问题。答案一定来自文档中的一部分。
    * 输入：文档 `D = {d_1, ..., d_N}`，查询 `Q = {q_1, ..., q_M}`。
    * 输出：两个整数 `(s, e)`，表示答案在文档中的起始和结束位置。
    * 输入格式：`[CLS] q1 q2 [SEP] d1 d2 d3`
    * **处理流程**：

      1. 使用 BERT 提取特征。
      2. 分别使用两个向量来与文档和查询内容进行内积计算：

         * 一个向量与文档的d经过bert后的结果进行内积，再用softmax确定答案的起始位置。
         * 另一个向量与查询的q经过bert后的结果进行内积，再用softmax确定答案的结束位置。



---



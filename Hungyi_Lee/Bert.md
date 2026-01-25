# 【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介

## Basic info

[link](https://www.youtube.com/watch?v=gh0hewYkjgo)<br>
<p>
一堆文章，进行情感分析：如果有标注，有很多方法<br>

self-supervised learning: 自己想办法做supervised，在没有label的情况下

一堆文章，一部分作为输入，一部分作为标注<br>
可以把self-supervised learning看作unsupervised的一部分
> *Token 级（MLM）：在同一序列中随机把若干 token 替成 [MASK]（或有时隨機替換/保留），模型輸入是被遮蔽的整段序列，目標是預測被 mask 的原詞；損失只在被 mask 的位置計算。[填空题]*<br>
>*句子／段落級（SOP/NSP 類）：把一個句子當作輸入，把另一個句子（或句子關係，如是否為下句）當作目標，任務不是填單詞而是判斷句子順序或關係。[句子关系判断题]*<br>
<p>


## Masking input

- Bert使用了ransformer的encoder架构
- 输入一排向量，输出一排向量,即输入一段sequence，输出一段sequence,输入多长，输出多长
- 输入一段文字，Randomly masking some tokens<br>
> mask的方法（随机地不变/选择1或2）：<br>
1. 换成一个不存在的special token,  自造字
2. 随机将其换成另一个字符 复旦大学 -> 复旦小学
- 对每个被 mask 的位置，取对应输出向量，经过线性层（通常映射到词表大小） + softmax 【即MLM-head】，计算与真实 token 的交叉熵；只对这些位置累加 MLM 损失。因此 MLM 在该意义上是对词表的分类任务。

- linear transform 需要linear model 也有参数需要训练
>什么是 Encoder（在 Transformer / BERT 中）: Transformer 的 encoder 是由若干相同层堆叠而成，每层包含：<br>
   - 多头自注意力（multi-head self-attention） + -残差连接 + LayerNorm，<br>
- 前馈网络（position-wise feed-forward） + 残差连接 + LayerNorm。<br>
- Encoder 输入是 token 的 embedding（加上位置编码），输出是一组与输入长度相同的“上下文化”向量（每个位置的向量都编码了该位置相对于整段序列的上下文）。BERT 就是只使用这种 encoder 堆栈（双向/非自回归地同时看到左右上下文），不是包含 decoder 的完整 Transformer。

- 最大化正确词概率的交叉熵 $\leftrightarrow$ 让 hidden vector 与正确词 embedding 的内积变大,与错误词 embedding 的内积变小。<br>
  优化目标是 最大化正确词的内积，最小化错误词的内积<br>

- 模型进行运算得到隐藏层hidden vector, 即上下文推断出的词向量，
> I @ apples.  @是被mask的东东，先是简单地embed到embedding空间，其经过（L次）encode，得到向量$h$（是由I，apples，.，与位置一起计算出来的）<br>
> hidden vector —— 一个“语义需求说明书”
>此时的 h_mask：<br>
不是 eat<br>
不是 like<br>
不是任何具体词<br>
它更像一句话：<br>
“我需要一个<br>
动词<br>
能和 I 当主语<br>
能和 apples 当宾语<br>
常见、自然、合理<br>
- 再MLM head ，将hidden vector 投影到 embedding 空间，输出 logits—— 把“需求”翻译成“词表语言”，在embedding层去【内积】<br>
> h_mask $\rightarrow$ Dense + bias $\rightarrow$ GELU $\rightarrow$ LayerNorm $\rightarrow$ Dense（与 embedding 权重共享）$\rightarrow$ logits（每个词一个分数）【这是MLM-head】<br>
-  上面的三个部分，embed，encode，MLM-head都会在训练中不断地【反向传播】优化。


## BERT and Linear Transform for NLP Tasks

### [CLS] and [SEP] Token Usage in BERT

* BERT模型通常在每个句子的开头加上一个 `[CLS]` 标记，在句子之间加上 `[SEP]` 标记。例如：

  ```
  [CLS] sentence 1 [SEP] sentence 2
  ```
* 经过 BERT 处理后，`[CLS]` 部分的输出用于后续任务（如分类等）。

### SOP (Sentence Order Prediction) - Used in ALBERT

* **任务描述**: 判断下一句话是不是原文中紧跟在第一句话后面(Next Sentence Prediction)*太弱了*
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


## Why does BERT work?

> You shall know a word by the company it keeps. ---John Rupert Firth

Bert: 在预测mask的过程中，上下文之间的语义能够被显现出来，由此看得出单个词/整个句子<br>
- contextualized word embedding<br>
- CBOW,一个曾经一层的线性模型，做过类似的工作，由于算力问题没做下去
> 一个词的“意思” = 它在“什么上下文中出现时是合理的”
>
> gpt warns:⚠️ BERT 并不是“理解情感”或“理解意思”
> 它只是学会了：<br>
> “在统计上，什么向量配置最有利于完成预测任务”<br>
但：这个统计结构恰好与我们人类的“语义 / 情感 / 逻辑”高度对齐，所以我们感觉它在“理解”
<p>
奇怪的是,<br>
用pre-trained on English的BERT，去在dna，protein,music上微调（不同的atcg，doremifasolaxi映射到随意但固定的英语单词，比如a是apple，t是table，c是cat，g是girl）然后做音乐，蛋白质，dna预测，他们的效果也比 bert那块是随机初始化而不是pretrainedonEnglish   好上不好。
<p>

> 好像bert是一种很好的初始化方法，或者是好的初始化参数，特别适合大模型训练

<p>
以及，<br>
multi-lingual pretrain出的bert，在英语上finetune问答(Case4那样子的)，【没有finetuen中文上的问答训练】，<br>
结果在中文问答上有不错的表现。
<p>

>【意思】在不同语言下是共同的，或者说向量的“样子”相似<br>
> alignment<br>
> 很依赖大规模训练数据<br>

<p>
但是，明明pre-trained的时候，是在multi-language下的，那为什么做mask的时候，它不会傻乎乎地把 I *** apple 填一个 “吃”进去呢（正确的是eat）？说明语言间还是有差别的吧？<br>
有人做了实验，比较中文词向量还有英文词向量，发现这个做差做出来的偏移向量是有点规律在里头的。<br>
取这个偏移向量的平均值，对于an English sentence(词向量化后), 加上这个平均后的偏移量，得到了 Chinglish(中英夹杂的)意思类似的话。
<p>

> BERT 的成功可能更多来自<br>“对序列中局部—全局依赖的高效建模方式”，
而不仅是自然语言本身。



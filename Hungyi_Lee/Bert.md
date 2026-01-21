#### 【機器學習2021】自督導式學習 (Self-supervised Learning) (二) – BERT簡介
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
### next sentence prediciton



[CLS]sentence 1 [sep] sentence 2
这串东西需要经过bert
希望[CLS]经历bert后，再linear transform 变成yes or no的分类， 然后说 sentence1 和 sentence2接不接在一起

但这个被证明用处不大，因为这个判断太简单了


SOP Sentence order prediction  , used in ALBERT
sentence1 sentence2 本就相邻，判断相对前后顺序

看似是训练他在填空题上，这里我们就叫它 Pre-train
但它在 Downstream Tasks
- The tasks we care
- We have a little bit labeled data

We Fine-tune后就很可以

https://www.youtube.com/watch?v=gh0hewYkjgo



# 详解Transformer （Attention Is All You Need）



## Transformer 实现变长的方式

对于Encoder 实现可变长的方式：

* 对于multi-head-attention 中的attention部分 本身可以做到可变长的方式；
* 在FFN网络中进行交互的是每一个单词向量的512维吗？换句话说，是让一个单词的各个维度进行交互而非不同单词的维度是吗？是的，每次进入全连接层的只能是一个样本，让其特征之间进行交互。有点类似于图像领域，他们让一张图片进入CNN网络，最后经过全连接层，让其在底层cnn卷积得到的特征之间进行组合，激活。得到最后的结果，就是判定这张照片属于什么。
* 对于add和norm 是进行每个input单独进行的，因此也可以实现变长的方式；

对于Decoder 实现可变长的方式：

* Decoder和Encoder的模块基本一致，只有最后一层output，和 memory attention部分是不同的；
* output其实做不到可变长，因此Decoder其实是不可变长的，用的是mask的方式进行；
* Decoder 是一个下三角的input，对于没有数值的地方进行补零处理；

## Transformer 的优缺点

优点：

* Decoder是一个语言模型，Encoder 和embedding是一个翻译模型，模型复杂度高，所以能处理的特征和效果会很好；
* 另外可以处理变长的问题、且有残差的处理方式，效果和局限度会好很多；
* 使用 attention的的方式，处理效果会更好，另外multi-head可以使提取的效果更佳；


## 参考文献
* https://www.zhihu.com/question/362131975/answer/2182682685
* https://www.zhihu.com/question/499274875/answer/2250085650
* https://zhuanlan.zhihu.com/p/69290203
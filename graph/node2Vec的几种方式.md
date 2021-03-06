## ALS、LDA和Word2Vec的相同点和不同之处

### 相同之处
[ALS](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)是将users-products之间的关系转化为，users-topic-products三者的关系，其中topic是潜层话题的意思。即映射成用户因为喜欢某一类话题，而一些商品中拥有一些话题的元素，所以用户通过喜欢话题而喜欢上某些商品。

[LDA](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)是将documents-words之间的关系转化为documents-topics-words三者之间的关系，其中topic是潜层的话题。即一篇文章所叙述的内容是为了表达一些话题，而为了表达这些话题，需要用了一些词。即将文章和词之间的关系，映射成文章和主题之间的关系+主题和词之间的关系。

Word2Vec的最初思想可能和上面的到是不一样，它的目标很明确，就是词是所有句子、段落、文章的最细粒度的表示，所以在希望用数学来处理NLP的时候，就希望能用数学里的方法来表示一个词，
那么需要做的就是词嵌入(word embedding)，其中最基本的做法就是one-hot encoder的形式，但是one-hot encoder的向量太大，所以就希望能进行无损的压缩，即对one-hot encoder的向量空间进行降维，
一般降到30-300左右。从Word2Vec得到的结果来看，其实也是一种document-topic的形式，而word2Vec的结果是当document就是单个词的时候，所对应的topic表示。

### ALS 算法简介
ALS是用于协同过滤中非常常见的算法。其基本思想如下：

* 根据一个users-products-rating的评分数据集，ALS需要建立一个 users * products的n*m的矩阵，(其中n为users的数量，m为prodocts的数量)
* 因为ALS基于的假设是用户-主题-商品，因此用户商品的R(n*m)得分矩阵可以表示为 U[n *k] * V[k*m]的矩阵
* 那得出这种假设之后，接下来的优化目标就很容易了，其中k为定植，因此E是一个固定的矩阵，即通过矩阵R 来求解出最优的 U和V
* 一种常见的解法(也是spark-ml的解法)是用最小二乘法来求解U和V：
    * 先为U和V中的任意一个矩阵，给出一个随机的结果，并固定；// 假设先固定V
    * 然后，在用 已知的R和V来求解该情况下最优的U，方法是[最小二乘法](https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95)求解
    * 再得到上一步的U之后，将U作为已知，用R和U来求解V
    * 然后依次迭代，直到最小二乘法结果的E满足一定条件，或者迭代次数完成之后，从而得到最终的 U和V。
* 得到的U即为用户对k个主题的喜好程度，V即为某个商品和k个主题之间的关系


### LDA 算法简介
LDA是概率主题模型，目的是为了将文档以主题概率分布的形式给出。可以用于做文本聚类/document2Vec。它的主要思想如下：一篇文章所叙述的内容是为了表达一些话题，而为了表达这些话题，需要用了一些词。因此在LDA模型中，一篇文档是这么生成的：

* 从狄利克雷分布中，一篇文档i具有一定的主题分布；
* 然后以主题多项式的分布，按概率产生文档i中第j个词所要表述的主题；
* 然后从狄利克雷分布中抽取主题对应的词的分布；
* 从词语的多项式分布中，采样生成最终的词语；
* 其中，类似Beta分布是二项式分布的共轭先验概率分布，而狄利克雷分布（Dirichlet分布）是多项式分布的共轭先验概率分布。

LDA的求解过程比较复杂，另外为什么有些时候是多项式分布，有些时候是狄利克雷分布，则需要分析整个LDA的基本假设，在这里就不做细讲。

### Word2Vec 算法简介
word2Vec的目标是希望将通过某种压缩算法把one-hot encoding 无损的压缩到低微空间里，实现的算法有很多，比如CBOW和Skip-gram等算法。

CBOW的算法思想：
* 将每个词i表示出Ci的k维向量；
* 然后用前后n个词(wi...wi+n)去预测中间某个词wi+j(也可能是一个[huffman树](https://baike.baidu.com/item/%E5%93%88%E5%A4%AB%E6%9B%BC%E6%A0%91)，包含了所有符合的词，huffman树是根绝各词在预料中出现的次数作为权重构建出来的)
* 然后用n个词中每个词的k维向量替换掉各自词本身
* 然后输入到神经网络中进行迭代求解
* 整体图如下所示：
    
 ![GitHub](http://img.blog.csdn.net/20140525173342578 "GitHub,Social Coding")

Skip-gram的算法思想：
* 将每个词i表示出Ci的k维向量；
* 然后用第i个词去预测上下文中的其它词
* 用词向量来表示所有词
* 然后输入到神经网络中去
* 整体图如下所示：

 ![GitHub](http://img.blog.csdn.net/20140525191842156 "GitHub,Social Coding")
 
这些算法的具体求解过程和他们的拓展在此就不一一讲了，有很多优化的办法，也有和DeepLearning结合的算法。

### ALS/LDA/word2Vec的应用场景
基本上来说，一个算法的最基本假设就明确了这个算法的应用场合。
比如ALS算法：其最终目的是为了解决用户和商品之间关系的问题。如果说我们是否可以用ALS做LDA的一些工作，从理论假设上讲是可以的，用文档替换用户，用词替换商品，用词在该文档中权重来替换用户商品得分。
这样去硬套，在一定程度上也是行的通的。不过其实中间有不少问题，比如词权重作为喜欢得分的合理性；
另外一个是相对用户商品消费来说词的量更大一些，文档-词 one-hot encoding的矩阵更稀疏一些，当求解参数个数为 MAX(document_num * k, word_num * k)时，需要文章中词的个数都大于K才不会过拟合

如果说ALS不适合LDA的工作场景，是因为ALS算法过拟合要求更严格一些，那LDA不能替换ALS算法则是因为目的不同。因为LDA的主要目的是得到，文档(用户)-主题(潜层主题)的权重，而协同过滤的目的还是为了
得到文档(用户)和词(商品)的关系，两者目的不同。不过感觉其实可以用LDA和R矩阵来求解User2Vector(用户特征)，甚至可以解决新用户消费商品数不足k个是ALS不准的问题。

另外word2Vec的目标就更为直接，就是将one-hot encoding降维。不过因为word2Vec算法的优越性我们还是可以考虑它的拓展应用，比如用户商品推荐上，文档(用户)-词(商品)，中间的关系可以是关注/购买等行为，最后可以得到词(商品)的向量化结果，然后也可以做相似词(商品)。

简单来说 LDA和word2Vec通过某种转化和映射，都能解决协同过滤中的一些问题，但是文本中的问题，因为矩阵过于稀疏，普通的协同过率算法都很难在NLP中有所建树。
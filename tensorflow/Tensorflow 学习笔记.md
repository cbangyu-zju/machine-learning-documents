# Tensorflow 学习笔记

## 基本知识点

* Tensor(张量)：表示某种相同数据类型的多维数组，数据类型+数组维度和各维度大小
* Variable(变量)：
* Operation(操作)
* Session(会话)
* Optimizer(优化器)

Tensorflow数据流图时一种声明式编程范式。

## 基本框架图
![学习笔记_1](images/学习笔记_1)

## Tensor 张量
表示某种相同数据类型的多维数组，数据类型+数组维度和各维度大小。

* 张量是用来表示多位数据的
* 张量是执行操作时的输入或输出数据
* 用户通过执行操作来创建或计算张量
* 张量的形状不一的在编译时确定，可以在运行时通过形状推断计算得出

在Tensorflow中，有几类比较特别的张量，由以下操作产生：

* tf.constant   // 常量
* tf.placeholder     // 占位符
* tf.Variable      // 变量

## Variable 变量
Tensorflow变量的主要作用是维护特点节点的状态。如深度学习或机器学习的模型参数

tf.Variable方法是操作，返回值是变量(特殊张量)。

通过tf.Variable方法创建的变量，与张量一样，可以作为操作的输入和输出，不同之处在于：

* 张量的生命周期通常随以来的计算完成而结束，内容也随即释放；
* 变量则常驻内存，在每一步训练时不断更新其值，以实现模型参数的更新

```
import tensorflow as tf

# 创建变量

w = tf.Variable(<initial-value>, name=<optional-name>)

# 讲变量作为操作的输入

y = tf.matmul(w, ...another variable or tensor...)
z = tf.sigmoid(w + y)

w.assign(w + 1.0)
w.assign_add(1.0)
```

![学习笔记_2](images/学习笔记_2)

## Operation 操作

![学习笔记_3](images/学习笔记_3)

Tensorflow用数据流图表示算法模型。数据流图由节点和有向边组成，每个节点均对应一个具体的操作。因此，操作是模型功能的实际载体。

数据流图中的节点按照功能不同可以分为3种：

* 存储节点：有状态的变量操作，通常用来存储模型参数；
* 计算节点：无状态的计算或控制操作，主要负责算法逻辑表达或流程控制；
* 数据节点：数据的占位符操作，用于描述图外输入数据的属性。

操作的输入和输出是张量或操作(函数式编程)

### 典型计算和控制操作

| 操作类型 | 典型操作 |
| --- | --- |
| 基础算术 | add/multiply/mod/sqrt/sin/trace/fft/argmin |
| 数组运算 | size/rank/split/reverse/cast/one_hot/quantize |
| 梯度裁剪 | clip_by_value/clip_by_norm/clip_by_global_norm |
| 逻辑控制和调试 | identity/logical_and/equal/less/is_finite/is_nan |
| 数据流控制 | enqueue/dequeue/size/take_grad/apply_grad/ |
| 初始化操作 | zeros_initializer/random_normal_initializer/orthogonal_initializer |
| 神经网络运算 | convolution/pool/bias_add/softmax/dropout/erosion2d |
| 随机运算 | random_normal/random_shuffle/multinomial/random_gamma |
| 字符串运算 | string_to_hash_bucket/reduce_join/substr/encode_base64 |
| 图像处理运输 | encode_png/resize_images/rot90/hsv_to_rgb/adjust_gamma |

### 占位符操作

Tensorflow使用占位符操作表示图外输入的数据，如训练和测试数据。

Tensorflow数据流图描述了算法模型的计算拓扑，其中的各个操作(节点)都是抽象的函数映射或数学表达式。换句话说，数据流图本身是一个具有计算拓扑和内部结构的“壳”。在用户向数据流图填充数据前，图中并没有真正执行任何计算。

## Session 会话
会话提供了估算张量和执行操作的运行环境，它是发放计算任务的客户端，所以计算任务都由它连接的执行引擎完成。一个会话的典型使用流程分为以下3个步骤：1. 创建会话；2. 估算张量或执行操作；3. 关闭会话。

### 会话执行原理

当我们调用sess.run(trian_op)语句执行训练操作时：

* 首先，程序内部提取操作依赖的所有前置操作。这些操作的节点共同组成一幅子图；
* 然后，程序会将自图中的计算节点、存储节点和数据节点按照各自的执行设备分类，相同设备上的节点组成了一幅局部图；
* 最后，每个设备上的局部图在实际执行时，根据节点间的依赖关系将各个节点有序地加载到设备上执行。

对于单机程序来说，相同机器上不同编号的CPU或GPU就是不同的设备，我们可以在创建节点时制定执行该节点的设备。

## Optimizer 优化器

优化器时实现优化算法的载体。
一次典型的迭代优化应该分为3个步骤：

1. 计算梯度：调用compute_gradients方法；
2. 处理梯度：用户按照自己需求处理梯度值，如梯度裁剪和梯度加权等；
3. 应用梯度：调用apply_gradients方法，将处理后的梯度值应用到模型参数。

### Tensorflow 内置优化器

| 优化器名称 | 文件路径 | 
| --- | --- |
| Adadelta | tensorflow/python/training/adadelta.py |
| Adagrad | tensorflow/python/training/adagrad.py |
|| Adagrad Dual Averaging |tensorflow/python/training/adagrad_da.py |
| Adam | tensorflow/python/training/adam.py |
| Ftrl | tensorflow/python/training/ftrl.py |
| Gradient Descent | tensorflow/python/training/gradient_descent.py |
| Momentum | tensorflow/python/training/momentum.py |
| Proximal Adagrad | tensorflow/python/training/proximal_adagrad.py |
| Proximal Gradient Descent | tensorflow/python/training/proximal_gradient_descent.py |
| Rmsprop | tensorflow/python/training/rmsprop.py |
| Synchronize Replicas | tensorflow/python/training/sync_replicas_optimizer.py |



# Tensorflow 实战 (一)

## 数据分析库
* Pandas 是一个BSD开源协议许可的，面向Python的高性能和易于上手的数据结构化和数据分析工具。
* DataFrame 是一个二维带表姐的数据结构，每列数据类型可以不同。可以作为表格或者数据库表使用

## 数据可视化库

* matplotlib 是一个 Python 2D 绘图库，可以生成出版物质量级别的图像和各种硬拷贝格式，并广泛支持多种平台，如:Python 脚本，Python，IPython Shell 和 Jupyter Notebook。
* seaborn 是一个基于 matplotlib的 Python 数据可视化库。它提供了更易用的高级接口，用于绘制精美且信息丰富的统计图形。
* mpl_toolkits.mplot3d 是一个基础 3D绘图(散点图、平面图、折线图等)工具集，也是matplotlib 库的一部分。同时，它也支持轻量级的独立安装模式。

## 房价预测
 假设房价y，和一些参数是线性关系，y = WX

```
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf


def read_data(data_filename):
    df1 = pd.read_csv(data_filename, names=['square', 'bedrooms', 'price'])
    df1.head()
    return df1

def prepare_sample(original_data):
    # 特征归一化
    normalized_data = original_data.apply(lambda column: (column - column.mean()) / column.std())

    # 增加 b
    ones = pd.DataFrame({"ones": np.ones(len(normalized_data))})
    normalized_data = pd.concat([ones, normalized_data], axis=1)

    # 输出 X 和 Y
    X_data= np.array(normalized_data[normalized_data.columns[0:3]])
    y_data = np.array(normalized_data[normalized_data.columns[-1]]).reshape(len(original_data), 1)

    return y_data, X_data


def trian(X_data, y_data):
    alpha = 0.01  # 学习率
    epoch = 500  # 迭代轮输

    # 创建线性回归模型(数据流图)

    X = tf.placeholder(tf.float32, X_data.shape) # 输入 X，形状 [47, 3]
    y = tf.placeholder(tf.float32, y_data.shape) # 输入 y，形状 [47, 1]

    # 权重变量 W，形状 [3, 1]
    W = tf.get_variable("weights", (X_data.shape[1], 1), initializer=tf.constant_initializer)

    # 假设函数 h(x) = w0x0 + w1x1 + w2x2 + w3x3，其中 x0 恒为1
    # 推理只 y = y_pred 形状 [47, 1]
    y_pred = tf.matmul(X, W)

    # 损失函数采用最小二乘法， y_pred - y 是形如 [47, 1]的向量
    # tf.matmul(a, b, transpose_a=True) 表示矩阵a的转置乘矩阵b，即[1, 47]*[47, 1]
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)

    # 随机梯度下降优化器 opt
    opt = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    # 但不训练操作 train_op:w
    train_op = opt.minimize(loss_op)

    with tf.Session() as sess:

        # 初始化全局变量
        sess.run(tf.global_variables_initializer())

        # 因为训练集较小 所以采用批量梯度下降优化算法，且每次使用全量数据训练
        for e in range(1, epoch + 1):
            sess.run(train_op, feed_dict={X: X_data, y: y_data})
            if e % 10 == 0:
                loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
                log_str = "Epoch %d \t LOss=%.4g \t Model: y = %.4g + %.4gx1 + %.4gx2"
                print(log_str % (e, loss, w[0], w[1], w[2]))

    return [w[0], w[1], w[2]]


def main():
    input_filename = "./data/dnn/data1.csv"
    original_data = read_data(input_filename)
    y_data, X_data = prepare_sample(original_data)
    W = trian(X_data, y_data)
    print(W)


if __name__ == '__main__':
    main()
```

## TensorBoard 可视化工具

* 在数据处理过程中，用户通常想要可视化地直观查看`数据集分布`情况。
* 在模型设计过程中，用户往往需要分析和检查`数据流图`是否正确实现。
* 在模型训练过程中，用户也常常需要关注`模型参数`和`超参数变化趋势`。
* 在模型测试过程中，用户也往往需要查看`准确率`和`召回率`等评估指标。


## 重写上述部分代码
```
def trian(X_data, y_data):
    alpha = 0.01  # 学习率
    epoch = 500  # 迭代轮输

    # 创建线性回归模型(数据流图)

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, X_data.shape) # 输入 X，形状 [47, 3]
        y = tf.placeholder(tf.float32, y_data.shape) # 输入 y，形状 [47, 1]

    with tf.name_scope('hypothesis'):
        # 权重变量 W，形状 [3, 1]
        W = tf.get_variable("weights", (X_data.shape[1], 1), initializer=tf.constant_initializer)

        # 假设函数 h(x) = w0x0 + w1x1 + w2x2 + w3x3，其中 x0 恒为1
        # 推理只 y = y_pred 形状 [47, 1]
        y_pred = tf.matmul(X, W)

    with tf.name_scope('loss'):
        # 损失函数采用最小二乘法， y_pred - y 是形如 [47, 1]的向量
        # tf.matmul(a, b, transpose_a=True) 表示矩阵a的转置乘矩阵b，即[1, 47]*[47, 1]
        loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)

    with tf.name_scope('train'):
        # 随机梯度下降优化器 opt
        train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

    with tf.Session() as sess:

        # 初始化全局变量
        sess.run(tf.global_variables_initializer())

        # 创建FileWriter实例，并传入当前会话加载的数据流图
        writer = tf.summary.FileWriter("./summary/linear-regression-1", sess.graph)
        # 因为训练集较小 所以采用批量梯度下降优化算法，且每次使用全量数据训练
        for e in range(1, epoch + 1):
            sess.run(train_op, feed_dict={X: X_data, y: y_data})
            if e % 10 == 0:
                loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
                log_str = "Epoch %d \t LOss=%.4g \t Model: y = %.4g + %.4gx1 + %.4gx2"
                print(log_str % (e, loss, w[0], w[1], w[2]))
    writer.close()
    return [w[0], w[1], w[2]]
```

## 启动TensorBoard

```
tensorboard --logdir ./ --host localhost
打开：http://localhost:6006
```




 
# Tensorflow 平台的基础知识点

## 平台基础

* protocol buffer：和xml、json类似，用于处理结构化数据，可序列化，是一种schema+二进制数据形式，序列化数据小，且解析速度快。
* Bazel：自动化构建工具，通过BUILD文件进行编译，类似于MakeFile，WORKSPACE。

## 计算框架

* Tensor：张量，从功能来看是一个n维数组，从实现上来看，只是计算结果的一个引用，没有给出计算结果，而是给出结果张量的结构。主要保存了三个属性：名字(name，唯一标识符)、维度(shape)、类型(type)
* Senssion：
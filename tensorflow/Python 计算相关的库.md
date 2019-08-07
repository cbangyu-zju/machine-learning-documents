# Python 计算相关的库

摘抄自：[NumPy 官方快速入门教程(译)](https://juejin.im/post/5a76d2c56fb9a063557d8357)
## Numpy
Numpy的主要操作对象是同类型的多维数组。它是一个由正整数元祖索引，元素类型相同的表。

### 基本概念
Numpy的主要操作对象是同类型的多维数组。它是一个由正整数元祖索引，元素类型相同的表。在Numpy维度被称为axes。axes的数量称为rank。
例如，在3D空间的一个点[1，2，1]是一个rank=1的数组，因为他只有一个axes，这个axes的长度是3.在下面这个例子中，rank=2
```
[[1,0,0], [0,1,2]] // rank=2，第一维(axes)长度为2，第二维长度为3
```

### 属性和函数

#### ndarray
Numpy的数组类是ndarray，也可以叫做array，和标准库的array.array不同，只提供处理一维数组且功能更是

* ndarray.ndim：数组的 axes （维数）数值大小。在 Python 中维数的大小可以参考 rank；
* ndarray.shape：数组的维数，这是由每个维度的大小组成的一个元组。对于一个 n 行 m 列的矩阵。shape 是 (n, m)。由 shape 元组的长度得出 rank 或者维数 ndim
* ndarray.size：数组元素的个数总和，这等于 shape 元组数字的乘积。
* ndarray.dtype：在数组中描述元素类型的一个对象。另外，NumPy也提供了它自己的类型：numpy.int32，numpy.int16，numpy.float64…
* ndarray.itemsize：数组中每个元素所占字节数。例如，一个 float64 的 itemsize 是 8 ( = 64/8bit)，complex32 的 itemsize 是 4 ( = 32/8bit)。它和 ndarray.dtype.itemsize 是相等的。
* ndarray.data：数组实际元素的缓存区。通常来说，我们不需要使用这个属性，因为我们会使用索引的方式访问数据。

#### 功能操作
1. 创建数组

```
import numpy as np
a = np.array([2,3,4])
np.array([(1.5,2,3), (4,5,6)])
np.array([[1.5,2,3], [4,5,6]])
np.array( [ [1,2], [3,4] ], dtype=complex ) // 指定为复数 
np.zeros( (3,4) ) // shape 为 (3,4)的全零
np.ones( (2,3,4), dtype=np.int16 ) // 指定类型
np.empty( (2,3) )  // shape 为 (2,3)的初始化参数
np.arange( 10, 30, 5 ) // 获得array([10, 15, 20, 25])
np.arange( 0, 2, 0.3 ) // 获得array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])，更倾向于用np.linspace( 0, 2, 9 )
np.random.random((2,3))
```

2. 基本数组操作

```
//在数组上的算数运算应用于每个元素。并创建一个用结果填充的新的数组
np.array( [20,30,40,50] ) - np.arange( 4 ) = array([20, 29, 38, 47]) // 每个元素相应加减
np.array([[1,1],[0,1]]) * np.array( [[2,0],[3,4]] ) = array([[2, 0], [0, 4]]) // 每个元素相应乘除
// 矩阵乘法通过 dot 函数进行模拟。
A.dot(B)
// += 和 *= 操作之类的，直接在原数组上做修改，不会创建新数组
np.ones((2,3), dtype=int) * 3  // 每个元素乘
// 在不同数组类型之间的操作，结果数组的类型趋于更普通或者更精确的一种（称为向上转型）
np.ones(3, dtype=np.int32) + np.linspace(0,pi,3) = array([ 1.,  2.57079633,  4.14159265])
// 许多类似于求数组所有元素的和的一元操作都是作为 ndarray 类的方法实现的。
a = np.random.random((2,3))
a.sum() // a.max(), a.min()等
```

3. 通用功能

```
NumPy 提供了很多数学上的函数，例如 sin、cos、exp。这些被叫做 "universal functions" (ufunc)。在 $ NumPy \）中这些函数是操作数组数字，产生一个数组作为输出。
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])
```

4. 索引、切片、迭代

```
a = np.arange(10) ** 3 // **3 表示3次方
a[2:5] // 和标准array类似 a[:6:2]，a[::-1] 反序
a[:6:2] = -1 // 对应的原素 -1 

// 多维数组对于每个 axis 都有一个索引，这些索引用逗号分隔
a = np.array([[2, 0], [0, 4]]) 
a[1,1] // 4 
for element in a.flat // 遍历多维数组
```

5. 操控形状

```
a = np.floor(10*np.random.random((3,4)))
a.shape // (3, 4)

// 原数组不变
a.ravel()  // 拍平
a.reshape(6,2) // 
a.T  // 转置

// 原数组改变
a.resize(6, 2) // reshape 函数返回修改的形状，而 ndarray.resize 方法直接修改数组本身。

// 不同数组的拼接
vstack 和 hstack // 横向拼接和众向拼接
// 切割
np.hsplit(a,3)   # Split a into 3
np.hsplit(a,(3,4))   # Split a after the third and the fourth column
// vplit 沿着竖直的 axis 分割，array_split 允许通过指定哪个 axis 去分割。
```

6. 拷贝和views

```
// 不同的数组对象可以分享相同的数据。view 方法创建了一个相同数据的新数组对象。 PS：这里 View（视图？） 不知道如何理解好，所以保留。
c = a.view()  // False
c.base is a                        // True c is a view of the data owned by a 浅拷贝， 切片数组返回一个 view
// 深拷贝， 
d = a.copy()
```

7. 花式索引

```
//  用索引数组索引
a = np.arange(12)**2       
i = np.array( [ 1,1,3,8,5 ] )
a[i]

// 第二种用布尔索引方法更像是整数索引，对于每个数组的维度，我们给一个 1D 的布尔数组去选择我们想要的切片。
```

8. 线性代数

```
a = np.array([[1.0, 2.0], [3.0, 4.0]])
a.transpose() // 转置

np.linalg.inv(a)  // 求逆

u = np.eye(2) # unit 2x2 matrix; "eye" represents "I"

np.trace(u) // trace 秩
np.linalg.eig(u) // 特征矩阵
```















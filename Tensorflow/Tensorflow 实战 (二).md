# Tensorflow 实战 (二)

## 手写体数字识别

### MNIST数据集
MNIST 是一套手写体数字的图像数据集，包含 60,000 个训练样例和 10,000 个测试样例， 由纽约大学的 Yann LeCun 等人维护。详见：[MNIST官网](http://yann.lecun.com/exdb/mnist/)

#### 数据集介绍
MNIST 图像数据集使用形如[28，28]的二阶数组来表示每个手写体数字，数组中 的每个元素对应一个像素点，即每张图像大小固定为 28x28 像素。

MNIST 数据集中的图像都是256阶灰度图，即灰度值 0 表示白色(背景)，255 表示 黑色(前景)，使用取值为[0，255]的uint8数据类型表示图像。为了加速训练，我 们需要做数据规范化，将灰度值缩放为[0，1]的float32数据类型。

数据读取：https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

```
import tensorflow.keras as keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('mnist/mnist.npz')

```
### MNIST Softmax 网络
模型结构图如下：
![学习笔记_4](images/学习笔记_4)

### 实战
```
import keras.datasets.mnist as mnist
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow.gfile as gfile
import os


def load_data(data_filename):
    (x_train, y_train), (x_test, y_test) = mnist.load_data(data_filename)
    print(x_train.shape, type(x_train))
    print(y_train.shape, type(y_train))
    return x_train, y_train, x_test, y_test


def summary_data(y_train):
    label, count = np.unique(y_train, return_counts=True)
    print(label, count)
    fig = plt.figure()
    plt.bar(label, count, width = 0.7, align='center')
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(label)
    plt.ylim(0,7500)

    for a,b in zip(label, count):
        plt.text(a, b, '%d' % b, ha='center', va='bottom',fontsize=10)
    plt.show()


def prepare_sample(x_train, x_test, y_train, y_test):
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channel_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 将数据类型转换为float32
    X_train = x_train.astype('float32')
    X_test = x_test.astype('float32')
    # 数据归一化
    X_train /= 255
    X_test /= 255

    n_classes = 10
    print("Shape before one-hot encoding: ", y_train.shape)
    Y_train = np_utils.to_categorical(y_train, n_classes)
    print("Shape after one-hot encoding: ", Y_train.shape)
    Y_test = np_utils.to_categorical(y_test, n_classes)
    return X_train, X_test, Y_train, Y_test, input_shape, n_classes


def init_model(input_shape, n_classes):
    model = Sequential()
    ## Feature Extraction
    # 第1层卷积，32个3x3的卷积核 ，激活函数使用 relu
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))

    # 第2层卷积，64个3x3的卷积核，激活函数使用 relu
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

    # 最大池化层，池化窗口 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dropout 25% 的输入神经元
    model.add(Dropout(0.25))

    # 将 Pooled feature map 摊平后输入全连接网络
    model.add(Flatten())

    ## Classification
    # 全联接层
    model.add(Dense(128, activation='relu'))

    # Dropout 50% 的输入神经元
    model.add(Dropout(0.5))

    # 使用 softmax 激活函数做多分类，输出各数字的概率
    model.add(Dense(n_classes, activation='softmax'))
    return model


def summary_model(model):
    model.summary()
    for layer in model.layers:
        print(layer.get_output_at(0).get_shape().as_list())
    return


def train(X_train, Y_train, X_test, Y_test, input_shape, n_classes):
    batch_size = 128
    epochs = 5
    verbose = 2
    model = init_model(input_shape, n_classes)
    summary_model(model)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(X_train,
                        Y_train,
                        batch_size=128,
                        epochs=5,
                        verbose=2,
                        validation_data=(X_test, Y_test))

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    plt.show()
    return model


def save_mode(model, model_dir, model_name):
    if gfile.Exists(model_dir):
        gfile.DeleteRecursively(model_dir)
    gfile.MakeDirs(model_dir)
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def evaluation():
    pass


def main():
    input_filename = 'mnist/mnist.npz'
    x_train, y_train, x_test, y_test = load_data(input_filename)
    summary_data(y_train)
    X_train, X_test, Y_train, Y_test, input_shape, n_classes = prepare_sample(x_train, x_test, y_train, y_test)
    model = train(X_train, Y_train, X_test, Y_test, input_shape, n_classes)
    save_mode(model, "./model", "mnist_cnn")


if __name__ == '__main__':
    main()
```

## 

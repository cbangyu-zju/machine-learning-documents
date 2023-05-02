# 目标检测和分类之Fast RCNN
继2014年的RCNN之后，Ross Girshick在15年推出Fast RCNN，构思精巧，流程更为紧凑，大幅提升了目标检测的速度。同样使用最大规模的网络，Fast RCNN和RCNN相比，训练时间从84小时减少为9.5小时，测试时间从47秒减少为0.32秒。在PASCAL VOC 2007上的准确率相差无几，约在66%-67%之间.

RCNN的基础思想已经在上一节讲过，大致如下：1. 在图像中确定约1000-2000个候选框；2.对于每个候选框内图像块，使用深度网络提取特征；3. 对候选框中提取出的特征，使用分类器判别是否属于一个特定类；4. 对于属于某一特征的候选框，用回归器进一步调整其位置。改进的Fast RCNN主要解决的是RCNN中一个大缺点：由于每一个候选框都要独自经过CNN，这使得花费的时间非常多。Fast RCNN的解决办法如下：共享卷积层，现在不是每一个候选框都当做输入进入CNN了，而是输入一张完整的图片，在第五个卷积层再得到每个候选框的特征。改良后的FastRCNN性能得到大幅度提升。

![目标检测和分类之FastRCNN_1](images/目标检测和分类之FastRCNN_1)

以下将对FastRCNN的流程进行详细讲解。

## SPP Net
SPP：Spatial Pyramid Pooling（空间金字塔池化）。SPP-Net是出自2015年发表在IEEE上的论文-《Spatial Pyramid Pooling in Deep ConvolutionalNetworks for Visual Recognition》。众所周知，CNN一般都含有卷积部分和全连接部分，其中，卷积层不需要固定尺寸的图像，而全连接层是需要固定大小的输入。所以当全连接层面对各种尺寸的输入数据时，就需要对输入数据进行crop（crop就是从一个大图扣出网络输入大小的patch，比如227×227），或warp（把一个边界框bounding box的内容resize成227×227）等一系列操作以统一图片的尺寸大小，比如224\*224（ImageNet）、32\*32(LenNet)、96\*96等。

![目标检测和分类之FastRCNN_2](images/目标检测和分类之FastRCNN_2)
![目标检测和分类之FastRCNN_3](images/目标检测和分类之FastRCNN_3)

所以才如你在上文中看到的，在R-CNN中，“因为取出的区域大小各自不同，所以需要将每个Region Proposal缩放（warp）成统一的227x227的大小并输入到CNN”。但warp/crop这种预处理，导致的问题要么被拉伸变形、要么物体不全，限制了识别精确度。SPP Net的作者Kaiming He等人逆向思考，既然由于全连接FC层的存在，普通的CNN需要通过固定输入图片的大小来使得全连接层的输入固定。那借鉴卷积层可以适应任何尺寸，为何不能在卷积层的最后加入某种结构，使得后面全连接层得到的输入变成固定的呢？这个“化腐朽为神奇”的结构就是spatial pyramid pooling layer。下图便是R-CNN和SPP Net检测流程的比较：

![目标检测和分类之FastRCNN_4](images/目标检测和分类之FastRCNN_4)

它的特点有两个:

1. 结合空间金字塔方法实现CNNs的多尺度输入。SPP Net的第一个贡献就是在最后一个卷积层后，接入了金字塔池化层，保证传到下一层全连接层的输入固定。换句话说，在普通的CNN机构中，输入图像的尺寸往往是固定的（比如224*224像素），输出则是一个固定维数的向量。SPP Net在普通的CNN结构中加入了ROI池化层（ROI Pooling），使得网络的输入图像可以是任意尺寸的，输出则不变，同样是一个固定维数的向量。
2. 只对原图提取一次卷积特征。在R-CNN中，每个候选框先resize到统一大小，然后分别作为CNN的输入，这样是很低效的。而SPP Net根据这个缺点做了优化：只对原图进行一次卷积计算，便得到整张图的卷积特征feature map，然后找到每个候选框在feature map上的映射patch，将此patch作为每个候选框的卷积特征输入到SPP layer和之后的层，完成特征提取工作。如此这般，R-CNN要对每个区域计算卷积，而SPPNet只需要计算一次卷积，从而节省了大量的计算时间，比R-CNN有一百倍左右的提速。

## Fast RCNN
SPP Net真是个好方法，R-CNN的进阶版Fast R-CNN就是在R-CNN的基础上采纳了SPP Net方法，对R-CNN作了改进，使得性能进一步提高。先说R-CNN的缺点：即使使用了Selective Search等预处理步骤来提取潜在的bounding box作为输入，但是R-CNN仍会有严重的速度瓶颈，原因也很明显，就是计算机对所有region进行特征提取时会有重复计算，Fast-RCNN正是为了解决这个问题诞生的。

![目标检测和分类之FastRCNN_5](images/目标检测和分类之FastRCNN_5)

与R-CNN框架图对比，可以发现主要有两处不同：一是最后一个卷积层后加了一个ROI pooling layer，二是损失函数使用了多任务损失函数(multi-task loss)，将边框回归Bounding Box Regression直接加入到CNN网络中训练。

1. ROI pooling layer实际上是SPP-NET的一个精简版，SPP-NET对每个proposal使用了不同大小的金字塔映射，而ROI pooling layer只需要下采样到一个7x7的特征图。对于VGG16网络conv5_3有512个特征图，这样所有region proposal对应了一个7\*7\*512维度的特征向量作为全连接层的输入。换言之，这个网络层可以把不同大小的输入映射到一个固定尺度的特征向量，而我们知道，conv、pooling、relu等操作都不需要固定size的输入，因此，在原始图片上执行这些操作后，虽然输入图片size不同导致得到的feature map尺寸也不同，不能直接接到一个全连接层进行分类，但是可以加入这个神奇的ROI Pooling层，对每个region都提取一个固定维度的特征表示，再通过正常的softmax进行类型识别。
2. R-CNN训练过程分为了三个阶段，而Fast R-CNN直接使用softmax替代SVM分类，同时利用多任务损失函数边框回归也加入到了网络中，这样整个的训练过程是端到端的(除去Region Proposal提取阶段)。也就是说，之前R-CNN的处理流程是先提proposal，然后CNN提取特征，之后用SVM分类器，最后再做bbox regression，而在Fast R-CNN中，作者巧妙的把bbox regression放进了神经网络内部，与region分类和并成为了一个multi-task模型，实际实验也证明，这两个任务能够共享卷积特征，并相互促进。

![目标检测和分类之FastRCNN_6](images/目标检测和分类之FastRCNN_6)

所以，Fast-RCNN很重要的一个贡献是成功的让人们看到了Region Proposal + CNN这一框架实时检测的希望，原来多类检测真的可以在保证准确率的同时提升处理速度，也为后来的Faster R-CNN做下了铺垫。

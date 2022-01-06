# Dual-Model-Integration
A sunspot magnetic classification algorithm based on dual model integration
# 双模型说明

针对太阳黑子数据集，由于数据集中的alpha类和betax类的样本数量差别较大，无法用一个模型兼顾两者（如大模型易使betax类过拟合，小模型易使alpha类欠拟合），且无法使用相同的数据增强和其他调优手段兼顾二者，且经实验验证：alpha和betax之间基本无交集（即不存在alpha类被误判为betax类、betax类被误判为alpha类的情况），故采用针对alpha类和betax类两个类别分别训练模型，即训练较深的双resnet18网络模型侧重于alpha的分类，训练较浅的改进alexnet网络侧重于betax的分类，最终融合二者的结果，即可得到较好的beta分类结果。两个模型的具体描述如下：


# alpha-model

用来预测alpha的模型

### 一、环境说明

- Windows 10
- CUDA 10.2
- CUDNN 7.6.0
- python  3.7.6
- numpy 1.18.4
- torch  1.5.0
- torchvision  0.6.0
- opencv 4.2
- matplotlib 3.1.3
- astropy.io 4.0
- PIL
- skimage

### 二、解决方案

#### 神经网络

针对拥有较多样本不易过拟合的alpha类别，我们使用参数较多的双网络并行resnet18-D，两个并行的resnet18-D卷积层分别提取单通道白光图和磁场图的特征，再将二者的特征拼接送入全连接进行分类；其中resnet18-D是基于resnet18改进了下采样层，加入平均池化用以改善精度。

#### 训练技巧

首先使用单通道白光图和磁场图分别对两个单网络resnet-D模型进行预训练（不采用ImageNet预训练），再将两个预训练模型的卷积层迁移至双网络模型的并行卷积层。

使用加权的交叉熵损失，样本较少的类别享有更大的权重，用以减小各类别样本数量不均衡带来的影响。

#### 数据增强

由于此模型侧重于alpha类别，而alpha类别和相邻的beta类别样本数量均较大，故只采用了水平镜像和竖直镜像的数据增强方法，再使用其他数据增强方法不仅无法提升正确率，反而会影响训练时间。


# betax-model

用来预测betax的模型

### 一、环境说明

- Windows  10
- CUDA  10.2
- CUDNN  7.6.0
- python  3.7.6
- numpy  1.18.4
- torch  1.5.0
- torchvision  0.6.0
- opencv  4.2
- matplotlib  3.1.3
- astropy.io  4.0
- PIL
- skimage

### 二、解决方案

#### 数据集划分

​	取全部的alpha数据和beta数据，对于betax数据，先按照时间进行排序，发现某些时间段，betax的数据比较少，只有少数的几张，而有些时间段betax数据却很多，对于这些betax数据多的时间段，对连续拍摄的betax数据每隔四份取一份，意味着删去了将近3/4的betax 数据。

#### 神经网络

​	基于Alexnet做了一些修改，使用Alexnet的backbone，后接一个output_size为3x3的自适应池化层，最后再加一个全连接层，输出神经元个数为3。使用SGD优化器，学习率设为0.0008，动量为0.9。

#### 增大差异性

​	对原先的输出的logit除以20，再进行softmax，这样做的目的是增大softmax各个类别的差异性，从而得到更好的训练。	

#### 数据增强

​	使用随机旋转数据增强，旋转角度控制在90以内。

​	将原图像尺寸变换为 500x375。

#### 后处理

​	通过分析验证集的logit输出分布，发现当betax的logit值减去beta的logit值在[0, 0.5] 这个区间时，往往会将beta误分为betax。

​	当betax的logit值减去beta的logit值在[0, 0.5] 这个区间时，增大beta的logit值，进行概率性校正。

# project使用方法

1. 运行fits2jpg.py，将.fits文件转成jpg图片并存储至tmp_data中。
2. 运行inference_test_alpha.py，得到侧重于alpha的分类结果result_1.txt。
3. 运行inference_test_betax.py，得到侧重于betax的分类结果result_2.txt。
4. 运行TXT.exe（位于code文件夹中，代码请看code\TXT\TXT\TXT.cpp），将第二步和第三步的两个结果进行融合，得到最终分类结果result.txt。

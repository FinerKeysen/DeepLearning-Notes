[toc]

内容：
- Nets, Layers, and Blobs： Caffe 模型解析；
- Forward and Backward： 层状模型的基本计算；
- Loss： 由 loss 定义待学习的任务；
- Solver： solver 协调模型的优化；
- Layer Catalogue： “层”是模型和计算的基本单元， Caffe 提供的结构中包含了构建先进模型所需的各种层；
- Interface： Caffe 的命令行， Python，和 MATLAB 版接口；
- Data： 如何为模型添加 caffe 式的输入数据

# 第一章 Blobs, Layers, and Nets： Caffe 模型解析

caffe模型特点：逐层定义(layer-by-layer)方式形成网络，网络从数据输入层到损失层自上而下地定义模型

Blobs结构：存储、交换、处理forward和backward的数据和导数信息，是caffe的标准数组结构
Layer：caffe模型和计算的基本单元
Net：一系列layers和其连接的集合

## 1.1 Blob的存储、交换
Blob是 Caffe 中处理和传递实际数据的数据封装包，可以在CPU和GPU之间同步。按C风格连续存储的N维数组。

FOR IMAGE DATA：blob的常规维数是 图像数量 N ×通道数 K ×图像高度 H ×图像宽度 W，并按行(row-major)存储，那么在一个4维的blob中，坐标(n, k, h, w)的像素点的物理位置为(((n×K + k)×H + h)×W + w)，所以有最后面/最右边的维度更新最快。
Number/N：每批次处理的数据量
Channel/K：特征维度，比如RGB图像的K=3

参数 Blob 的维度是根据层的类型和配置而变化的

### 1.1.1 实现细节
blob中的关键数据 values(值){网络中传送的普通数据} 和 gradients(梯度){通过网
络计算得到的梯度}

两种数据访问方式：
静态方式，不改变数值；`const DType* cpu_data() const;`
动态方式，改变数值；`DType* mutable_cpu_data();`
gpu 和 diff 的操作与之类似

blob 使用了一个 SyncedMem 类来同步 CPU 和 GPU 上的数值，以隐藏同步的细节和最小化传送数据

实际上，使用 GPU 时， Caffe 中 CPU 代码先从磁盘中加载数据到 blob，同时请求分配一个 GPU 设备核（ device kernel） 以使用 GPU 进行计算，再将计算好的 blob 数据送入下一层，这样既实现了高效运算，又忽略了底层细节。 只要所有 layers 均有 GPU 实现，这种情况下所有的中间数据和梯度都会保留在 GPU 上。

示例，用以确定 blob 何时会复制数据：

```C++
// 假定数据在 CPU 上进行初始化，我们有一个 blob
const Dtype* foo;
Dtype* bar;
foo = blob.gpu_data(); // 数据从 CPU 复制到 GPU
foo = blob.cpu_data(); // 没有数据复制，两者都有最新的内容
bar = blob.mutable_gpu_data(); // 没有数据复制
// ... 一些操作 ...
bar = blob.mutable_gpu_data(); // 仍在 GPU，没有数据复制
foo = blob.cpu_data(); // 由于 GPU 修改了数值，数据从 GPU 复制到 CPU
foo = blob.gpu_data(); //没有数据复制，两者都有最新的内容
bar = blob.mutable_cpu_data(); // 依旧没有数据复制
bar = blob.mutable_gpu_data(); //数据从 CPU 复制到 GPU
bar = blob.mutable_cpu_data(); //数据从 GPU 复制到 CPU
```

## 1.2 Layer 的计算和连接
Layer 是 Caffe 模型的本质内容和执行计算的基本单元，可以进行如convolve（卷积）、 pool（ 池化）、 inner product（内积）， rectified-linear 和 sigmoid 等非线性运算，元素级的数据变换， normalize（ 归一化）、 load data（数据加载）、 softmax 和 hinge等 losses（损失计算）。

一个 layer 通过 bottom（底部）连接层接收数据，通过 top（顶部）连接层输出数据。

<div align=center>
<img src="http://img.blog.csdn.net/20180402202845366">
</div>

每个 layer 都定义有 3 种运算：
- setup（初始化设置）：在模型初始化时重置 layers 及其相互之间的连接 ;
- forward（前向传播）：从 bottom 层中接收数据，进行计算后将输出送入到 top 层中 ;
- backward（反向传播）：给定相对于 top 层输出的梯度，计算其相对于输入的梯度，并传递到 bottom层。

注意：一个有参数的 layer 需要计算相对于各个参数的梯度值并存储在内部。

因此，自定义的 layer 只要定义好 layer 的 setup、forward、backward。

特别地，Forward 和 Backward 函数分别有 CPU 和 GPU 两种实现方式。如果没有实现 GPU 版本，那么 layer 将转向作为备用选项的 CPU 方式。尽管这样会增加额外的数据传送成本（输入数据由 GPU 上复制到 CPU，之后输出数据从 CPU 又复制回到 GPU）。但对于一般的实验还是很方便。

## 1.3 Net 的定义和操作
Net 是由一系列层组成的有向无环（ DAG）计算图， Caffe 保留了计算图中所有的中间值以确保前向和反向迭代的准确性。一个典型的 Net 开始于 data layer——从磁盘中加载数据，终止于 loss layer——计算如分类和重构这些任务的目标函数。

一个简单的逻辑回归分类器的定义如下：

<div align=center>
<img src="http://img.blog.csdn.net/20180402203720459">
</div>

```Protocol
name: "LogReg"
layer {
    name: "mnist"
    type: "Data"
    top: "data"
    top: "label"
    data_param {
        source: "input_leveldb"
        batch_size: 64
    }
}

layer {
    name: "ip"
    type: "InnerProduct"
    bottom: "data"
    top: "ip"
    inner_product_param {
        num_output: 2
    }
}

layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "ip"
    bottom: "label"
    top: "loss"
}
```

Net::Init()进行模型的初始化。初始化主要实现两个操作：创建 blobs 和 layers 以搭建整个网络 DAG 图以及调用 layers 的 SetUp()函数。

初始化时也会做另一些记录，例如确认整个网络结构的正确与否等。 另外，初始化期间， Net 会打印其初始化日志到 INFO 信息中。

Caffe 中网络的构建与设备无关，网络构建完之后，通过设置 Caffe::mode()函数中的Caffe::set_mode()，即可实现在 CPU 或 GPU 上的运行。

### 1.3.1 模型格式
模型是利用文本 protocol buffer（ prototxt）语言定义的，学习好的模型会被序列化地存储在二进制 protocol buffer (binaryproto) .caffemodel 文件中。

模型格式用 protobuf 语言定义在 `caffe.proto` 文件中。

# 第二章 Forward and Backward（前传/反传）

<div align=center>
<img src="http://img.blog.csdn.net/20180402205733781">
</div>

## 2.1 前传forward

在前传过程中， Caffe 组合每一层的计算以得到整个模型的计算“函数”。本过程自底向上进行。 数据 x 通过一个内积层得到 g(x )，然后通过 softmax 层得到h(g(x))，通过 softmax loss 得到fw(x)。

<div align=center>
<img src="http://img.blog.csdn.net/20180402210026074">
</div>

## 2.2 反传backward

在反传过程中， Caffe 通过自动求导并反向组合每一层的梯度来计算整个网络的梯度。这就是反传过程的本质。本过程自顶向下进行。

<div align=center>
<img src="http://img.blog.csdn.net/20180402210322813">
</div>

反传过程以损失开始，然后根据输出计算梯度 $\frac{\partial{f_W}}{\partial h}$ 。根据链式准则，逐层计算出模型其余部分的梯度。有参数的层，例如 INNER_PRODUCT 层， 会在反传过程中根据参数计算梯度 $\frac{\partial{f_W}}{\partial{W_{ip}}}$

## 2.3 Caffe 中前传和反传的实现

- `Net::Forward()`和 `Net::Backward()`方法实现**网络的前传和后传**，而 `Layer::Forward()`和
`Layer::Backward()`计算**每一层的前传后传**。

- 每一层都有 `forward_{cpu, gpu}()`和 `backward_{cpu, gpu}`方法来适应不同的计算模式。由
于条件限制或者为了使用便利，一个层可能仅实现了 CPU 或者 GPU 模式。

**Solver** 优化一个模型，首先通过调用前传来获得输出和损失，然后调用反传产生模型的梯度，将梯度与权值更新后相结合来最小化损失。

# 第三章 Loss

caffe的学习由 损失函数(误差、代价或者目标函数)驱动，一个损失函数通过将参数集（即当前的网络权值）映射到一个可以标识这些参数“不良程度”的标量值来学习目标。学习的目的是找到一个网络权重的集合，使得损失函数最小。

在 Caffe中，损失是通过网络的前向计算得到的。每一层由一系列的输入 blobs (bottom)，然后产生一系列的输出 blobs (top)。这些层的某些输出可以用来作为损失函数。典型的一对多分类任务的损失函数是 softMaxWithLoss 函数，使用以下的网络定义，例如:

```Protocol
layer {
name: "loss"
type: "SoftmaxWithLoss"
bottom: "pred"
bottom: "label"
top: "loss"
}
```

在 softMaxWithLoss 函数中， top blob 是一个标量数值， 该数值是整个 batch 的损失平均值（ 由预测值 pred 和真实值 label 计算得到）。

## 3.1 Loss weights

按照惯例，有着 Loss 后缀的 Caffe 层对损失函数有贡献，其他层被假定仅仅用于中间计算。然而，通过在层定义中添加一个 loss_weight:<float>字段到由该层的 top blob，任何层都可以作为一个 loss。对于带后缀 Loss 的层来说，其对于该层的第一个 top blob 含有一个隐式的 loss_weight:1；其他层对应于所有 top blob 有一个隐式的 loss_weight:0。因此，上面的softMaxWithLoss 层等价于：

```
layer {
name: "loss"
type: "SoftmaxWithLoss"
bottom: "pred"
bottom: "label"
top: "loss"
loss_weight: 1
}
```

然而，任何可以反向传播的层，可允许给予一个非 0 的 loss_weight，例如，如果需要，对网络的某些中间层所产生的激活进行正则化。对于具有相关非 0 损失的非单输出，损失函数可以通过对所有 blob 求和来进行简单地计算。

那么，在 Caffe 中最终的损失函数可以通过对整个网络中所有的权值损失进行求和计算获得，正如以下的伪代码：

```
loss := 0
for layer in layers:
for top, loss_weight in layer.tops, layer.loss_weights:
loss += loss_weight * sum(top)
```

# 第四章 Solver
## 4.1 Solver简介

Solver 通过协调 Net 的前向推断计算和反向梯度计算（ forward inference and backward
gradients）， 来对参数进行更新， 从而达到减小 loss 的目的。

Caffe 模型的学习被分为两个部分：
- Solver 进行优化、更新参数
- 由 Net 计算出 loss 和 gradient

Solver：
1. 用于优化过程的记录、 创建训练网络（用于学习）和测试网络（用于评估）；
2. 通过 forward 和 backward 过程来迭代地优化和更新参数；
3. 周期性地用测试网络评估模型性能；
4. 在优化过程中记录模型和 solver 状态的快照（ snapshot）；

每一次迭代过称中：
1. 调用 Net 的前向过程计算出输出和 loss；
2. 调用 Net 的后向过程计算出梯度（ loss 对每层的权重 w 和偏置 b 求导）；
3. 根据下面所讲的 Solver 方法，利用梯度更新参数；
4. 根据学习率（ learning rate）， 历史数据和求解方法更新 solver 的状态,使权重从初始
化状态逐步更新到最终的学习到的状态。 solvers 的运行模式有 CPU/GPU 两种模式。

## 4.2 Methods

<div align=center>
<img src="http://img.blog.csdn.net/20180403104519700">
<p></p>
<img src="http://img.blog.csdn.net/20180403105101433">
</div>

### 4.2.1 SGD

<div align=center>
<img src="http://img.blog.csdn.net/20180403105729747">
</div>

学习的超参数（ α 和 μ）需要一定的调整才能达到最好的效果。参考`L. Bottou. Stochastic Gradient Descent Tricks. Neural Networks: Tricks of the Trade: Springer, 2012.`

**设定学习率 α 和动量 μ 的经验法则:** 

一个比较好的建议是,将学习速率（learning rate α）初始化为$α ≈ 0.01 = 10^2$，然后在训练（ training）中当 loss 达到稳定时，将 α 除以一个常数（例如 10），将这个过程重复多次。对于动量（ momentum μ）一般设置为μ = 0.9， μ 使 weight 的更新更为平缓，使学习过程更为稳定、快速。

caffe中SolverParameter可以简单的实现，详见：`./examples/imagenet/alexnet_solver.prototxt.`
实现上述技巧，可添加下面的代码到自定义的solver prototxt文件中：

```
base_lr: 0.01 # 开始学习速率为： α = 0.01=1e-2
lr_policy: "step" # 学习策略: 每 stepsize 次迭代之后，将 α 乘以 gamma
gamma: 0.1 # 学习速率变化因子
stepsize: 100000 # 每 100K 次迭代，降低学习速率
max_iter: 350000 # 训练的最大迭代次数 350K
momentum: 0.9 # 动量 momentum 为： μ = 0.9
```

上述例子中，当训练次数达到一定量后，更新值（ update）会扩大到 $\frac{μ}{1-μ}$ 倍，所以如果增加 $μ$ 的话，最好是相应地减少 $α$ 值（ 反之亦然）。举个例子，设 $μ = 0.9$ ，则更新值会扩大 $\frac{1}{1-0.9} = 10$ 倍。如果将 $μ$ 扩大为 0.99，那么更新值会扩大 100 倍，所以 $α$ 应该再除以 10。

上述技巧也只是经验之谈，不保证绝对有用，甚至可能一点用也没有。如果训练过程中出现了发散现象（例如， loss 或者 output 值非常大导致显示 NAN， inf 这些符号），试着减小基准学习速率（例如 base_lr 设置为 0.001）再训练，重复这个过程，直到找到一个比较合适的学习速率(base_lr)。

### 4.2.2 AdaDelta

<div align=center>
<img src="http://img.blog.csdn.net/20180403144115095">
</div>

### 4.2.3 AdaGrad

<div align=center>
<img src="http://img.blog.csdn.net/20180403144505556">
</div>

### 4.2.4 Adam

<div align=center>
<img src="http://img.blog.csdn.net/20180403144606885">
</div>

### 4.2.5 NAG

<div align=center>
<img src="http://img.blog.csdn.net/20180403144818098">
</div>

### 4.2.6 RMSprop

<div align=center>
<img src="http://img.blog.csdn.net/20180403144921927">
</div>

# 第五章 Layer Cataloge

**注：新版的caffe在代码布局上有所调整，但是基本实现方式完全相同。新版本中将`vision_layer.hpp`拆分成许多独立的小部分，每个小部分独立的描述一种 layer 结构，然后将这些 layer 收纳在`layers`文件夹中。**

    带有&号标记的为旧版本的代码布局和描述

caffe用一个 protocol buffer(prototxt)文件中定义模型的结构，层和相应的参数都定义在 caffe.proto 文件里。

##  5.1 &视觉层 Vision Layers
头文件： `./include/caffe/vision_layers.hpp`

图像的广义概念：高，宽及不限通道数的空间结构都可称之为图像数据，通常将输入图像看作是一个维度为 CHW 的“单个大向量”

### 5.1.1 卷积 Convolution
层类型 : `Convolution`

- CPU 实现代码 : ./src/caffe/layers/conv_layer.cpp
- CUDA,GPU
- 实现代码 : ./src/caffe/layers/conv_layer.cu
- 参数 (ConvolutionParameter convolution_param)

实例：`./models/bvlc_reference_caffenet/train_val.prototxt`
其中部分字段的含义
- group(g)[defult 1]:（ 译者注： 指定分组卷积操作的组数，默认为 1 即不分组）如果 g >1,我们可以将卷积核的连接限制为输入数据的一个子集。 具体地说, 输入图像和输出图像在通道维度上分别被分成 g 个组, 输出图像的第 i 组只与输入图像第 i 组连接（ 即输入图像的第 i 组与相应的卷积核卷积得到第 i 组输出）。
- pad (或者 pad_h 和 pad_w) [default 0]: 指定在输入图像周围补 0 的像素个数；

输入$n\times c_i\times h_i\times w_i$
输出$n\times c_o\times h_o\times w_o$,式中$h_o = \frac{h_i + 2\times pad_h - kernel_h}{stride_h}+1$,$W_o$可同理计算。

```
layer {
    name: "conv2"
    type: "Convolution"
    bottom: "norm1"
    top: "conv2"
    
    # 卷积核的局部学习率和权值衰减因子
    param { 
        lr_mult: 1 
        decay_mult: 1 
    }
    
    # 偏置项的局部学习率和权值衰减因子
    param { 
        lr_mult: 2 
        decay_mult: 0 
    }
    convolution_param {
        num_output: 256 # 学习 256 组卷积核
        pad:2 #
        kernel_size: 5 # 卷积核大小为 5x5
        group: 2
        # stride: 4 # 卷积核滑动步长为 4
        weight_filler {
            type: "gaussian" # 使用高斯分布随机初始化卷积核
            std: 0.01 # 高斯分布的标准差为 0.01 (默认均值： 0)
        }
        bias_filler {
            type: "constant" # 使用常数 0 初始化偏置项 0
            value: 0
        }
    }
}
```

### 5.1.2 池化 Pooling

层类型: Pooling

- CPU 实现代码: ./src/caffe/layers/pooling_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/pooling_layer.cu
- 参数 (PoolingParameter pooling_param)

实例：`./models/bvlc_reference_caffenet/train_val.prototxt`
其中部分字段的含义
- pool [default MAX]: 池化方法, 目前提供三种: 最大值池化, 均值池化,和 随机池化
- pad (或者 pad_h 和 pad_w) [default 0]: 指定在输入图像周围补 0 的像素个数

输入 $n\times c\times h_i\times w_i$
输出 $n\times c\times h_o\times w_o$, 式中$h_o = \frac{h_i + 2\times pad_h - kernel_h}{stride_h}+1$,$W_o$可同理计算。

```
layer {
    name: "pool1"
    type: "Pooling"
    bottom: "conv1"
    top: "pool1"
    pooling_param {
        pool: MAX
        kernel_size: 3 # 池化窗口大小为 3x3
        stride: 2 # 池化窗口在输入图像上滑动的步长为 2
    }
}
```

### 5.1.3 局部响应值归一化 Local Response Normalization (LRN)
层类型: LRN

- CPU 实现代码: ./src/caffe/layers/lrn_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/lrn_layer.cu
- 参数 (LRNParameter lrn_param)

实例：`./models/bvlc_reference_caffenet/train_val.prototxt`
其中部分字段的含义
- local_size [default 5]: 对于跨通道的归一化， 该参数指参与求和的通道数，对于通道内的规范化，该参数指的是参与求和的方形区域的边长
- alpha [default 1]: 尺度参数（见下文）
- beta [default 5]: 指数参数（见下文）
- norm_region [default ACROSS_CHANNELS]: 指定在通道之间进行规范化

局部响应值归一化层通过对输入数据的局部归一操作执行了一种“侧抑制”的机制。 
在 ACROSS_CHANNELS 模式下， 局部区域沿着临近通道延伸（ 译者注： 而非在特征图的平面内），而没有空间扩展（即局部区域的形状为 local_size x 1 x 1）。
在 WITHIN_CHANNEL 模式下， 局部区域在各自通道内部的图像平面上延伸（即局部区域的形状为 1 x local_size x local_size）。

每个输入值除以 $(1+(\alpha / n)\sum_i{x_i^2})^\beta$ 以实现归一化， 式中， n 是局部区域的大小，在以当前输入值为中心的区域内计算加和（如有需要，需在边缘补零）。

### 5.1.4 im2col

实现图像到“列向量”的转换

## 5.2 损失层 Loss Layers

### 5.2.1 Softmax 损失

层类型： SoftmaxWithLoss

softmax 损失层一般用于计算多类分类问题的损失，在概念上等同于 softmax 层后跟随一个多变量 Logistic 回归损失层(multinomial logistic loss)，但能提供数值上更稳定的梯度。

### 5.2.2 平方和/欧式损失 Sum-of-Squares / Euclidean

层类型: EuclideanLoss
Euclidean 损失层用来计算两个输入差值的平方和
$$\frac{1}{2N}\sum^N_{i=1}{||x_i^1 - x_i^2||^2_2}$$

### 5.2.3 Hinge / Margin 损失

> hinge 损失层用来计算 one-vs-all hinge 或者 squared hinge 损失。

层类型: HingeLoss
- CPU 实现代码: ./src/caffe/layers/hinge_loss_layer.cpp
- CUDA GPU 实现代码: 未实现
- 参数 (HingeLossParameter hinge_loss_param)
    - 可选
- norm [default L1]: 正则项类型，目前有 L1 和 L2
输入 
$n \times c \times h \times w$ 预测值
$n \times 1 \times 1 \times 1$ 真实标签
输出
$1 \times 1 \times 1 \times 1$ 计算的损失
示例：
```
# L1 Norm
layer {
    name: "loss"
    type: "HingeLoss"
    bottom: "pred"
    bottom: "label"
}
# L2 Norm
layer {
    name: "loss"
    type: "HingeLoss"
    bottom: "pred"
    bottom: "label"
    top: "loss"
    hinge_loss_param {
        norm: L2
    }
}
```

### 5.2.4 交叉熵损失 Sigmoid Cross-Entropy

### 5.2.5 信息熵损失 Infogain

### 5.2.6 准确率 Accuracy and Top-k

> Accuracy 用来计算网络输出相对目标值的准确率， 它实际上并不是一个损失层， 所以没有反传操作。

## 5.3 激活层 Activation / Neuron Layers

一般来说，激活层执行逐个元素的操作， 输入一个底层 blob， 输出一个尺寸相同的顶层 blob。

输入尺寸：$n\times c\times h\times w$
输出尺寸：$n\times c\times h\times w$

### 5.3.1 ReLU / Rectified-Linear and Leaky-ReLU

层类型: ReLU
- CPU 实现代码: ./src/caffe/layers/relu_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/relu_layer.cu
- 参数 (ReLUParameter relu_param)

字段含义：
- negative_slope [default 0]: 设置激活函数在负数部分的斜率（默认为 0），对输入数据小于零的部分乘以这个因子，斜率为 0 时，小于零的部分完全滤掉。

示例：`./models/bvlc_reference_caffenet/train_val.prototxt`

```
layer {
    name: "relu1"
    type: "ReLU"
    bottom: "conv1"
    top: "conv1"
}
```

给定一个输入数据， 当 x > 0 时， ReLU 层的输出为 x， 当 x <= 0 时， 输出为negative_slope * x。 当 negative_slope 未指定时，等同于标准的 ReLU 函数 max(x, 0)，该层也支持原（址 in-place）计算， 即它的底层 blob 和顶层 blob 可以是同一个以节省内存开销。

*注：其他激活函数有相似的用法，差别在于有不同的参数字段。*

### 5.3.2. Sigmoid

层类型: Sigmoid
- CPU 实现代码: ./src/caffe/layers/sigmoid_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/sigmoid_layer.cu

sigmodi(x)

### 5.3.3 TanH / Hyperbolic Tangent

层类型: TanH
- CPU 实现代码: ./src/caffe/layers/tanh_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/tanh_layer.cu

thnh(x)

### 5.3.4 Absolute Value

层类型: AbsVal
- CPU 实现代码: ./src/caffe/layers/absval_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/absval_layer.cu

abs(x)

### 5.3.5 Power

层类型: Power
- CPU 实现代码: ./src/caffe/layers/power_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/power_layer.cu

字段：
- power [default 1]
- scale [default 1]
- shift [default 0]

使用 $(shift + scale\times x)^{power}$ 计算每个输入 x 的输出

### 5.3.5 BNLL

层类型: BNLL
- CPU 实现代码: ./src/caffe/layers/bnll_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/bnll_layer.cu

使用 $log(1 + exp(x))$ 计算每个输入 x 的输出

## 5.4 数据层 Data Layers

数据能过数据层进入 caffe 网络：数据层处于网络的最底层， 数据可以从高效率的数据库中读取（如 LevelDB 或 LMDB）， 可以直接从内存中读取， 若对读写效率要求不高也可以从硬盘上的 HDFT 文件或者普通的图片文件读取.

常见的数据预处理操作（减均值，尺度变换，随机裁剪或者镜像）可以能过设定参数 `TransformationParameter` 来实现。

### 5.4.1 数据库 Database

层类型: Data
- 参数
    - 必填
        1. source: 数据库文件的路径
        2. batch_size: 网络单次输入数据的数量
    - 可选
        1. rand_skip: 跳过开头的 rand_skip * rand(0,1)个数据，通常在异步随机梯度下降法里使用；
        2. backend [default LEVELDB]: 选择使用 LEVELDB 还是 LMDB。

### 5.4.2 内存数据 In-Memory

层类型: MemoryData
- 参数
    - 必填
batch_size, channels, height, width: 指定从内存中读取的输入数据块的尺寸

memory data 层直接从内存中读取数据而不用拷贝。使用这个层时需要调用MemoryDataLayer::Reset (C++) 或者 Net.set_input_arrays(Python) 来指定数据来源（四维按行存储的数组）， 每次读取一个大小为 batch-sized 的数据块。

### 5.4.3 HDF5 Input

层类型: HDF5Data
- 参数
    - 必填
        1. source: 文件路径；
        2. batch_size。

### 5.4.4 HDF5 Output

层类型: HDF5Output
- 参数
    - 必填
        file_name: 写入文件的路径
        HDF5 output 层执行了一个与数据读取相反的操作， 它将数据写进硬盘。

### 5.4.5 图像数据 Images

层类型: ImageData
- 参数
    - 必填
        1. source: text 文件的路径名，该 text 文件的每一行存储一张图片的路径名和对应的标签；
        2. batch_size: 打包成 batch 的图片数量。
    - 可选
    1. rand_skip
    2. shuffle [default false]
    3. new_height, new_width: 根据设置的值，输入的图片将会被调整成给定的尺寸。

### 5.4.6 窗口 Windows
### 5.4.7 Dummy

## 5.5 普通层 Common Layers
### 5.5.1 内积／全连接 Inner Product

InnerProduct 层（也被称作全连接层）将输入看成一个一向量，输出也为向量（输出 blob 的高和宽都为 1）

层类型: InnerProduct
- CPU 实现码: ./src/caffe/layers/inner_product_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/inner_product_layer.cu
- 参数 (InnerProductParameter inner_product_param)
    - 必填
        num_output (c_o): 层的输出节点数或者理解为滤波器的个数
    - 推荐
        weight_filler [default type: 'constant' value: 0]
    - 可选
        1.bias_filler [default type: 'constant' value: 0]
        2.bias_term [default true]: 指定是否给每个滤波器添加并训练偏置项

输入 $n \times c_i \times h_i \times w_i$
输出 $n \times c_o \times 1 \times 1$

示例:`./models/bvlc_reference_caffenet/train_val.prototxt`

```
layer {
    name: "fc8"
    type: "InnerProduct"
    bottom: "fc7"
    top: "fc8"
    # learning rate and decay multipliers for the weights
    param { 
        lr_mult: 1 
        decay_mult: 1 
    }
    # learning rate and decay multipliers for the biases
    param { 
        lr_mult: 2 
        decay_mult: 0 
    }
    inner_product_param {
        num_output: 1000
            weight_filler {
            type: "gaussian"
            std: 0.01
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
}
```

### 5.5.2 分裂 Splitting

The Split 是一个可以将输入的 blob 分裂（复制）成多个输出 blob 的功能层， 通常当一个 blob 需要给多个层作输入数据时该层会被使用。

### 5.5.3 摊平 Flattening

flatten 层用来将尺寸为 $n \times c \times h \times w$ 的输入 blob 转换成一个尺寸为 $n \times (c\times h\times w)$ 的输出 blob。

### 5.5.4 变形 Reshape

层类型: Reshape
- 实现代码: ./src/caffe/layers/reshape_layer.cpp
- 参数 (ReshapeParameter reshape_param)
    - 可选: (见下文)
    shape
        - 输入: 任意维度的 blob
        - 输出: 按照参数 reshape_param 修改维度的 blob

示例

```
layer {
    name: "reshape"
    type: "Reshape"
    bottom: "input"
    top: "output"
    reshape_param {
        shape {
            dim: 0 # copy the dimension from below
            dim: 2
            dim: 3
            dim: -1 # infer it from the other dimensions
        }
    }
}
```

Reshape 层在不改变数据的情况下改变输入 blob 的维度，和 Flatten 操作一样，处理过程只在输入 blob 上进行，没有进行数据的拷贝。输出数据的维度通过参数 ReshapeParam 设定， 可以使用正数直接指定输出 blob 的相
应维度， 也可以使用两个特殊的值来设定维度：

- 0 表示从层的底层 blob 中直接取相应的维度，即不改变对应的维度。 例如，在参数中设置第一个维度为 dim: 0，底层 blob 在第一个维度上是 2，则输出的顶层 blob 的第一个维度也是 2

- -1 表示用其它的维度计算该维度的值。这个操作与 numpy 中的－1 或者 MATLAB 中 reshap 操作时的[]相似：对应的维度通过保证 blob 中数据的总个数不变来计算， 因此，在 reshape 操作中最多设定一个－1

因此通过设定参数 reshape_param { shape { dim: 0 dim: -1 } }我们可以得到与 flatten 层相同的操作结果。

### 5.5.5 连结 Concatenation

Concat 层用来将多个输入 blob 连结成一个 blob 输出

层类型: Concat
- CPU 实现代码: ./src/caffe/layers/concat_layer.cpp
- CUDA GPU 实现代码: ./src/caffe/layers/concat_layer.cu
- 参数 (ConcatParameter concat_param)
    - 可选
        axis [default 1]: 0 表示沿着样本个数的维度(num)串联， 1 表示沿着通道维度(channels)串联。

输入： $n_i * c_i * h * w$ 对于第 i 个输入 blob， i 的取值为{1,2,...,K}
输出：
1. if axis = 0: $(n_1 + n_2 + ... + n_K) * c_1 * h * w$， 所有输入 blob 在通道上的维度 c_i 需相同；
2. if axis = 1: $n_1 * (c_1 + c_2 + ... + c_K) * h * w$， 所有输入 blob 在样本个数上的维度 n_i 需相同。

### 5.5.6 切片 Slicing

Slice 层按照给定维度（ num 或者 channel）和切分位置的索引将一个输入 blob 分成多个 blob 输出。

axis 指定执行切分操作的所在的维度； slice_point 指定所选维度上切分位置的索引（索引的个数需等于顶层 blob 的个数减 1）。

### 5.5.7 逐个元素操作 Elementwise Operations
### 5.5.8 Argmax
### 5.5.9 Softmax
### 5.5.10 Mean-Variance Normalization

# 第六章 Interfaces
## 6.1 Command Line

命令行接口 - cmdcaffe - 是 caffe 中用来模型训练，计算得分以及方法判断的工具。没有附加参数的情况下运行 caffe 可得到帮助提示。

### 6.1.2 训练

caffe train 命令可以从零开始学习模型, 也可以从已保存的 snapshots 继续学习, 或将已经训练好的模型应用在新的数据与任务上进行微调即 fine-tuning 学习：
- 所有的训练都需要添加-solver solver.prototxt 参数完成 solver 的配置。
- 继续训练需要添加 -snapshot model_iter_1000.solverstate 参数来加载 solver snapshot。
- Fine-tuning 需要添加-weights model.caffemodel 参数完成模型初始化。

示例：

```
# 训练 LeNet
caffe train -solver examples/mnist/lenet_solver.prototxt
# 在 2 号 GPU 上训练
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 2
# 从中断点的 snapshot 继续训练
caffe train -solver examples/mnist/lenet_solver.prototxt -snapshot examples/mnist/lenet_iter_5000.solverstate
```

对于 fine-tuning 的完整例子，可以参考 examples/finetuning_on_flickr_style，但是只调用训练命令如下：

```
# 微调 CaffeNet 模型的权值以完成风格识别任务（ style recognition）
caffe train –solver examples/finetuning_on_flickr_style/solver.prototxt -weights  models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
```

### 6.1.2 测试

caffe test 命令通过在 test phase 中运行模型得到分数，并且用这分数表示网络输出的最终结果。网络结构必须被适当定义，生成 accuracy 或 loss 作为其结果。测试过程中，终端会显示每个 batch 的得分，最后输出全部 batch 得分的平均值。

```
# 对于网络结构文件 lenet_train_test.prototxt 所定义的网络
# 用 validation set 得到已训练的 LeNet 模型的分数
caffe test -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 100
```

### 6.1.3 Benchmarking

caffe time 命令 通过逐层计时与同步，执行模型检测。这是用来检测系统性能与测量模型相对执行时间。

```
# (这些例子要求你先要完成 LeNet / MNIST 的例子)
# 在 CPU 上， 10 iterations 训练 LeNet 的时间
caffe time -model examples/mnist/lenet_train_test.prototxt -iterations 10
# 在 GPU 上，默认的 50 iterations 训练 LeNet 的时间
caffe time -model examples/mnist/lenet_train_test.prototxt -gpu 0
# 在第一块 GPU 上， 10 iterations 训练已给定权值的网络结构的时间
caffe time -model examples/mnist/lenet_train_test.prototxt -weights examples/mnist/lenet_iter_10000.caffemodel -gpu 0 -iterations 10
```

### 6.1.4 诊断

caffe device_query 命令对于多 GPU 机器上，在指定的 GPU 运行，输出 GPU 细节信息用来参考与检测设备序号。

```
# 查询第一块 GPU
caffe device_query -gpu 0
```

### 6.1.5 并行模式

caffe 工具的 -gpu 标识，对于多 GPU 的模式下，允许使用逗号分开不同 GPU 的 ID 号。solver 与 net 在每个 GPU 上都会实例化，因此 batch size 由于具有多个 GPU 而成倍增加，增加的倍数为使用的 GPU块数。如果要重现单个 GPU的训练，可以在网络定义中适当减小 batchsize。

```
# 在序号为 0 和 1 的 GPU 上训练 ( 双倍的 batch size )
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu 0,1
# 在所有 GPU 上训练 ( 将 batch size 乘上 GPU 数量)
caffe train -solver examples/mnist/lenet_solver.prototxt -gpu all
```

## 6.2 Python

Python 接口 – pycaffe – 是 caffe 的一个模块，其脚本保存在 caffe/python。通过 import caffe 加载模型，实现 forward 与 backward、 IO、网络可视化以及求解模型等操作。所有的模型数据，导数与参数都可读取与写入。

- caffe.Net 是加载、配置和运行模型的中心接口
- caffe.Classsifier 与 caffe.Detector 为一般任务实现了便捷的接口
- caffe.SGDSolver 表示求解接口
- caffe.io 通过预处理与 protocol buffers，处理输入/输出
- caffe.draw 实现数据结构可视化
- Caffe blobs 通过 numpy ndarrays 实现易用性与有效性

IPython notebooks 教程在 caffe/examples: 可通过 ipython notebook caffe/examples 来尝试使用。对于开发者的参考提示贯穿了整个代码。

make pycaffe 可编译 pycaffe。 通过 `export PYTHONPATH= /path/to/caffe/python: $PYTHONPATH` 将模块目录添加到自己的 `$PYTHONPATH` 目录，或者相似的指令来实现 `import caffe`。

## 6.3 MATLAB

MATLAB 接口(matcaffe)是在 caffe/matlab 路径中的 caffe 软件包。在 matcaffe 的基础上，可将 Caffe 整合进你的 Matlab 代码中。

在 MatCaffe 中，你可实现：
- 在 Matlab 中创建多个网络结构(nets)
- 进行网络的前向传播(forward)与反向传导(backward)计算
- 存取网络中的任意一层，或者网络层中的任意参数
- 在网络中，读取数据或误差，将数据或误差写入任意层的数据(blob)，而不是局限在输入 blob 或输出 blob
- 保存网络参数进文件，从文件中加载
- 调整 blob 与 network 的形状
- 编辑网络参数，调整网络
- 在 Matlab 中，创建多个 solvers 进行训练
- 从 solver 快照(snapshots)恢复并继续训练
- 在 solver 中， 访问训练网络(train nets)与测试网络(test nets)
- 迭代一定次数后将网络结构交回 Matlab 控制
- 将梯度方法融合进任意的 Matlab 代码中

caffe/matlab/demo/classification_demo.m 中有一个 ILSVRC 图片的分类 demo（需要在Model Zoo 中下载 BVLC CaffeNet 模型）。

### 6.3.1 编译 MatCaffe

### 6.3.2 使用 MatCaffe
#### 6.3.2.1 设置运行模式和 GPU 设备
#### 6.3.2.2 创建网络，保存及读取其 layers 和 blobs
#### 6.3.2.3 网络的前向传导(forward)与后向传播(backward)
#### 6.3.2.4 调整网络形状
#### 6.3.2.5 训练网络
#### 6.3.2.6 Input and output
#### 6.3.2.7 清除 Nets 和 Solvers

# 第七章 数据
## 7.1 数据：输入与输出
Caffe 中数据流以 Blobs 进行传输。 数据层将输入转换为 blob 加载数据，将 blob 转换为其他格式保存输出。 均值消去、特征缩放等基本数据处理都在数据层进行配置。新的数据格式输入需要定义新的数据层，网络的其余部分遵循caffe 中层目录的模块结构设定。

数据层的定义：

```
layer {
    name: "mnist"
    # 数据层加载 leveldb 或 lmdb 的数据库存储格式保证快速传输
    type: "Data"
    
    # 第一个顶部（ top）是数据本身：“ data”的命名只是方便使用
    top: "data"
    
    # 第二个顶部（ top）是数据标签：“ label”的命名只是方便使用
    top: "label"
    
    # 数据层具体配置
    data_param {
        # 数据库路径
        source: "examples/mnist/mnist_train_lmdb"
        # 数据库类型： LEVELDB 或 LMDB（ LMDB 支持并行读取）
        backend: LMDB
        # 批量处理，提高效率
        batch_size: 64
    }
    
    # 常用数据转换
    transform_param {
        # 特征归一化系数，将范围为[0, 255]的 MNIST 数据归一化为[0, 1]
        scale: 0.00390625
    }
}
```

**预获取（ Prefetching）**： 为了提高网络吞吐量，数据层在网络计算当前数据块的同时在后台获取并准备下一个数据块。

数据预处理通过转换参数来定义

```
layer{
    name: "data"
    Type: "Data"
    [...]
    transform_param{
        scale: 0.1
        mean_file_size: "\xx\xx\mean.binaryproto"
        # 对 images 进行水平镜像处理或者随机裁剪处理
        # 可作为简单的数据增强处理
        mirror: 1 # 1 = on; 0 = off
        # 裁剪块大小为 `crop_size` x `crop_size`:
        # - 训练时随机处理
        # - 测试时从中间开始
        crop_size: 227
    }
}
```



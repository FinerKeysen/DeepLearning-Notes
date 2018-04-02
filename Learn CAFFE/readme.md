*申明：部分内容摘自以下博客*

[卜居：Caffe 增加自定义 Layer 及其 ProtoBuffer 参数](https://blog.csdn.net/kkk584520/article/details/52721838)

[wanggao_1990：Caffe添加自定义的层 ](https://blog.csdn.net/wanggao_1990/article/details/78863669)

[BVLC：Developing new layers](https://github.com/BVLC/caffe/wiki/Development)

在使用 `Caffe` 过程中经常会有这样的需求：已有 `Layer` 不符合我的应用场景；我需要这样这样的功能，原版代码没有实现；或者已经实现但效率太低，我有更好的实现。

#### 方案一：简单粗暴的解法——偷天换日


如果你对 ConvolutionLayer 的实现不满意，那就直接改这两个文件：`$CAFFE_ROOT/include/caffe/layers/conv_layer.hpp` 和 `$CAFFE_ROOT/src/caffe/layers/conv_layer.cpp` 或 `conv_layer.cu` ，将 `im2col` + `gemm` 替换为你自己的实现（比如基于 `winograd 算法`的实现）。

优点：快速迭代，不需要对 `Caffe` 框架有过多了解，糙快狠准。

缺点：代码难维护，不能 `merge` 到 `caffe master branch`，容易给使用代码的人带来困惑（效果和 `#define TRUE false` 差不多）。


#### 方案二：稍微温柔的解法——千人千面

和方案一类似，只是通过预编译宏来确定使用哪种实现。例如可以保留 `ConvolutionLayer` 默认实现，同时在代码中增加如下段：

```c++
#ifdef SWITCH_MY_IMPLEMENTATION  
// 你的实现代码  
#else  
// 默认代码  
#endif 
```

这样可以在需要使用该 `Layer` 的代码中，增加宏定义：

```c++
#define SWITCH_MY_IMPLEMENTATION  
```

就可以使用你的实现。而未定义该宏的代码，仍然使用原版实现。

优点：可以在新旧实现代码之间灵活切换；

缺点：每次切换需要重新编译；


#### 方案三：优雅转身——山路十八弯

同一个功能的 `Layer` 有不同实现，希望能灵活切换又不需要重新编译代码，该如何实现？

这时不得不使用 `ProtoBuffer` 工具了。

首先，要把你的实现，要像正常的 `Layer` 类一样，分解为声明部分和实现部分，分别放在 `.hpp` 与 `.cpp`、`.cu` 中。`Layer` 名称要起一个能区别于原版实现的新名称。`.hpp` 文件置于 `$CAFFE_ROOT/include/caffe/layers/`，而 `.cpp` 和 `.cu` 置于 `$CAFFE_ROOT/src/caffe/layers/`，这样你在 `$CAFFE_ROOT` 下执行 `make` 编译时，会自动将这些文件加入构建过程，省去了手动设置编译选项的繁琐流程。

其次，在 `$CAFFE_ROOT/src/caffe/proto/caffe.proto` 中，增加新 `LayerParameter` 选项，这样你在编写 `train.prototxt` 或者 `test.prototxt` 或者 `deploy.prototxt` 时就能把新 `Layer` 的描述写进去，便于修改网络结构和替换其他相同功能的 `Layer` 了。

最后也是最容易忽视的一点，在 `Layer` 工厂注册新 `Layer` 加工函数，不然在你运行过程中可能会报如下错误：
```c++
F1002 01:51:22.656038 1954701312 layer_factory.hpp:81] Check failed: registry.count(type) == 1 (0 vs. 1) Unknown layer type: AllPass (known types: AbsVal, Accuracy, ArgMax, BNLL, BatchNorm, BatchReindex, Bias, Concat, ContrastiveLoss, Convolution, Crop, Data, Deconvolution, Dropout, DummyData, ELU, Eltwise, Embed, EuclideanLoss, Exp, Filter, Flatten, HDF5Data, HDF5Output, HingeLoss, Im2col, ImageData, InfogainLoss, InnerProduct, Input, LRN, Log, MVN, MemoryData, MultinomialLogisticLoss, PReLU, Pooling, Power, ReLU, Reduction, Reshape, SPP, Scale, Sigmoid, SigmoidCrossEntropyLoss, Silence, Slice, Softmax, SoftmaxWithLoss, Split, TanH, Threshold, Tile, WindowData)  
*** Check failure stack trace: ***  
    @        0x10243154e  google::LogMessage::Fail()  
    @        0x102430c53  google::LogMessage::SendToLog()  
    @        0x1024311a9  google::LogMessage::Flush()  
    @        0x1024344d7  google::LogMessageFatal::~LogMessageFatal()  
    @        0x10243183b  google::LogMessageFatal::~LogMessageFatal()  
    @        0x102215356  caffe::LayerRegistry<>::CreateLayer()  
    @        0x102233ccf  caffe::Net<>::Init()  
    @        0x102235996  caffe::Net<>::Net()  
    @        0x102118d8b  time()  
    @        0x102119c9a  main  
    @     0x7fff851285ad  start  
    @                0x4  (unknown)  
Abort trap: 6 
```

下面给出一个[实际案例](https://blog.csdn.net/wanggao_1990/article/details/78863669)，走一遍方案三的流程。

这里我们实现一个新 `Layer`，名称为 `AllPassLayer`，顾名思义就是全通 `Layer`，“全通”借鉴于信号处理中的全通滤波器，将信号无失真地从输入转到输出。

虽然这个 `Layer` 并没有什么卵用，但是在这个基础上增加你的处理是非常简单的事情。另外也是出于实验考虑，全通层的 `Forward/Backward` 函数非常简单不需要读者有任何高等数学和求导的背景知识。读者使用该层时可以插入到任何已有网络中，而不会影响训练、预测的准确性。

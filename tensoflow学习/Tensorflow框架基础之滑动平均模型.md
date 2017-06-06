---
title: tensoflow学习/Tensorflow框架基础之滑动平均模型 
tags: tensorflow
grammar_cjkRuby: true
---



在采用随机梯度下降算法训练神经网络时，使用滑动平均模型在一定程度上可以提高模型在测试数据上的表现。

method：`tf.train.ExponentialMovingAverage(decay, num_updates=none)`

初始化`ExponentialMovingAverage`时提供一个衰减率（decay），用于控制模型更新的速度。`ExponentialMovingAverage` 对每一个（待更新训练学习的）变量（variable）都会维护一个影子变量（shadow variable）。影子变量的初始值就是这个变量的初始值，更新的方式：
$$shadow\_variable=decay\times shadow\_variable+(1-decay)\times variable$$
`shadow_variable`为影子变量，`variable`为待更新变量，`decay`为衰减率。实际应用中，`decay`一般设置为接近1的数（如0.999或0.9999）。为使得模型在训练前期更新的更快，`ExponentialMovingAverage`提供了`num_updates`参数来动态的设置`decay`的大小，如果提供了`num_updates`，那么`decay`的取值为：
$$min\{ decay, \frac{1+num\_updates}{10+num\_updates}\}$$

举例说明
```python

```
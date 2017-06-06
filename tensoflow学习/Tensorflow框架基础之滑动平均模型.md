---
title: tensoflow学习/Tensorflow框架基础之滑动平均模型 
tags: tensorflow
grammar_cjkRuby: true
---



在采用随机梯度下降算法训练神经网络时，使用滑动平均模型在一定程度上可以提高模型在测试数据上的表现。

method：`tf.train.ExponentialMovingAverage(decay, num_updates)`

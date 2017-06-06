---
title: tensorflow框架基础之Variable 
tags: tensorflow
grammar_cjkRuby: true
---


> 变量：保存和更新网络中的参数

- 变量的初始化
```python
W = tf.Variable(tf.random_normal( # 初始化方法：正态分布
                                [2,3], # 大小：2*3矩阵
                                mean=1, # 均值mean:1,默认值:0
                                stddev=2) # 标准差stddev:2
                )
```
**初始化方法**

1.随机数生成函数
|函数|分布|主要参数|
|:---:|:---:|:---:|
|tf.random_normal|正态分布|mean、stddev、type|
|tf.truncated_normal|正态分布,一旦随机数偏离mean值2个stddev，就重新随机生成|mean、stddev、type|
|tf.random_uniform|平均分布|最大、最小值，type|
|tf.random_gamma|Gama分布|形状参数alpha、尺度参数beta、type|

2.常数生成函数
|函数|功能|例子|
|:---:|:---:|:---:|
|tf.zeros|全0||
|tf.ones|全1||
|tf.fill|按给定数填满|tf.fill([2,3],4) -> [[4,4,4],[4,4,4]]|
|tf.constant|按给定值填写|tf.constant([1,2,3]) -> [1,2,3]|

3.按其他变量初始值初始化
```python
w2 = tf.Variable(W.initialized_value()*2.0)
```
`c = tf.matmul(a,b)`，在计算`c`之前，要先初始化所用到的变量，再计算输出
```python
with tf.Session() as sess:
    sess.run(a.initializer)
    sess.run(b.initializer)

    print(sess.run(c))
```
一旦变量过多，则操作起来会很不方便，所以`TF`中可以统一的处理所有变量的初始化过程，就不需要一个个的初始化啦
```python
init_ops = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_ops)
    sess.run(...)
```

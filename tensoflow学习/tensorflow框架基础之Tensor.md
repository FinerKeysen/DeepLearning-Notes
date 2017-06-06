---
title: tensorflow学习/tensorflow框架基础之Tensor 
tags: tensorflow
grammar_cjkRuby: true
---

 
### Tensor的概念
> 在Tensorflow中，所有的数据都通过张量的形式表示

零阶张量表示标量，即一个数；一阶张量为向量，即一维数组；n阶张量理解为一个n维的数组；*但是张量不真正的保存数字，它保存的是如何得道这些数字的计算过程，即操作*。

```python
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print result

'''
result：
Tensor("add:0", shape=(2,), dtype=float32)
'''
```
运行得到的是对结果的引用，而不是加法的结果。这种结果也是一个张量，主要保存了三个重要属性(当然也具有其他属性)：*name、shape(张量维度)、type(张量类型).*

**TF** 的计算由计算图模型构成，每个节点代表一个计算，计算的结果保存在张量中。因此计算图的节点所代表的计算结果与张量相对应。

*name*
张量的唯一标识符，同时可以看出他的计算方式，如**result**是*add*加法计算后的结果。
张量的命名形式：“node:src_output”；  

 * node：节点名称
 * src_output：张量来自节点的第几个输出(从0开始编号)

### Tensor的使用
```python
# 第一种：使用张良记录中间结果
a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([1.0, 2.0], name='b')
result = a + b

# 第二种：直接计算向量的结果，但可读性差
c = tf.constant([1.0, 2.0], name='a') + tf.constant([1.0, 2.0], name='b')
```


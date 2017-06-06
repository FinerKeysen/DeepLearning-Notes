---
title: tensorflow框架基础之优化(二) 
tags: tensorflow
grammar_cjkRuby: true
---


## tensorflow框架基础之优化(二)
### 防止过拟合

当神经网络得到的是一个过拟合模型时，这个模型基本是没什么应用价值的，因为它的泛化性能非常不好(*泛化即是，机器学习模型学习到的概念在它处于学习的过程中时模型没有遇见过的样本时候的表现，简单理解为预测能力*)，对一些''异常''数据过分的估计，而忽视了问题的整体规律。

为避免过拟合，常采用的方式是添加正则化项，正则化*通过限制权重大小，使得模型不能任意拟合训练数据中的随机噪声*。一般有两种正则化方式：
- L1正则化
$$R(w)=\Vert w \Vert_1=\sum_i\vert w_i\vert$$

- L2正则化
$$R(w)=\Vert w \Vert^2_2=\sum_i\vert w_i\vert^2$$

两种方式的区别参考[L1、L2范数的比较](http://blog.csdn.net/zouxy09/article/details/24971995/)

当然，正则化也可以是多种方式的组合，如$R(w)=\sum_i{\alpha \vert w_i\vert+(1-\alpha)w_i^2}$

所以，损失函数转换成$J(\theta)+\lambda R(w)$，在tensorflow中实现正则项
```python
weights = tf.constant([[1, 2],[3, 4]])
lambda = 0.2
# L1范数，regu=(|1|+|2|+|3|+|4|)*0.2
regu1 = tf.contrib.layers.l1_regularizer(lambda)(weights)

# L2范数(TF会将L2的结果除2，使得求导的结果更简洁)，regu=(|1|^2+|2|^2+|3|^2+|4|^2)*0.2/2
regu2 = tf.contrib.layers.l2_regularizer(lambda)(weights)
```
在多层神经网络中，这样对每层的weights都进行正则化的处理会显得繁琐臃肿，tensorflow也提供了一种集合`collection`的概念，它在一个计算图`tf.Graph`中保存一组实体。

example：
```python
import tensorflow as tf

# 定义一个获取权重并自动加入正则项到损失的函数
def get_weight(shape, lambda1):
    # 生成变量
	var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将变量var的L2正则损失加入名为'losses'的collection中
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
	# 返回所生成的变量
    return var

# 加载训练数据
data = []
label = []
...	
data_len = len(data)
	
# 定义网络
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

batch_size = 8

# 定义每层的节点的个数
layer_dimension = [2,8,8,8,1]

# 网络的层数
n_layers = len(layer_dimension)

# 开始为输入层，中间层时作为下层的输入层
cur_layer = x
# 当前层的节点数
in_dimension = layer_dimension[0]

# 生成网络结构
for i in range(1, n_layers):
	# 下层的节点数
    out_dimension = layer_dimension[i]
	# 当前层与下层间中权值，并将其加入到L2正则项losses中
    weight = get_weight([in_dimension, out_dimension], 0.003)
	# 偏置
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
	# 经过ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
	# 进入下层之前，更新为当前层输出节点数
    in_dimension = layer_dimension[i]

y= cur_layer

# 损失函数的定义
mse_loss = tf.reduce_sum(tf.square(y_ - y)) / sample_size
# 将均方误差添加到损失集合
tf.add_to_collection('losses', mse_loss)
# get_collection 返回一个列表，它包含这个集合中的所有元素，在该例子中也就是
# 损失函数的各个部分，将它们加起来得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))

# 定义训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
TRAINING_STEPS = 40000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
		start = (i*batch_size) % data_len
        end = (i*batch_size) % data_len + batch_size
        sess.run(train_step, feed_dict={x: data[start:end], y_: label[start:end]})
        if i % 2000 == 0:
            print("After %d steps, loss: %f" % (i, sess.run(loss, feed_dict={x: data, y_: label})))
```


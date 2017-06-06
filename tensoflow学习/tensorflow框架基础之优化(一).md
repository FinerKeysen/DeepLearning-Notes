---
title: tensorflow框架基础之优化(一) 
tags: tensorflow匠
grammar_cjkRuby: true
---


## Tensorflow框架基础之优化(一)
### 反向传播算法和梯度下降算法 

*梯度下降算法*主要用于优化单个参数的取值，[反向传播算法](https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf)则以一种高效的方式在所有的参数上使用梯度下降算法

若用$\theta$表示神经网络的参数，$J(\theta)$表示整个网络的损失函数，那优化过程就是找到一个参数$\theta$使得$J(\theta)$最小。梯度下降法以迭代的方式沿着梯度的反方向(也即是让参数朝着总损失更小的方向)更新参数$\theta$
$$\theta_{n+1}=\theta_n-\alpha\frac{\partial J(\theta_n)}{\partial \theta_n}$$其中$\alpha$为学习率，定义了每次参数更新的幅度(通俗讲，也就是参数每次移动的幅度)

梯度下降法的问题
- 不能保证被优化函数达到全局最优解，可能在达到局部最优时就停止更新
> a. 参数的初始化有很大影响；b. 损失函数尽可能为凸函数

- 计算時間太长
> 为加速训练，可采用随机梯度下降法：该算法优化的不是在全部训练数据上的损失函数，而是在每一轮迭代中，随机优化某一条训练数据上的损失函数，由此加快了每次迭代的更新速度。但又得必有失，随机梯度下降法所得到的最小损失代表不了全部数据的最小损失，甚至有可能达不到局部最优

*结合梯度下降和随机梯度下降的优缺点，折中的方式是：每次计算一小部分训练数据的损失函数。这一小部分数据也就是我们常看到的batch，batch的利用使得在每次迭代中优化的参数不至于太少；又能够减少达到收敛的迭代次数，使收敛结果更接近梯度下降的效果*

网络训练的一般过程
```python
# 给一个合适大小的batch
batch_size = n

# 每次读取batch_size个数据作为当前迭代的训练数据来执行 BP算法
x = tf.placeholder(tf.float32, shape=(batch_size, 2), name='x_input')
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name='y_input')

# 网络结构及优化算法
loss = ...
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 训练网络
init_op = ...
with tf.Sesion as sess:
    # 变量初始化
    sess.run(init_op)
    ...
    
    # 迭代参数，更新
    for i in range(STEPS):
        # 每次准备batch_size个训练数据
        current_x, current_y = ...
        
        # 喂入数据开始训练
        sess.run(train_step, feed_dict={x:current_x, y:current_y})
```
tensorflow中常用的优化器，[官方文档](https://www.tensorflow.org/versions/r0.11/api_docs/python/train/)，[博客讲解](http://blog.csdn.net/xierhacker/article/details/53174558)，[莫烦视频讲解](http://v.youku.com/v_show/id_XMTYwMzk1NDM4OA==.html)
> **GradientDescentOptimizer**
**MomentumOptimizer**
**AdamOptimizer**
AdagradOptimizer
AdagradDAOptimizer
FtrlOptimizer
RMSPropOptimizer  

### 学习率的设置
学习率控制参数更新的速度，也即是参数每次更新的幅度。`learn_rate`过大，会导致参数在最优值附近来回移动；`learn_rate`过小会影响优化的速度；

在tensorflow中提供了一个折中的设置方法-指数衰减法，通过`tf.train.exponential_decay`函数实现：在开始的时候有较大的学习率以加快优化的速度，在逐步迭代过程中衰减学习率，以保持训练的稳定

函数的使用
```python
# 函数调用
tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=False, name=None)

# exponential_decay()函数返回值decayed_learning_rate
decayed_learning_rate = learning_rate *
                        decay_rate ^ (global_step / decay_steps)
'''
decayed_learning_rate:每轮优化时使用的学习率
learning_rate:预设定的初始学习率
decay_rate：衰减系数
decay_steps:率减速度，通常代表完整的使用一遍训练数据所用需要的迭代次数，也就是总训练样本数/每个batch里的训练样本数
staircase：default is False，学习率的曲线是平滑的；当staircase is True，global_step/decay_steps会取整，学习率的曲线是阶梯型的
'''                     
```
举例：
```python
TRAINING_STEPS = 100
global_step = tf.Variable(0)
# 生成学习率
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 10 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print "After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value)
```
一般而言，初始学习率、衰减系数、率减速度会依据经验设置，损失函数下降的速度与训练完成后总损失的大小没有必然的联系，因此并不能通过损失函数下降的速度来比较不同的神经网络效果。


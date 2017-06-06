---
title: tensorflow框架基础之简单网络训练 
tags: tensorflow
grammar_cjkRuby: true
---


大概为三个过程：
- 定义神经网络的结构和前向传播的输出
- 定义loss函数、选择反向传播优化的算法
- 生成会话、反复运行反向传播优化算法并喂入数据

### placeholder
> `tensorflow`提供`placeholder`机制用于提供输入数据，`placeholder`相当于定义了一个位置，这个位置中的数据在程序运行时再指定

该机制使得在程序中不需要生成大量的常量来提供输入数据，而只需要将数据通过`placeholder`传入计算图。定义`placeholder`定义时，要指定数据类型，不可改变。`placholder`的维度可以根据提供的数据推导出来，故不一定要给出

简单示例
```python
import tensorflow as tf

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方，维度不一定要定义
x = tf.placeholder(tf.float32, name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()  
sess.run(init_op)

print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))

sess.close()

'''结果：
[[ 3.95757794]
 [ 1.15376544]
 [ 3.16749239]]
'''
```
然后可以定义反向传播的算法，包含损失函数及其优化方法，常用的优化方法：`tf.train.GradientDescentOptimizer()`,`tf.train.AdamOptimizer()`,`tf.train.MomentumOptimizer()`.
```python
# 损失函数，描述预测值与真实值的差距
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
# 优化方法，使得达到最小的loss
lr = 0.001 # 学习率
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
```

完整示例
```python
import tensorflow as tf
from numpy.random import RandomState

# 定义batch大小
batch_size = 5

w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义placeholder作为存放输入数据的地方，维度不一定要定义
x = tf.placeholder(tf.float32, name="inputX")
y = tf.placeholder(tf.float32, name="inputY")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 损失函数，描述预测值与真实值的差距
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
# 优化方法，使得达到最小的loss
lr = 0.001 # 学习率
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# 生成输入数据(train data)
rdm = RandomState(1)
sample_size = 96
X = rdm.rand(sample_size, 2)
# label, 0或1样本标签
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  
    print(sess.run(y, feed_dict={x: [[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))
    sess.run(init_op)

    print("Before training, the weights values(w1 & w2):")
    print sess.run(w1)
    print sess.run(w2)
    
    # 训练次数
    steps = 5000
    for i in range(steps):
	# 每次选batch_size个样本
	start = (i*batch_size)%sample_size
	end = min(start+batch_size, saample_size)

	# 通过选取的样本训练网络并更新参数
	sess.run(train_step, feed_dict={x:X[start:end],y:Y[start:end]})
	if i%1000 == 0:
	    # 每隔一段时间计算所有数据上的交叉熵
	    total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y:Y})
	    print("After %d training steps, cross entropy on all data is %g" % (i, total_cross_entropy))

    print("After training, the weights values(w1 & w2):")
    print sess.run(w1)
    print sess.run(w2)

'''结果：

'''
```

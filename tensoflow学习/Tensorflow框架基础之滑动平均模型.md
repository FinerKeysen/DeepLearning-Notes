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
import tensorflow as tf

# 定义一个初始值为0的变量，用于滑动平均
v1 = tf.Variable(0, dtype=tf.float32)
# 这里的step用于模拟迭代次数，可用于动态控制decay，即num_updates=step
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类，初始decay=0.99, num_updates=step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 给定一个列表，每次执行该操作时，列表中的变量都会被更新
maintain_averages_op = ema.apply([v1]) 

with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print sess.run([v1, ema.average(v1)]) # 输出[0.0, 0.0]
    
    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
	# decay=min{0.99, 1/10}=0.1
	# shadow=0.1*0+(1-0.1)*5=4.5
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)]) # 输出[5.0, 4.5]
    
    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))  
    sess.run(tf.assign(v1, 10))
	# decay=min{0.99, 10001/10010}=0.99
	# shadow=0.99*4.5+(1-0.99)*10=4.555
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])  # 输出[10.0, 4.555]     
    
    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)]) 
```
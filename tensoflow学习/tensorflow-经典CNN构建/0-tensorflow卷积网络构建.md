### tensorflow卷积网络构建

[Classic_CNN][1]

dropout——[Hinton G E, Sruvastava N, Krizhevsky A, et al. *Improving neural networks by preventing co-adaptation of feature detectors*[J]. Computer Science, 2012.]()

#### 卷积层和池化层的实现的主要步骤  
```python
filter_weight = tf.get_variable(‘weights’, [5, 5, 3, f_num], initializer=tf.truncated_normal_initializer(stddev=0.1))
biases = tf.get_variable('biases', [f_num], initializer=tf.constant_initializer(0.1))
conv = tf.nn.conv2d(input, filter_weight, strides=[1,k1,k2,1], padding='SAME')
bias = tf.nn.bias_add(conv, biases)
actived_conv = tf.nn.relu(bias)
pool = tf.nn.max_pool(actived_conv, ksize=[1, p1, p2, 1], strides=[1, sp1, sp2, 1], padding='SAME')
```

#### 池化层和卷积层的前向传播在tensorflow中实现的异同点：  
卷积层，`conv = tf.nn.conv2d(input, filter_weight, strides=[1,k1,k2,1], padding='SAME')`  
- input, 当前层的节点矩阵，该矩阵为四维矩阵，后面三个维度对应一个节点矩阵，第一维对应输入batch。如在输入层，input[0, :, :, :]表示第一张图片，input[1, :, :, :]表示第二张图片，，依此类推；
- filter_weight， 提供卷积层的权重，四维矩阵，前两个维度代表过滤器的尺寸，第三维表示当前层的深度(输入节点矩阵的深度)，第四维表示过滤器的深度（通常指的是过滤器的个数）；
- strides， 不同维度上的步长，四维矩阵，但第一维和第四维固定为1，因为卷积层的步长只对矩阵的长和宽有效；
- padding， 填充方式，‘SAME’或者‘VALID’，‘SAME’表示添加全0填充， ‘VALID’表示不添加

---------
池化层，`pool = tf.nn.max_pool(actived_conv, ksize=[1, p1, p2, 1], strides=[1, sp1, sp2, 1], padding='SAME')`  ，平均池化`tf.nn.avg_pool()`调用格式相同
- actived_conv， 需要传入当前层的节点矩阵，该矩阵是四维矩阵，格式与`tf.nn,conv2d()`的第一个参数格式相同；
- ksize， 过滤器的尺寸，长度为4的一位数组，第一个与第四个固定为1，表示池化层的过滤器是不可以跨不同输入样例或者节点矩阵深度的；
- strides， 步长参数，四维矩阵，第一维与第四维固定为1，表示池化层不能减少节点矩阵的深度或者输入样例的个数；
- padding， ‘SAME’或者‘VALID’；

### LeNet的实现
#### LeNet前向传播
```python
# -*- coding:utf-8 -*-
# LeNet5_infernece.py
import tensorflow as tf

# 设定神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节电个数
FC_SIZE = 512

# 定义传播过程
# 添加的新参数train，用于区分训练过程和测试过程。程序中用到dropout方法，可进一步提升模型可靠性并防止过拟合，只在训练时使用
def inference(input_tensor, train, regularizer):
# 通过使用不同的命名空间來隔离不同层的变量，可以让每一层中的变量命名只需要考虑在当前层的作用，而不需要担心重命名的问题。
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
		# 将第四层池化层的输出转化为第五层全连接层的输入格式。第五层全连接层需要的输入是向量，将第四层的输出矩阵拉成一个向量。pool2.get_shape().as_list()函数可得到第四层输出矩阵的维度。注意的是，因为每层神经网络的输入输出都为一个batch的矩阵，所以这里得到的维度也敖汉一个batch中的数据的个数。
        pool_shape = pool2.get_shape().as_list()
		# 计算将矩阵拉直成一个向量之后的长度，该长度等于矩阵长宽及深度的乘积，注意pool_shape[0]为一个batch中的数据的个数
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
		# 通过tf.reshape函数将第四层输出编程一个batch的向量
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

# dropout一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		# 只有全连接层的权重需要加入正则化
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
```
#### LeNet的training
```python
# LeNet5_train.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece
import os
import numpy as np

# 定义神经网络相关的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

# 定义训练过程
def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,									# 第一维表示一个batch中的样例的个数
            LeNet5_infernece.IMAGE_SIZE,	# 图片尺寸
            LeNet5_infernece.IMAGE_SIZE,
            LeNet5_infernece.NUM_CHANNELS],	# 图片深度
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = LeNet5_infernece.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.IMAGE_SIZE,
                LeNet5_infernece.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				
# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()
```

----------
[1]: ./attachments/0-classic_CNN.py
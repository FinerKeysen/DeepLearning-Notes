
### tensorflow学习之Inception结构 

Inception结构
> GoogleNet中的重要结构，将不同卷积层通过并联的方式结合在一起。

![1505560658467](https://i.loli.net/2018/05/02/5ae9465065a33.jpg)

Inception模块用不尺寸的过滤器处理输入矩阵，然后将处理的结果矩阵拼接成一个更深的矩阵，如上图。  
不同的组合方式的Inception模块见图2。

![1505561288288](https://i.loli.net/2018/05/02/5ae9467d79fec.jpg)

Inception-v3模型共46层，由11个Inception模块组成。图2中每个框框就是一个Inception结构，有96个卷积层。使用Tensorflow-Slim工具来简洁的实现一个卷积。
```python
# 直接使用Tensorflow原始的API实现卷积层
with tf.variable_scope(scope_name):
    weights = tf.get_variable("weights", ...)
    biases = tf.get_variable("biases", ...)
    conv = tf.nn.conv2d(...)
relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# 使用Tensorflow-Slim实现卷积层。Tensorflow-Slim一行代码就能实现一个卷积层的
# 前向传播计算。slim-conv2d函数有3个必填的参数，第一个是输入节点矩阵，第二个是
# 当前卷积层过滤器的深度，第三个参数是过滤器的尺寸。可选参数有过滤器滑动步长、
# 是否用‘0’填充、激活函数的选择及变量的命名空间
net = slim.conv2d(input, 32, [3, 3])
```
[完整的Inception-v3模型源码定义](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/inception_v3.py)  
示例实现图2中最后一个框框的结构，如下。
```python
...
# 此处省略其他框框的实现代码
# 假设输入图片经过之前的神经网络前向传播的结果保存在变量net中
net = 上一层的输出节点矩阵
# 为一个Inception模块声明一个统一的变量命名空间
with tf.variable_scope("Mixed_7c"):
    # 给Inception模块中的每一条路径声明一个命名空间
    with tf.variable_scope("Branch_0"):
        # 实现一个过滤器变长为1， 深度为320的卷积层
        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1×1')

    # 第二条路径
    with tf.variable_scope("Branch_1"):
        branch_1 = slim.conv2d(net, 384, [1, 1], scope="Conv2d_0a_1×1")
        # tf.concat函数将多个矩阵按指定的维度拼接，第一个参数指定了拼接的维度，这里的
        # 3代表矩阵是在深度这个维度上进行拼接的，如图1展示
        branch_1 = tf.concat(3, 
                            # 此处多层卷积层的输入对应前一层的输出
                            [slim.conv2d(branch_1, 384, [1, 3], scope="Conv2d_0b_1×3"),
                                slim.conv2d(branch_1, 384, [3, 1], scope="Conv2d_0c_3×1")]
                            )
    # 第三条路径
    with tf.variable_scope("Branch_2"):
        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1×1')
        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3×3')
        branch_2 = tf.concat(3, 
                            [slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1×3')
                                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3×1')]
                            )
    # 第四条路径
    with tf.variable_scope("Branch_3"):
        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool2d_0a_3×3')
        branch_3 = slim.max_pool2d(branch_3, 192, [1, 1], scope='MaxPool2d_0b_1×1')
    # 拼接
    net = tf.concat(3, 
                    [branch_0,
                        branch_1,
                        branch_2,
                        branch_3])

```

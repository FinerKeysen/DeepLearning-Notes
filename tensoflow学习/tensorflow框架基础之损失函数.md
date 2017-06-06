---
title: tensorflow框架基础之损失函数 
tags: tensorflow
grammar_cjkRuby: true
---


## tensorflow框架基础之损失函数
### 激活函数
> 激活函数去线性化

常见的激活函数有`sigmoid()`，`tanh()`，[ReLU](http://www.cnblogs.com/neopenx/p/4453161.html)，关于激活函数的一些作用参考[activation function](http://www.cnblogs.com/rgvb178/p/6055213.html)

在tensorflow中，实现上述函数的代码：
```python
tf.nn.sigmoid()
tf.nn.tanh()
tf.nn.relu()
```
### 传统损失函数
监督学习的两大类
- 分类问题：将不同的样本划分到已知的类别当中

多分类中，神经网络对每个样本产生一个n维数组作为输出结果，代表每个样本属于各类别的可能性。当然，如果是*one-hot coding*，那么输出应该只有所属类别的维度值是1，在其余类别的维度是0.  
  
*如何判断输出`y`与真实值`y_truth`有多接近？*

常用方法：`交叉熵cross_entropy`，它描述了两个概率分布之间的距离，当交叉熵越小说明二者之间越接近

交叉熵定义，概率$q$表示概率$p$的交叉熵为：$$H(p,q)=-\sum p(x)\log q(x) \qquad (1)$$
```python
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# y_:真实值，y:原始输出
# tf.reduce.mean函数求解平均数
# tf.clip_by_value函数将一个张量的数值限制在一个范围内，上面代码将 q 的值限制在(1e-10, 1.0)之间
```
但是，网络的输出不一定是概率分布，因此需要将网络前向传播的结果转换成概率分布。常用方法是`Softmax回归`.tensorflow中，`Softmax回归`只作为一层额外的处理层，进行概率分布的转换。转换公式：
$$softmax(y_i)=y'_i=\frac{e^{y_i}}{\sum^n_{j=1}e^{y_j}} \qquad (2)$$
式中原始的网络输出是$y_i$，$y'_i$是转换后的概率分布。注意的是公式(1)并不是对称的，也即是$(H(p,q))\neq H(q,p)$，公式(1)描述的是概率$q$表达概率$p$的困难程度。因此在神经网络的损失函数中，$q$代表预测值，$p$代表真实值。`tensorflow`将`cross_entropy`与`softmax`统一封装实现了`softmax`后的`cross_entropy`损失函数
```python
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)
```

- 回归问题：对具体数值的预测

如房价预测、销量预测等问题需要预测的是一个任意实数，其网络输出只有一个节点，也即是预测值。

常用的损失函数：均方误差(MSE, mean squared error)
$$MSE(y, y')=\frac{\sum^n_{i=1}(y_i-y'_i)^2}{n}  \qquad (3)$$
其中batch中的第$i$个输入的正确值记作$y_i$，其预测值记作$y'_i$
```python
mse = tf.reduce.mean(tf.squared(y_ - y))
# y_:正确值，y:预测值
```

### 自定义损失函数

自定义的损失函数通常更加符合所应用的场景，如在销量预测中，我们对预测值与正确值之间的大小关系作为损失调整的条件，那么得到类似下面的式子：
$$loss(y, y')=\sum^n_{i=1}f(y_i, y'_i) , \space\space\space\space  f(x,y)=\left\{\begin{matrix}
a(x-y) & x>y\\ 
b(x-y) & x\leq y
\end{matrix}\right.  \qquad (4)$$
在tensorflow中实现
```python
loss = tf.reduce.sum(tf.select(tf.greater(v1, v2), a*(v1-v2), b*(v1-v2)))
```
其中`tf.select()`函数
```python
# a>b时，m=True，否则m=False
m = tf.greater(a, b)

# m=True时，执行func1；m=False时，执行func2
tf.select(m, func1, func2)
```

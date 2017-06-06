---
title: tensorflow框架基础之Session 
tags: tensorflow
grammar_cjkRuby: true
---


> 绘画持有并管理tensorflow程序运行时的所有资源

调用会话的两种方式
**方式一：明确的调用会话的生成函数和关闭会话函数**
```python
# create a session
sess = tf.Session()

# use this session to run a result
sess.run(...)

# close this session, release memeory
sess.close()
```
调用这种方式时，要明确调用Session.close()，以释放资源。当程序异常退出时，关闭函数就不能被执行，从而导致资源泄露。

**方式二：上下文管理机制自动释放所有资源**
```python
# 创建会话，并通过上下文机制管理器管理该会话
with tf.Session() as sess:
    sess.run(...)
# 不需要再调用"Session.close()"
# 在退出with statement时，会话关闭和资源释放已自动完成
```

会话类似计算图机制，可以指定为默认
```python
sess = tf.Session()
with sess.as_default():
    # result为某个张量
    print(result.eval())

# 一下代码可完成相同的功能
sess = tf.Session()

print(sess.run(result)) # 或者
print(result.eval(session=sess))   
```

另外，在交互式环境下，通过设置默认会话的方式来获取张量的取值更加方便，调用函数`tf.InteractiveSession()`.省去将产生的会话注册为默认会话的过程。

以上，最常用的还是方式二，但这三种方式都可以通过`ConfigProto Protocol Buffer`来配置需要生成的会话，如并行线程数、GPU分配策略、运算超时时间 等参数，最常用的两个是`allow_soft_placement`和`log_device_placement`.
`ConfigProto`配置方法：
```python
config = tf.ConfigProto(allow_soft_placement=True, 
                        log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)                              
```	
`allow_soft_placement`，布尔型，一般设置为`True`，很好的支持多GPU或者不支持GPU时自动将运算放到CPU上。

`log_device_placement`，布尔型，为`True`时日志将会记录每个节点被安排在了哪个设备上以方便调试。在生产环境下，通常设置为`False`可以减少日志量。


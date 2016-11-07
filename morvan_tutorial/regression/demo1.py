# encoding=utf-8

import tensorflow as tf
import numpy as np

# create data
x_data = np.random.random(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
# 占位符,需要调整或者迭代的参数
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# 预测输出
y = weights * x_data + biases

# 设置损失函数
loss = tf.reduce_mean(tf.square(y - y_data))
# 设置最优化模型
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 调用最小化模型的minimize函数降低loss，返回训练模型
train = optimizer.minimize(loss)

# 初始化所有Variable
init = tf.initialize_all_variables()

### create tensorflow structure end ###

# 返回Session
sess = tf.Session()
sess.run(init)  # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(weights), sess.run(biases)

# encoding=utf-8


import tensorflow as tf

# 定义占位符 与sess.run的feed_dict绑定
ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)
# 定义操作

output = tf.mul(ph1, ph2)

with tf.Session() as sess:
    print sess.run(output, feed_dict={ph1: [2.], ph2: [7.]})

"""
tensorflow 常用操作符函数：
add,sub,mul,div,abs,matmul,assign,

"""

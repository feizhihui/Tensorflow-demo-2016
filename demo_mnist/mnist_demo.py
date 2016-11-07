# encoding=utf-8

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

# 输入变量
x = tf.placeholder(tf.float32, [None, 784])
# 参数变量
W = tf.Variable(tf.zeros([784, 10]))
# 偏置变量
b = tf.Variable(tf.zeros([10]))
# 隐藏层输出(784个隐藏单元)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义标签
y_ = tf.placeholder("float", [None, 10])

# 定义交叉熵损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 定义训练方法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化所有变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    # 随机取出一批数据 100*784,100*10
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 输出一组bool值
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 将bool值转化为浮点值求和再取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

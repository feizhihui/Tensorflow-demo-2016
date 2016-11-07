# encoding=utf-8


from __future__ import print_function

import tensorflow as tf
import input_data as load

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

img, label = load.read_and_decode('train.tfrecords')

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

init = tf.initialize_all_variables()
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=30, capacity=2000,
                                                min_after_dequeue=1000)
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # 启动队列
    threads = tf.train.start_queue_runners(sess=sess)
    step = 1
    # Keep training until reach max iterations

    while step * batch_size < training_iters:
        print('step:', step)

        print('shuffled batch')

        batch_x, batch_y = sess.run([img_batch, label_batch])
        print('generated batch')

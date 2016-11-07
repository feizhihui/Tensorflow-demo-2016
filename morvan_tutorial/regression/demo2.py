# encoding=utf-8

import tensorflow as tf

matrix1 = tf.constant([[3, 3]])  # 1 * 2
matrix2 = tf.constant([[2], [2]])  # 2 * 1

dotproduct = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print sess.run(dotproduct)

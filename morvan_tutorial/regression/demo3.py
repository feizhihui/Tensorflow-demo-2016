# encoding=utf-8

import tensorflow as tf

state = tf.Variable(0, name='counter')

# print type(state.name), state.name

init = tf.initialize_all_variables()  # must have if define Variable

one = tf.constant(1)

new_state = tf.add(state, one)

update = tf.assign(state, new_state)

with tf.Session() as sess:
    print sess.run(init)  # must have if define Variable

    print sess.run(state)

    for _ in range(3):
        sess.run(update)  # 执行一组操作
        print sess.run(state)  # 输出一个中间值

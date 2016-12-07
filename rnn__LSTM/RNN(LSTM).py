# encoding=utf-8


from PIL import Image
import tensorflow as tf
import numpy as np
import new_inputdata
import time

# configuration
#                        O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#                       ^ (O: output 28 vec from 28 vec input)
#                       |
#      +-+  +-+       +--+
#      |1|->|2|-> ... |28| time_step_size = 28
#      +-+  +-+       +--+
#       ^    ^    ...  ^
#       |    |         |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#      each input size = input_vec_size=lstm_size=28

# configuration variables
learning_rate = 0.005
training_iters = 2000
display_step = 20
channel_num = 3

input_vec_size = lstm_size = 128  # input_vec_size一行的128pixel,lstm_size表示128个隐藏层节点(等于每个节点处理的pixel数)
time_step_size = 128  # 128行 * 3通道

n_classes = 10

batch_size = 32

# 验证集长度
test_size = 64


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size*channel_num, input_vec_size)
    # permute time_step_size and batch_size=> (time_step_size,batch_size,input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_size])  # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(0, time_step_size * channel_num, XR)  # split them to time_step_size (128 arrays)
    #  (batch_size, input_vec_size) * time_step_size * channel_num arrays

    # Make lstm with lstm_size (each input vector size)
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=1.0)

    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    # lstm cell is divided into two pairs {cstate,mstate}
    outputs, _states = tf.nn.rnn(lstm, X_split, dtype=tf.float32)  # _states={cstate,mstate}

    # Linear activation
    # Get the last output outputs[-1]==_states[1]
    return tf.matmul(outputs[-1], W) + B, lstm.state_size  # State size to initialize the stat


cifar = new_inputdata.read_data_sets(one_hot=True)
teX, teY = cifar.test.images, cifar.test.labels
teX = teX.reshape(-1, time_step_size * channel_num, input_vec_size)

X = tf.placeholder("float", [None, time_step_size * channel_num, input_vec_size])
Y = tf.placeholder("float", [None, n_classes])

# get lstm_size and output 10 labels
W = init_weights([lstm_size, n_classes])
B = init_weights([n_classes])

py_x, state_size = model(X, W, B, lstm_size)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

start_time = time.time()
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    count = 0
    for i in range(training_iters):
        # print '第%i次迭代；' % (i + 1)
        batch_x, batch_y = cifar.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, time_step_size * channel_num, input_vec_size])
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        test_indices = np.arange(len(teX))  # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        if i % display_step == 0:
            print(i, 'Accuracy:', np.mean(np.argmax(teY[test_indices], axis=1) ==
                                          sess.run(predict_op, feed_dict={X: teX[test_indices]})))
    print('Final Accuracy:', np.mean(np.argmax(teY, axis=1) ==
                                     sess.run(predict_op, feed_dict={X: teX})))

print 'runtime:', time.time() - start_time, 's'

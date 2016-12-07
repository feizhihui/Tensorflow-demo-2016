# encoding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
import urllib

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# http://mattmahoney.net/dc/text8.zip

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# 当前路径下保存text8.zip
filename = maybe_download('text8.zip', 31344016)


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# 读取文件所有单词 17005207
words = read_data(filename)
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000


def build_dataset(words):
    count = [['UNK', -1]]
    # 取出单词字典中value最大的49999个单词
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            # rare words
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # 返回元组型列表（序号对应单词）
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


# 文章用字典序号表示[],单词对应数量{}，单词对应序号{}，序号对应单词{}
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
# 从2*skip_windows上下文单词中选出num_skips个上下文单词保存到(batch,label)中{w1:w0,w1:w2,w2:w1,w2:w3}
def generate_batch(batch_size, num_skips, skip_window):
    # 使用全局的data_index
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # 1行8列
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # 8行1列
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # span=2*1+1=3
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # 限制长度的队列 maxlen=3
    buffer = collections.deque(maxlen=span)
    # 从data中循环选取数据填满buffer
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # 8//2=4次外循环
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        # 2次内循环
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)  # 下标0到2,buffer中随机取一个
            targets_to_avoid.append(target)
            # batch总是取buffer的中间值
            batch[i * num_skips + j] = buffer[skip_window]
            # 中间值的上下文（不包括自身单词）
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


# ===============================================================================================
# ===============================================================================================
# this is a test!

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1  # How many words to consider left and right.
num_skips = 2  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
# 从0~99随机选出16个数组成valid_examples
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    pred_num = tf.placeholder(tf.int32, shape=[1])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        # 定义embeddings张量为 5000 words * 128 features -1到1的均匀分布
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 对批数据中的单词建立嵌套向量，TensorFlow提供了方便的工具函数 -->> mode partition
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        # 对语料库的每个单词定义权重和偏差
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    # 计算 NCE 损失函数, 每次使用负标签的样本.
    # weights[N,K],biases[N],embed[batch_size,K],labels[batch_size]
    # num_sampled:number of negative samle
    # vacabulary_size： sample from [0~vacabulary_size-1] k越小，抽样概率越高
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                       num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    # 横向平方取和，维度不变
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # 正则化
    normalized_embeddings = embeddings / norm
    valid_embed = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    #
    similarity = tf.matmul(
        valid_embed, normalized_embeddings, transpose_b=True)  # 将第二个矩阵先转置 [16,128] * [128,5000] =[16,5000]

    # ==============================预测valid_data的上下文 ===============================
    # pred_num is a list
    pred_embed = tf.nn.embedding_lookup(
        embeddings, pred_num)
    pred_y = tf.nn.softmax(tf.matmul(pred_embed, nce_weights, transpose_b=True) + nce_biases)
    pred_label = tf.argmax(pred_y, 1)
    # ==============================预测valid_data的上下文 ===============================



    # Add variable initializer.
    init = tf.initialize_all_variables()

# Step 5: Begin training.
# num_steps = 100001
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 迭代2000次输出一次损失值
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        # 迭代10000次
        if step % 10000 == 0:
            print('================================================================')
            sim = similarity.eval()  # 16*5000 valid_size=16
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                # 取出与余弦相似性最大的前topk个(排除itself)
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

    # ==============================预测valid_data的上下文 ===============================
    print('======================================================')
    word_input = raw_input('请输入目标单词：')
    while word_input != 'STOP':
        ylabel, ypred = session.run([pred_label, pred_y], feed_dict={pred_num: [dictionary[word_input]]})
        # print(ylabel)
        # print(ypred)
        print('它的上下文单词可能是:', reverse_dictionary[ylabel[0]])
        print('======================================================')
        word_input = raw_input('请输入目标单词：')


        # Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


try:
    # t-distributed stochastic neighbor embedding
    # http://mtpgm.com/2015/08/17/t-sne/
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # n_comments:目标维数
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    # 选取embeddings的前500行数据进行拟合降维
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    np.savetxt("embeddings.csv", final_embeddings[:plot_only, :], delimiter=',');
    # 前500行单词
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    # 画图
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

# encoding=utf-8


import os

from PIL import Image
import tensorflow as tf
import numpy as np

cwd = os.getcwd()
train_classes = ['/examples/0', '/examples/1', '/examples/2', '/examples/3']
test_classes = ['/examples/test0', '/examples/test1', '/examples/test2', '/examples/test3']


def create_record(train_classes, record_name):
    print cwd
    '''
    此处我加载的数据目录如下：
    0 -- img1.jpg
         img2.jpg
         img3.jpg
         ...
    1 -- img1.jpg
         img2.jpg
         ...
    2 -- ...
    ...
    '''
    writer = tf.python_io.TFRecordWriter(record_name)
    for index, name in enumerate(train_classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            print img
            img = img.resize((100, 100))
            img.save('123.jpg')
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            # tf.train.FeatureLists
        writer.write(example.SerializeToString())

    writer.close()


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [30000])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    # label = tf.reshape(label, 4)

    return img, label


# batch_size  30
def read_record(record_name):
    img, label = read_and_decode(record_name)

    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=20, capacity=2000,
                                                    min_after_dequeue=1000)
    # 初始化所有的op
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        # 启动队列
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(3):
            val, l = sess.run([img_batch, label_batch])
            # l = to_categorical(l, 12)
            print(val.shape, l)
            print len(val), type(val)


def read_data_sets():
    an = Animals()
    return


class Animals(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        if not os.path.exists('animals.train'):
            create_record(train_classes, 'animals.train')
        if not os.path.exists('animals.test'):
            create_record(test_classes, 'animals.test')

    def read_record(self, batch_size):
        self.batch_size = batch_size
        self.img, self.label = read_and_decode('animals.train')
        self.timg, self.tlabel = read_and_decode('animals.test')
        self.img_batch, self.label_batch = tf.train.shuffle_batch([self.img, self.label],
                                                                  batch_size=self.batch_size, capacity=2000,
                                                                  min_after_dequeue=1000)
        self.timg_batch, self.tlabel_batch = tf.train.shuffle_batch([self.timg, self.tlabel],
                                                                    batch_size=self.batch_size, capacity=2000,
                                                                    min_after_dequeue=1000)

    def train_next_batch(self, sess):

        batch_x, batch_y = sess.run([self.img_batch, self.label_batch])
        yarr = np.zeros(shape=[self.batch_size, self.n_classes])
        # 生成one-hot
        for k in range(self.batch_size):
            yarr[k][batch_y[k]] = 1
        batch_y = yarr
        return batch_x, batch_y

    def test_next_batch(self, sess):

        batch_x, batch_y = sess.run([self.timg_batch, self.tlabel_batch])
        yarr = np.zeros(shape=[self.batch_size, self.n_classes])
        # 生成one-hot
        for k in range(self.batch_size):
            yarr[k][batch_y[k]] = 1
        batch_y = yarr
        return batch_x, batch_y


if __name__ == '__main__':
    create_record(train_classes, 'animals.train')
    create_record(test_classes, 'animals.test')
    read_record('animals.train')

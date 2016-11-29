# encoding=utf-8
import os
import numpy as np
from PIL import Image
import tensorflow as tf

cwd = os.getcwd()
train_classes = ['/examples/0', '/examples/1', '/examples/2', '/examples/3', '/examples/4', '/examples/5',
                 '/examples/6', '/examples/7', '/examples/8', '/examples/9']
test_classes = ['/examples/T0', '/examples/T1', '/examples/T2', '/examples/T3', '/examples/T4', '/examples/T5',
                '/examples/T6', '/examples/T7', '/examples/T8', '/examples/T9']


class DataSet(object):
    def __init__(self, images, labels, one_hot=False,
                 dtype=tf.float32):
        """Construct a DataSet.

        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   labels.shape))
        # images数据量
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 3
        # 格式转换[num examples, rows*columns]
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
        if dtype == tf.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    # ==================================================================================
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""

        # 定义起始窗口位置
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # Shuffle the data 打乱整个数据集的顺序
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        # 窗口末尾位置超过数据量大小
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


# ==========================================================================

def create_record(data_classes, one_hot=False, num_classes=10):
    img_raw = str()
    labels = np.array([], dtype=np.int32)
    total = 0
    for index, name in enumerate(data_classes):
        class_path = cwd + name + "/"
        subdir = os.listdir(class_path)
        total += len(subdir)
        for img_name in subdir:
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((128, 128))
            img.save('123.jpg')
            img_raw = img_raw + img.tobytes()  # 将图片转化为原生bytes
            labels = np.hstack((labels, index))
    # Interpret a buffer as a 1-dimensional array.
    data = np.frombuffer(img_raw, dtype=np.uint8)
    # 转化成[,28,28,3]格式
    data = data.reshape(total, 128, 128, 3)
    if one_hot:
        labels = dense_to_one_hot(labels, num_classes)
    return data, labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    # flat展开浅拷贝，ravel展开深拷贝
    pos_index = (index_offset + labels_dense.ravel())
    labels_one_hot.flat[pos_index] = 1
    return labels_one_hot


def read_data_sets(one_hot=False, dtype=tf.float32):
    class DataSets(object):
        pass

    data_sets = DataSets()

    train_images, train_labels = create_record(train_classes, one_hot=one_hot)
    test_images, test_labels = create_record(test_classes, one_hot=one_hot)

    data_sets.train = DataSet(train_images, train_labels, dtype=dtype)
    data_sets.test = DataSet(test_images, test_labels, dtype=dtype)

    return data_sets


if __name__ == '__main__':
    images, labels = create_record(train_classes, one_hot=True)
    cifar = DataSet(images, labels, one_hot=True)
    for k in range(10):
        images_batch, labels_batch = cifar.next_batch(5)
        print images_batch
        print labels_batch

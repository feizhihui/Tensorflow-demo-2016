# encoding=utf-8


import os
from PIL import Image
import tensorflow as tf

cwd = os.getcwd()
classes = ['/examples/0', '/examples/1', '/examples/2', '/examples/3']


def create_record():
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
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + name + "/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            print img
            img = img.resize((224, 224))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    create_record()

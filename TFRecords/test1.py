# encoding=utf-8

import tensorflow as tf
import Image
import numpy as np

filename_queue = tf.train.string_input_producer(['png25.png'])  # list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value)  # use png or jpg decoder based on your files.

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)

for i in range(1):  # length of your filename list
    image = my_img.eval()  # here is your image Tensor :)

print(image.shape)
Image.show(Image.fromarray(np.asarray(image)))

coord.request_stop()
# coord.join(threads)

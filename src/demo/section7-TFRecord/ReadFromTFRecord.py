# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

# 创建一个reader来读取TFReacord文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["E:/tensorflow/section7/toTFRecord/output.tfrecords"])

# 从文件中读出一个样例，也可以使用read_up_to函数一次性读出多个
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例，如果需要解析多个样例，可以使用parse_example
features = tf.parse_single_example(
    serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
    })
# tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# 启动多线程处理输入数据，
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(image)
    print(label)
    print(pixel)

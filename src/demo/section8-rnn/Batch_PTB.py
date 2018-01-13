# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf
import reader
DATA_PATH = "../data/PTB/data/"
train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(DATA_PATH)
# #将训练数据组织成大小为4，截断长度为5的数据组
# result = reader.ptb_producer(train_data,4,5)
# for (x,y) in result:
#     print("X:",x)
#     print("Y:",y)

result = reader.ptb_producer(train_data, 4, 5)

# 通过队列依次读取batch。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print("X%d: "%i, x)
        print("Y%d: "%i, y)
    coord.request_stop()
    coord.join(threads)

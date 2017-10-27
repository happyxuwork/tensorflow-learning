# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

rq = tf.RandomShuffleQueue(capacity=5, min_after_dequeue=2, dtypes=tf.int32)
init = rq.enqueue_many(([1, 2, 3, 4, 5],))
x = rq.dequeue()
y = x + 1
rq_in = rq.enqueue([y])
with tf.Session() as sess:
    init.run()
    for i in range(5):
        val, _ = sess.run([x, rq_in])
        print(val)

# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

q = tf.FIFOQueue(capacity=5, dtypes=tf.int32)
# the initial process should be sepcifal call
init = q.enqueue_many(([0, 10, 6, 5, 9],))
x = q.dequeue()
y = x + 1
q_in = q.enqueue(vals=[y])
with tf.Session() as sess:
    sess.run(init)
    # init.run()
    for _ in range(10):
        v, _ = sess.run([x, q_in])
        print(v)

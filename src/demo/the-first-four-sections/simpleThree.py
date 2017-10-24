# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

'''
test the usage of tf.select(),and tf.greater()
'''
# import tensorflow as tf
# v1 = tf.constant([1.0,2.0,3.0,4.0])
# v2 = tf.constant([4.0,3.0,1.0,2.0])
# sess  = tf.InteractiveSession()
# print(tf.greater(v1,v2).eval())
# print(tf.where(tf.greater(v1,v2),v1,v2).eval())

# import numpy as np
# print(np.random.rand())

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8
# define paramter
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

# dtype, shape=None, name=None
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# the value of real y
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
# the value of predict y
y = tf.matmul(x, w1)

# generator the set of input data
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
# generator the set of real y
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]

# the constant value in the loss function
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # inital the varibles
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % data_size
        end = min(batch_size + start, data_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        print(sess.run(w1))

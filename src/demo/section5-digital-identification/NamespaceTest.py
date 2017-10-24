# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

with tf.variable_scope("foo"):
    v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
    v1 = tf.get_variable("v1", shape=[1], initializer=tf.constant_initializer(2.0))

with tf.variable_scope("foo", reuse=True):
    v3 = tf.get_variable("v1", [1])
    print(v3 == v1)
    print(v3)

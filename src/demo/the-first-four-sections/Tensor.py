# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

import tensorflow as tf

a = tf.constant([1.0, 2], name='a')
b = tf.constant([3, 9], name='b', dtype=tf.float32)
result = tf.add(a, b, name="add")
print(result)

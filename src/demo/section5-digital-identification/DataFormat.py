# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")

result = v1 + v2
saver = tf.train.Saver()
saver.export_meta_graph("E:/tensorflow/savemodel3-json/model.ckpt.meda.json", as_text=True)

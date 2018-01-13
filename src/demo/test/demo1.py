# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf
import glob
import os

# list1 = [2]
# list2 = [3, 4, 5]
# print(list1 + list2)

# input_ = tf.get_variable("Matrix", [4, 2], tf.float32, tf.random_normal_initializer(stddev=0.2))
# matrix = tf.get_variable("Matrix2", [2, 3], tf.float32, tf.random_normal_initializer(stddev=0.2))
# bias = tf.get_variable("bias", [3], initializer=tf.constant_initializer(1))
# H0 = tf.matmul(input_, matrix) + bias
# with tf.Session() as sess:
#     print(sess.run(input_))
#     print(sess.run(matrix))
#     print(sess.run(bias))
#     print(sess.run(H0))
# kk = []
# str = os.path.join("../data","picture", "*.jpg")
# strabs = os.path.abspath(str)
# kk = glob.glob(str)
# print(kk[0])

from PIL import Image

#im = Image.open('E:/programdate/python-all-in-for-happyxuwork/tensorflow-learning/src/demo/data/picture/3d6a9dc45b9ecad5.jpg')
# im = Image.open('E:/programdate/python-all-in-for-happyxuwork/tensorflow-learning/src/demo/data/picture/timg.jpg')
# pix = im.load()
# width = im.size[0]
# height = im.size[1]
# for x in range(width):
#     for y in range(height):
#         r, g, b = pix[x, y]











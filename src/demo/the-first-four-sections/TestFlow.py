# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import os

os.environ['IF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# a = tf.constant([1.0,2.0],name='a')
# b = tf.constant([2.0,3.0],name='b')
# result = a + b
# # sess = tf.Session()
# # print(sess.run(a))
# print(a.graph is tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    # 指定g1为默认计算图，定义变量"v",初始化为0.
    v = tf.get_variable("v", shape=[2, 2], initializer=tf.zeros_initializer())

g2 = tf.Graph()
with g2.as_default():
    # 指定g2为默认计算图，定义变量"v",初始化为1.
    v = tf.get_variable("v", shape=[2, 3], initializer=tf.ones_initializer())

# 在计算图g1中读取变量"v"的取值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()  # 初始化所有变量
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))  # 输出[0.]

# 在计算图g2中读取变量"v"的取值
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))  # 输出[1.]


# import numpy as np
# # 用 NumPy 随机生成 100 个数据
# x_data = np.float32(np.random.rand(2, 100))
# y_data = np.dot([0.100, 0.200], x_data) + 0.300
#
# # 构造一个线性模型
# b = tf.Variable(tf.zeros([1]))
# W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# y = tf.matmul(W, x_data) + b
#
# # 最小化方差
# loss = tf.reduce_mean(tf.square(y - y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 启动图 (graph)
# sess = tf.Session()
# sess.run(init)
#
# # 拟合平面
# for step in range(0, 201):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(W), sess.run(b))

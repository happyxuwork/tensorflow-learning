# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

import tensorflow as tf

# 四维变量代表：前两个代表过滤器的大小，第三个代表当前层的深度，第四个参数代表过滤器的深度
filter_weight = tf.get_variable("weights", [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))
# 深度
biases = tf.get_variable("biases", [16], initializer=tf.constant_initializer(0.1))
# 第一个参数是输入，四维矩阵，后面三维对应一个节点矩阵：例如input[0,:,:,:]表示第一张图片，input[1,:,:,:]表示第二张图片
# 第二个参数是卷积层的权重
# 第三个参数是长度为4的数组，第一维和最后一维必须维1，因为卷积层的步长只对矩阵的长和宽有效
# 第四个参数是填充，SAME和VALID，其中SAME表示填充全0，VALID表示不添加。
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')
biases = tf.nn.bias_add(conv, biases)
actived_conv = tf.nn.relu(biases)

pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

# 神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层的尺寸和大小
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积层的尺寸和大小
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层节点个数
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    # 第一层卷积层
    # 输入为28*28*1
    # filter大小为5*5，个数为32个，移动步长为1，填充0---填充的意义在于，，，，，
    # 输出为28*28*32
    with tf.variable_scope("layer1-conv1"):
        conv1_weights = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("biases", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层--最大池化层
    # 输入为28*28*32
    # filter大小为2*2，移动步长为2，填充0
    # 输出为14*14*32
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第三层卷积层
    # 输入为14*14*32
    # filter大小为5*5，个数为64，移动步长为1，填充0
    # 输出为14*14*64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层为池化层--最大池化层
    # 输入大小为14*14*64
    # filter大小为2*2，移动步长为2，填充0
    # 输出大小为7*7*64
    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 全连接层的输入格式为向量，所以需要将第四层的输出拉直为一个向量
    pool_shape = pool2.get_shape().as_list()
    # pool_shape[0]为一个batch中数据的个数
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])  # ?????????????????????

    # 第五层输入长度7*7*64=3136,输出为512
    # 这里引入dropout的概念，在训练时，dropout会随机将部分节点的输出改为0，可以避免过拟合问题，
    # dropout一般只在全连接层而不是卷积层或池化层使用
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接的权重需要正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("biases", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层输入为512，输出为10，通过softmax得到最后的分类结果
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = tf.get_variable("weigths", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("biases", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit

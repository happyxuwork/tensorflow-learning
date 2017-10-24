# -*- coding: UTF-8 -*-
'''
@author: xuqiang
定义神经网络的前向传播算法，
无论是训练还是测试，直接调用即可
'''
import tensorflow as tf

# 定义网络的结构
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if (regularizer != None):
        # if the regualrizer is not None ,add it the the collection of losses
        tf.add_to_collection("losses", regularizer(weights))
    return weights


# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        # 如果在同一个程序中多次调用，第一次调用之后，需要将reuse参数设置为True
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope("layer2"):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

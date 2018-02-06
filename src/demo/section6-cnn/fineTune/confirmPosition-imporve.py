# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import os
import numpy as np
import tensorflow as tf
import glob
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# nameAndPositon = {'orange':'0','apple':'1','one':'2','zebra':'3','chinese':'5','oil':'6','many':'7'}
# print(nameAndPositon.keys())

'''
提取文件名中，，，
'''
# name = "oiltochinese-15-15-test"
# values = name.split("to")
# first = values[0]
# second = values[1].split("-")[0]
# print(first)
# print(second)
# print(nameAndPositon[first])

def getOneImgClassifyProb(sess,imageToTensorPath):
    # chkName = tf.train.latest_checkpoint('E:/tensorflow/selfmodelclassify/')
    # saver = tf.train.import_meta_graph(chkName + '.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('E:/tensorflow/selfmodelclassify/'))
    with open(imageToTensorPath,'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.strip().split(',')]
    bottleneck_values_reshape = np.reshape(bottleneck_values, [-1, 2048])
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
    w1 = graph.get_tensor_by_name("final_training_ops/op_to_prob:0")
    feed_dict = {input: bottleneck_values_reshape}
    prob = sess.run(w1, feed_dict)
    return prob

def main(input_data_tensor_valuation_path):
        with tf.Session() as sess:
            chkName = tf.train.latest_checkpoint('E:/tensorflow/getPosition/')
            saver = tf.train.import_meta_graph(chkName + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('E:/tensorflow/getPosition/'))
            prob = getOneImgClassifyProb(sess,input_data_tensor_valuation_path+'tulips/112951022_4892b1348b_n.jpg.txt')
            print(prob)
if __name__ == "__main__":
    # input_data_tensor_valuation_path = 'F:/converImgtoTensor/tensor-valuation/'
    input_data_tensor_valuation_path = '../data/bottleneck/mix/'
    main(input_data_tensor_valuation_path)























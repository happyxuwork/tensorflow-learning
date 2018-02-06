# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

import yes
import fineTune

image_lists = fineTune.create_image_lists(10,10)
keyList = list(image_lists.keys())
print(keyList)
# print(keyList[0])
# print(keyList[1])

# print(image_lists.keys()[0])
# print(image_lists.keys()[1])

# n_classes = len(image_lists.keys())
# #Prepare to feed input, i.e. feed_dict and placeholders
# w1 = tf.placeholder("float", name="w1")
# w2 = tf.placeholder("float", name="w2")
# b1= tf.Variable(2.0,name="bias")
# feed_dict ={w1:4,w2:8}
# #Define a test operation that we will restore
# w3 = tf.add(w1,w2,name="op_to_add")
# w4 = tf.multiply(w3,b1,name="op_to_restore")
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# #Create a saver object which will save all the variables
# saver = tf.train.Saver()
# #Run the operation by feeding input
# print(sess.run(w4,feed_dict))
# #Prints 24 which is sum of (w1+w2)*b1
# #Now, save the graph
# saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=1000)



sess=tf.Session()
#First let's load meta graph and restore weights
chkName = tf.train.latest_checkpoint('E:/tensorflow/selfmodelclassify/')
saver = tf.train.import_meta_graph(chkName+'.meta')
saver.restore(sess,tf.train.latest_checkpoint('E:/tensorflow/selfmodelclassify/'))


# saver = tf.train.import_meta_graph('E:/tensorflow/savemodel2/yes.pb-1000.meta')
# saver.restore(sess,tf.train.latest_checkpoint('E:/tensorflow/savemodel2/'))
# Access saved Variables directly
# print(sess.run('bias:0'))
# print(sess.run('final_training_ops/weight:0'))
# print(sess.run('final_training_ops/biases:0'))
with open('F:/converImgtoTensor/tensor-valuation/appletoorange-5-5-test/fakeA_0_18.jpg.txt', 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
# bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
bottleneck_values = [float(x) for x in bottleneck_string.strip().split(',')]
aa = bottleneck_values
aa = np.reshape(aa,[-1,2048])

# print(sess.run('DecodeJpeg/contents:0'))
# print(sess.run('pool_3/_reshape:0'))
# This will print 2, which is the value of bias that we saved
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data



graph = tf.get_default_graph()
# all_vars = tf.trainable_variables()
# for v in all_vars:
#     print(v.name)

input = graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
w1 = graph.get_tensor_by_name("final_training_ops/op_to_prob:0")
# print(sess.run('DecodeJpeg/contents:0'))
# print(sess.run('pool_3/_reshape:0'))
# print(aa.shape)
feed_dict ={input:aa}
# #Now, access the op that you want to run.
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
# op_to_add = graph.get_tensor_by_name("op_to_add:0")
# print(sess.run(op_to_restore,feed_dict))
prob = sess.run(w1,feed_dict)
print(prob[0])
# print()
# print(prob[0][1])
# print(keyList[4]+" "+str(prob[0][5]))












'''
保存模型
'''
# # 构造网络
# a = tf.placeholder(dtype=tf.float32, name='a')
# b = tf.placeholder(dtype=tf.float32, name='b')
# # 一定要给输出tensor取一个名字！！
# output = tf.add(a, b, name='out')
#
# # 转换Variable为constant，并将网络写入到文件
#
# init_op = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init_op)
#     # grap_def = tf.get_default_graph().as_default()
#     grap_def = sess.graph_def
#     # 这里需要填入输出tensor的名字
#     output_graph_def = convert_variables_to_constants(sess,grap_def,['out'])
#     with tf.gfile.GFile(os.path.join(MODEL_SAVE_PATH, MODEL_NAME),"wb") as f:
#         f.write(output_graph_def.SerializeToString())
#

# '''
# 载入模型
# '''
# from tensorflow.python.platform import gfile
# with tf.Session() as sess:
#     file_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
#     with gfile.FastGFile(file_path,'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#         inputa = tf.import_graph_def(graph_def,return_elements=["import_2/a"])
#         inputb = tf.import_graph_def(graph_def,return_elements=["import_2/b"])
#         result = tf.import_graph_def(graph_def,return_elements=["out:0"])
#         # print(sess.run(result,feed_dict={inputa:float(4),inputb:float(5)}))
#         print(sess.run(result))
#














# ### 定义模型
# in_dim = 3
# h1_dim = 10
# out_dim = 5
#
# input_x = tf.placeholder(tf.float32, shape=(None, in_dim), name='input_x')
# input_y = tf.placeholder(tf.float32, shape=(None, out_dim), name='input_y')
#
# w1 = tf.Variable(tf.truncated_normal([in_dim, h1_dim], stddev=0.1), name='w1')
# b1 = tf.Variable(tf.zeros([h1_dim]), name='b1')
# w2 = tf.Variable(tf.zeros([h1_dim, out_dim]), name='w2')
# b2 = tf.Variable(tf.zeros([out_dim]), name='b2')
# keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# hidden1 = tf.nn.relu(tf.matmul(input_x, w1) + b1)
# hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
# ### 定义预测目标
# y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)
# # 创建saver
# saver = tf.train.Saver()
# # 假如需要保存y，以便在预测时使用
# tf.add_to_collection('pred_network', y)
# sess = tf.Session()
# train_op = tf.initialize_all_variables
# for step in range(1000):
#     sess.run(train_op)
#     if step % 100 == 0:
#         # 保存checkpoint, 同时也默认导出一个meta_graph
#         # graph名为'my-model-{global_step}.meta'.
#         saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=step)


# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform  import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '../data/model'
MODEL_FILE ='tensorflow_inception_graph.pb'
CHECK_DIR = '../data/bottleneck'
INPUT_DATA = '../data/dataforclassify/horsetozebra/'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 400
BATCH = 100
# 模型保存路径
MODEL_SAVE_PATH = "E:/tensorflow/classify"
MODEL_NAME = "model.ckpt"

meta_path = 'E:/tensorflow/classify/model.ckpt-399.meta'
model_path = 'E:/tensorflow/classify/model.ckpt-399'


















# saver = tf.train.import_meta_graph(meta_path)  # 导入图
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # with tf.Session(config=config) as sess:
# with tf.Session() as sess:
#     saver.restore(sess, model_path)  # 导入变量值
#     graph = tf.get_default_graph()
#     # prob_op = graph.get_operation_by_name('prob')  # 这个只是获取了operation， 至于有什么用还不知道
#     prediction = graph.get_tensor_by_name('pool_3/_reshape:0')  # 获取之前prob那个操作的输出，即prediction
#     print(prediction)
# #     print(ress.run(prediciton, feed_dict={...}))  # 要想获取这个值，需要输入之前的placeholder （这里我编辑文章的时候是在with里面的，不知道为什么查看的时候就在外面了...）
# #     print(sess.run(graph.get_tensor_by_name('final_training_ops/weights:0')))  # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重
#     print(sess.run(tf.trainable_variables()))  # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重



# with tf.Session() as sess:
#     # saver = tf.train.Saver()
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     chkpt_fname = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
#     # this is vital important the have the if
#     if chkpt_fname is not None:
#         print("yes")
#         saver = tf.train.Saver()
#         saver.restore(sess, chkpt_fname)
    # saver = tf.train.Saver()
    # with tf.Session() as sess:
    #     ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         # 加载模型
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #         # 通过文件名得到模型保存时迭代的轮数
    #         global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    #     else:
    #         print("NO checkpoint file found!")


        # saver.restore(sess, model_path)  # 导入变量值
        # graph = tf.get_default_graph()
        # prob_op = graph.get_operation_by_name('prob')  # 这个只是获取了operation， 至于有什么用还不知道
        # prediction = graph.get_tensor_by_name('prob:0')  # 获取之前prob那个操作的输出，即prediction
        # print(sess.run(prediction, feed_dict={...}))  # 要想获取这个值，需要输入之前的placeholder （这里我编辑文章的时候是在with里面的，不知道为什么查看的时候就在外面了...）
        # print(sess.run(graph.get_tensor_by_name('final_training_ops/weights:0')))  # 这个就不需要feed了，因为这是之前train operation优化的变量，即模型的权重









    # with gfile.FastGFile(os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 'r') as f:
    #     graph_def = tf.GraphDef()
    #     graph_def.ParseFromString(f.read())
    #
    # # 加载读取的Inception-v3模型，返回数据输入对应的张量以及计算瓶颈层结果对应的张量
    # bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])
    # print(bottleneck_tensor.shape())
    # print(jpeg_data_tensor.shape())

    # with tf.Session() as sess:
    #   new_saver = tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    #   new_saver.restore(sess, 'my-save-dir/my-model-10000')
    #   # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    #   y = tf.get_collection('pred_network')[0]
    #
    #   graph = tf.get_default_graph()
    #
    #   # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
    #   input_x = graph.get_operation_by_name('input_x').outputs[0]
    #   keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    #
    #   # 使用y进行预测
    #   sess.run(y, feed_dict={input_x:....,  keep_prob:1.0})

    # meta_path = './model/checkpoint/model.ckpt.meta'
    # model_path = './model/checkpoint/model.ckpt'
    # saver = tf.train.import_meta_graph(meta_path)  # 导入图

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # with tf.Session(config=config) as sess:
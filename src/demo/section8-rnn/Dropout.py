# -*- coding: UTF-8 -*-
'''
@author: xuqiang
dropout的使用
'''
import tensorflow as tf

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
#使用DropoutWrapper类来实现dropout功能。该类通过两个参数控制dropout的概率，
#一个参数为input_keep_prob,用它来控制输入dropout概率，
#另一个为output_keep_prob，可以用它来控制输出的dropout的概率
dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,output_keep_prob=0.5)

#在使用了dropout的基础上定义多层
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm]*number_of_layers)
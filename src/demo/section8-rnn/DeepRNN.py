# -*- coding: UTF-8 -*-
'''
@author: xuqiang
深层神经网络的模型结构
'''
import tensorflow as tf

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
#通过MultiRNNCell类实现深层循环神经网络
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm]*number_of_layers)
#通过zero_state函数进行初始状态的设置
state = stacked_lstm.zero_state(batch_size=batch_size,tf.float32)

for i in range(len(num_step)):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
        stacked_lstm_output,state = stacked_lstm(current_input,state)
        final_output = fully_connected(stacked_lstm_output)
        loss += calc_loss(final_output,expected_output)

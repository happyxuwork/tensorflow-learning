# -*- coding: UTF-8 -*-
'''
@author: xuqiang
LSTM结构的模型
'''
import tensorflow as tf
#自动创建LSTM结构
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

#将LSTM中的状态初始化全0数组
state = lstm.zero_state(batch_size,tf.float32)

loss = 0.0
#num_step--最大序列长度
for i in range(num_step):
    if i > 0:
        tf.get_variable_scope().reuse_variables()
    #当前输入（current_input）和前一时刻状态（state）传入到定义的LSTM中后，可以输出lstm_output和更新后的状态
    lstm_output,state = lstm(current_input,state)

    #将当前的输出传入一个全连接层得到最后的输出
    final_output = fully_connected(lstm_output)
    #损失函数的计算
    loss += calc_loss(final_output,expected_output)
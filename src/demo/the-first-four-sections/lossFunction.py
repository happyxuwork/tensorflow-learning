# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

'''
hwo to use the L1 and L2 regularizer mothod
'''
# import tensorflow as tf
# weights = tf.constant([[1.0,-2.0],[-3.0,4.0]])
# with tf.Session() as sess:
#     print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
#     print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))

'''
the following coding show how to use the collection when the construct of net is complex
there is a five laysers net and L2 regularizer function used in this demo
what we can get from the folloing code is that
use the collection when the net construct is very complex will make the code more readability
'''
import tensorflow as tf


# get the weight of a layer and then add loss of the L2 to a collection named 'losses'
def get_weight(shape, rate):
    # generator a vairable
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # use add_to_collection function add the loss generator by L2 function to the collection
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(rate)(var)
                         )
    return var


x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

batch_size = 8
# define the number of each layer
layer_dimension = [2, 10, 10, 10, 10, 1]

n_layers = len(layer_dimension)

# the current layer is the input layer
cur_layer = x
# number of the current layer
in_dimention = layer_dimension[0]

# generator a 5 laysers neural network by a loop
for i in range(1, n_layers):
    # out_dimension representation the number of next layer
    out_dimention = layer_dimension[i]
    weigth = get_weight([in_dimention, out_dimention], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimention]))
    # use the ReLu active function
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weigth) + bias)
    # updating the current layer number to the next layser when get int the next layer
    in_dimention = layer_dimension[i]

mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
tf.add_to_collection('losses', mes_loss)
loss = tf.add_n(tf.get_collection('losses'))

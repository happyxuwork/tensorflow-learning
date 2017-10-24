# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_TATE_BASE = 0.8
LEARNING_TATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    y = inference(x, None, weights1, biases1, weights2, biases2)
    # define the training terms the variable do not need MovingAverage so set the trainable=False
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # tf.trainable_variables()---return the collection of Graph
    # variable_averages.apply(xx)---apply the EMA to the xx
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # get the movingaverage value
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    # @tf.argmax(y_, 1).
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    # dim = y_.shape()
    # logit = y
    # kk = np.reshape(l,newshape=[None,10])
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y_,logits = y)
    # y = tf.nn.softmax(y)

    # use the value that do not use movingaverage
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    # use the value that use the movingaverage
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1), logits=average_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization
    # 这里使用一种更加灵活的学习效率的设置方法---指数衰减法
    # 在minimize函数中传入global_step将自动更新global_step参数，从而使得学习率也得到相应的更新
    learning_rate = tf.train.exponential_decay(
        LEARNING_TATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_TATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps ,validation accracy " "using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps ,test accracy " "using average model is %g" % (TRAING_STEPS, test_acc))


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    # print(mnist.validation.lables())
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

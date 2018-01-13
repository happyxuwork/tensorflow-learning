# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference_cnn

# 配置参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01  # 学习的基本速率不一样，在第五章的神经网络中时，该值就为0.8
LEARNING_RATE_DECAY = 0.99
REGULARRZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径
MODEL_SAVE_PATH = "E:/tensorflow/savecnnmodelpath1/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    mnist_inference_cnn.IMAGE_SIZE,
                                    mnist_inference_cnn.IMAGE_SIZE,
                                    mnist_inference_cnn.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, mnist_inference_cnn.OUTPUT_NODE], name="y-input")
    regularizer = tf.contrib.layers.l2_regularizer(REGULARRZATION_RATE)
    y = mnist_inference_cnn.inference(x, False, regularizer)

    # trainable=False 可以防止该变量被数据流图的 GraphKeys.TRAINABLE_VARIABLES 收集, 这样我们就不会在
    #训练的时候尝试更新它的值，一句话，那就是这个变量不会在训练时进行更新，但是可以显示改变，如
    global_step = tf.Variable(0, trainable=False)

    #tf.train.ExponentialMovingAverage(衰减率，控制衰减率因子)--只是控制变量更新的一种方式（滑动平均模型）
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # tf.trainable_variables返回的是需要训练的变量列表--所以上面的global_step不会返回，因为trainable=False
    # tf.all_variables返回的是所有变量的列表---而上面的global_step会返回

    #每次执行下面代码时，列表中的变量都会被更新
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #解释下面的tf.argmax()
    # test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    # np.argmax(test, 0)　　　＃输出：array([3, 3, 1]
    # np.argmax(test, 1)　　　＃输出：array([2, 2, 0, 0]
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #tf.add_n(tf.get_collection('losses'))是将所有的全连接层的连接进行相加处理，
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #exponential_decay（）--一个控制学习率的函数--指数衰减法--先使用一个较大的学习率来快速获得一个比较优的解，然后随着迭代的继续逐步减少学习率
    #
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    #下面将glob_step = global_step传如minimize中，这样每进行一轮，globla_step的值就会加一，这样一改动global_step就可以同时改动衰减率和学习率
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #global_step: Optional `Variable` to increment by one after the
    #variables have been updated.

    # 在训练神经网络模型时，每过一遍数据，即需要通过反向传播来更新神经网络中的参数，
    # 又要更新每一个参数的滑动平均值，为了一次完成多种操作，Tensorflow提供了tf.control_dependencies和tf.group两种机制
    # train_op = tf.group(train_step,variables_averages_op)等价与如下两行代码：
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化TF持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # tf.train.latest_checkpoint来自动获取最后一次保存的模型
        chkpt_fname = tf.train.latest_checkpoint(MODEL_SAVE_PATH)
        if chkpt_fname is not None:
            # 模型的恢复用的是restore()函数，它需要两个参数restore(sess, save_path)，
            # save_path指的是保存的模型路径。我们可以使用tf.train.latest_checkpoint（）
            # 来自动获取最后一次保存的模型
            saver.restore(sess, chkpt_fname)
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference_cnn.IMAGE_SIZE,
                                          mnist_inference_cnn.IMAGE_SIZE,
                                          mnist_inference_cnn.NUM_CHANNELS
                                          ))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            if (i % 10 == 0):
                print("After %d training setps,loss on training " "batch is %g" % (step, loss_value))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()

# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name="x-input")
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 这里不需要关注正则化，因为这里是测试，并非模型的训练
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过重命名的方式来加载模型，这样在向前传播的过程中就不需要调用滑动平均函数来获取平均值了
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_resore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_resore)

        # 每个EVAL_INTERVAL_SECS 调用一次计算正确率的过程
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training steps,validation " "accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("NO checkpoint file found!")
                    return
                time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()

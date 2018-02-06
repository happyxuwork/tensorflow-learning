# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf
logits = tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
ground_truth_input = tf.constant([[0.0,0.0,1.0],[0.0,0.0,1.0],[0.0,0.0,1.0]])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)

#step1:do softmax
y=tf.nn.softmax(logits)
# step2:do cross_entropy
cross_entropy = -tf.reduce_sum(ground_truth_input * tf.log(y))
# do cross_entropy just one step
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input))  # dont forget tf.reduce_sum()!!

with tf.Session() as sess:
    softmax = sess.run(y)
    c_e = sess.run(cross_entropy)
    c_e2 = sess.run(cross_entropy2)
    print("step1:softmax result=")
    print(softmax)
    print("step2:cross_entropy result=")
    print(c_e)
    print("Function(softmax_cross_entropy_with_logits) result=")
    print(c_e2)
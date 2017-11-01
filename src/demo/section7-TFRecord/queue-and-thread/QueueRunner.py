# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

queue = tf.FIFOQueue(100, "float")
enqueue_op = queue.enqueue([tf.random_normal([1])])

# 启动5个线程，每个线程中执行enqueue_op操作
qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

# 将qr加入到默认的集合tf.GraphKeys.QUEUE_RUNNERS
tf.train.add_queue_runner(qr)
# 出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:
    # 使用Coordinator协同启动的线程
    coord = tf.train.Coordinator()
    # 调用tf.train.start_queue_runners()来启动所有的线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3):
        out = sess.run(out_tensor)
        print(sess.run(out_tensor)[0])
    # 使用tf.train.Coordinator来停止所有的线程
    coord.request_stop()
    coord.join(threads)

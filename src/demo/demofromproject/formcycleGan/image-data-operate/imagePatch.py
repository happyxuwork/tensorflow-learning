# -*- coding: UTF-8 -*-
'''
@author: xuqiang
主要是将需要加载图片放到一个.csv文件中，然后利用tf.train.string_input_producer形成一个队列
利用reader.read（）进行队列内容的读取。。。。。
'''

import tensorflow as tf
import matplotlib.pyplot as plt
# filename = "../data/picture/love.jpg"
# file_content = tf.read_file(filename)
# image = tf.image.decode_png(file_content)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     img = sess.run(image)
#     print(img.shape)
#     print(img)
#     plt.imshow(img)
#     plt.show()
def imageload():
    filename_queue = tf.train.string_input_producer(["text1.csv"])
    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),tf.constant([], dtype=tf.string)]

    filename_i,filename_j = tf.decode_csv(csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)

    image_decoded_A = tf.image.decode_jpeg(
        file_contents_i, channels=3)
    image_decoded_B = tf.image.decode_jpeg(
        file_contents_j, channels=3)
    return image_decoded_A,image_decoded_B

if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #一定要加这句话，不然队列会阻塞
        #tf.train.start_queue_runners()

        imgA,imgB = imageload()
        # imgA = tf.image.convert_image_dtype(imgA, dtype=tf.uint8)  # 改变图像数据的类型
        # imgB = tf.image.convert_image_dtype(imgB, dtype=tf.uint8)  # 改变图像数据的类型
        #一定要加这句话，不然队列会阻塞（定义成函数的话，先执行函数，然后执行下句话）
        tf.train.start_queue_runners()

        # i = sess.run(filename_i)
        # j = sess.run(filename_j)
        # print(i)
        # print("=================")
        # print(j)

        # imgA = sess.run(image_decoded_A)
        # imgB = sess.run(image_decoded_B)

        print(imgA.shape)
        print(imgA.eval())
        plt.imshow(imgA.eval())
        plt.show()

        print(imgB.shape)
        print(imgB.eval())
        plt.imshow(imgB.eval())
        plt.show()






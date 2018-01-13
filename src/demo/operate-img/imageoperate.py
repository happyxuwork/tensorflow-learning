# -*- coding: UTF-8 -*-
'''
@author: xuqiang
tensorflow读取图片，解码，显示
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #一定要加这句话，不然队列会阻塞（注意如果没有定义成函数的话，下面这句话要先）
    tf.train.start_queue_runners()
    i = sess.run(filename_i)
    j = sess.run(filename_j)
    print(i)
    print("=================")
    print(j)

    imgA = sess.run(image_decoded_A)
    imgB = sess.run(image_decoded_B)

    print(imgA.shape)
    print(imgA)
    plt.imshow(imgA)
    plt.show()


    #这句话的意思的将图像进行归一化处理，自动每个像素除以255得到浮点数，范围控制在[0,1]
    imgA1 = tf.image.convert_image_dtype(imgA, dtype=tf.float32)
    imgA1 = sess.run(imgA1)
    print(imgA1)

    #这句话的意思是将归一化的图像进行还原，将原本是[0,1]范围的float乘上255，得到原始图像的灰度值
    imgA2 = sess.run(tf.image.convert_image_dtype(imgA1,dtype=tf.uint8))
    print(imgA2)
    plt.imshow(imgA2)
    plt.show()

    #这句话的作用同上诉的一致
    plt.imshow(sess.run(tf.cast(((imgA1) * 255),dtype=tf.uint8)))
    plt.show()



    # 进行图片大小的调整，调整为image_size_before_crop=286大小
    image_decoded_A = tf.image.resize_images(imgA, [1084, 1924])
    # # 按特定的大小对调整后的图片进行修剪
    image_decoded_A = sess.run(tf.random_crop(image_decoded_A, [1082, 1920,3]))

    #上诉两句话的作用是将原本灰度值为整数变成浮点数，这样的话，就可以和127.5相除--另一种图像的归一化过程
    print(image_decoded_A.shape)
    print(image_decoded_A)

    plt.imshow(sess.run(tf.subtract(tf.div(image_decoded_A, 127.5), 1)))
    plt.show()
    img = sess.run(tf.subtract(tf.div(image_decoded_A, 127.5), 1))
    print(img)

    #下面的作用是将归一化的图像，进行还原
    img = sess.run(tf.cast(((img + 1) * 127.5), dtype=tf.uint8))
    print(img)
    plt.imshow(img)
    plt.show()






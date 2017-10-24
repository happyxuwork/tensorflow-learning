# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
# 图像的大小是不固定的，但是神经网络的输入是固定的，所有需要将图像的大小统一
# 这就是本实验需要完成的任务：
# 图像调整方式有两种：第一种是通过算法使得新的图像尽量保存原始图像上的所有信息
# tensorflow提供四中不同的方法，并封装到了tf.image.resize_images函数中。
#

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

img_raw_data = tf.gfile.FastGFile("../../../data/input/images/cat.jpg", 'rb').read()
img_data = tf.image.decode_jpeg(img_raw_data)
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(img_raw_data)
# img_data = tf.image.convert_image_dtype(img_data,dtype=tf.uint8)

# resized = tf.image.resize_images(img_data,[1300,1300],method=0)
# print(img_data.get_shape())
# print(resized.get_shape())
# print(resized.eval())
# plt.imshow(img_data.eval())
# plt.show()
#
# plt.imshow(resized.eval())
# plt.show()

'''
用不同的算法对图像进行裁剪
'''
# with tf.Session() as sess:
#     resized1 = tf.image.resize_images(img_data, [1300, 1300], method=0)
#     resized2 = tf.image.resize_images(img_data, [1300, 1300], method=1)
#     resized3 = tf.image.resize_images(img_data, [1300, 1300], method=2)
#     resized4 = tf.image.resize_images(img_data, [1300, 1300], method=3)
#     # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
#     #print("Digital type: ", resized.dtype)
#     cat1 = np.asarray(resized1.eval(), dtype='uint8')
#     cat2 = np.asarray(resized2.eval(), dtype='uint8')
#     cat3 = np.asarray(resized3.eval(), dtype='uint8')
#     cat4 = np.asarray(resized4.eval(), dtype='uint8')
#     tf.image.convert_image_dtype(resized1, tf.float32)
#     tf.image.convert_image_dtype(resized2, tf.float32)
#     tf.image.convert_image_dtype(resized3, tf.float32)
#     tf.image.convert_image_dtype(resized4, tf.float32)
#     plt.imshow(cat1)
#     plt.show()
#     plt.imshow(cat2)
#     plt.show()
#     plt.imshow(cat3)
#     plt.show()
#     plt.imshow(cat4)
#     plt.show()


'''
通过API对图像进行处理
'''
# with tf.Session() as sess:  #具体看API文档
#     croped = tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
#     padded = tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)
#     plt.imshow(croped.eval())
#     plt.show()
#     plt.imshow(padded.eval())
#     plt.show()
#     #g通过比例
#     central_cropped = tf.image.central_crop(img_data,0.5)
#     plt.imshow(central_cropped.eval()) #如果不带上.eval()则会报出TypeError: Image data can not convert to float错误
#     plt.show()

'''
图像的翻转
图像的翻转不会影响识别的结果
可以达到扩充样本的好处
'''
# with tf.Session() as sess:
#     #上下翻转
#     flipped = tf.image.flip_up_down(img_data)
#     plt.imshow(flipped.eval())
#     plt.show()
#     #左右翻转
#     rl = tf.image.flip_left_right(img_data)
#     plt.imshow(rl.eval())
#     plt.show()
#     #对角线翻
#     dj = tf.image.transpose_image(img_data)
#     plt.imshow(dj.eval())
#     plt.show()
# 以一定的概率翻转
# ff = tf.image.random_flip_left_right(img_data)

'''
图像色彩调整
'''
# with tf.Session() as sess:
#     adhig = tf.image.adjust_brightness(img_data,0.5)
#     plt.imshow(adhig.eval())
#     plt.show()
#     adlow = tf.image.adjust_brightness(img_data, -0.5)
#     plt.imshow(adlow.eval())
#     plt.show()
#     #在-5---5范围内随机调整亮度
#     rmd = tf.image.random_brightness(img_data,5)
#     plt.imshow(rmd.eval())
#     plt.show()

'''
对比度的调整
'''
# with tf.Session() as sess:
#     ad = tf.image.adjust_contrast(img_data,-5)
#     plt.imshow(ad.eval())
#     plt.show()

'''
色相的调整
'''
# with tf.Session() as sess:
#     ad = tf.image.adjust_hue(img_data,0.4)
#     plt.imshow(ad.eval())
#     plt.show()

'''
图像的标准化
'''
# with tf.Session() as sess:
#     #将一张代表图像的三维矩阵中的数字均值变为0，方差变为1
#     ad = tf.image.per_image_standardization(img_data)
#     plt.imshow(ad.eval())
#     plt.show()

'''
处理标注框
'''
# with tf.Session() as sess:
#     #将图像缩小一些，可以使得标注框更明显
#     img_data = tf.image.resize_images(img_data,[180,267],method=0)
#     #加一维，因为进入tf.image.draw_bounding_boxes()中的是batch,四维的
#     batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
#     boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
#     result = tf.image.draw_bounding_boxes(batched,boxes)
#     plt.imshow(img_data.eval())
#     plt.show()

'''
标注并裁剪
tf.image.sample_distorted_bounding_box():解释：http://blog.csdn.net/tz_zs/article/details/77920116
返回值为3个张量：begin，size和 bboxes。前2个张量用于 tf.slice 剪裁图像。后者可以用于 tf.image.draw_bounding_boxes 函数来画出边界框
'''
with tf.Session() as sess:
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)

    distorted_image = tf.slice(img_data, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()

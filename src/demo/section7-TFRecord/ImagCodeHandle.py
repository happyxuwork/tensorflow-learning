# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 后面的参数一定要写rb,写r会出错
# image_raw_data = tf.gfile.FastGFile("../../../data/input/images/cat.jpg",'rb').read()
image_raw_data = tf.gfile.FastGFile("../../../data/input/images/c.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    # img_data = tf.image.resize_images(img_data,[180,267],method=1)
    # result = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)
    # result = tf.image.encode_jpeg(result)
    # with tf.gfile.GFile("C:/Users/Administrator/Desktop/tensorflow-tutorial-master/tensorflow-tutorial-master/Deep_Learning_with_TensorFlow/datasets/flower_photos/out/" + "cat.jpg",'wb') as f:
    #     f.write(result.eval())


    # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # boxes = tf.constant([[[0.0, 0.0, 0.5, 0.5]]])

    # begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
    #     tf.shape(img_data), bounding_boxes=boxes)
    # print("begin="+str(begin))
    # print("size="+str(size))

    # batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    # image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)

    # distorted_image = tf.slice(img_data, begin, size)








    for i in range(36):
        for j in range(76):
            b_k = i
            k_k = 1500
            b_g = j
            k_g = 1100
            img_5 = tf.image.crop_to_bounding_box(img_data, b_g, b_k, k_g, k_k)
            result = tf.image.convert_image_dtype(img_5, dtype=tf.uint8)
            result = tf.image.encode_jpeg(result)
            with tf.gfile.GFile(
                                                    "C:/Users/Administrator/Desktop/tensorflow-tutorial-master/tensorflow-tutorial-master/Deep_Learning_with_TensorFlow/datasets/flower_photos/out/" + "cat" + str(
                                            i) + str(j) + ".jpg", 'wb') as f:
                f.write(result.eval())







                # #print(result.get_shape())
                # plt.imshow(result.eval())
                # plt.show()








# with tf.Session() as sess:
#     #使用tf.image.decode.jpeg()来解码图像，得到图像对应的三维矩阵，但是tf.image.decode.png()解码之后是一个张量
#     img_data = tf.image.decode_jpeg(image_raw_data)
#     #使用resize_images()改变图片的大小，注意TensorFlow的函数处理图片后存储的数据是float32格式的
#     resized = tf.image.resize_images(img_data, [300, 300], method=2)
#    #TensorFlow的函数处理图片后存储的数据是float32格式--这种格式通用性很好
#    # 但是要将图片进行打印，或是存储的化，需要转换成uint8
#     resized = np.asarray(resized.eval(), dtype='uint8')
#
#
#     #使用pyplot工具可视化得到的图像
#     #eval()函数---将字符串str当成有效的表达式来求值并返回计算结果
#     # plt.imshow(img_data.eval())
#     # plt.show()
#
#
#     #将图片重新按照一定的编码方式进行编码
#     encode_image = tf.image.encode_jpeg(resized)
#     #存入到文件中，这里需要特别注意的是：如果在路径中没有指明图像的名称和后缀的化，会报如下的错误
#     # Failed to create a NewWriteableFile: ../../../data/output/ : \udcbeܾ\udcf8\udcb7\udcc3\udcceʡ\udca3; Input/output error
#     with tf.gfile.GFile("../../../data/output/cat.jpeg",'wb') as f:
#         f.write(encode_image.eval())
#     #一般情况下，在对图像进打印或是保存完成后，要将图像转成tf.float32---这样的通用性较好

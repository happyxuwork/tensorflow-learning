# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import collections
import glob
from tensorflow.python.framework.graph_util import convert_variables_to_constants

# nameAndPositon = {'orange':'0','apple':'1','one':'2','zebra':'3','chinese':'5','oil':'6','many':'7'}
# print(nameAndPositon.keys())

'''
提取文件名中，，，
'''
# name = "oiltochinese-15-15-test"
# values = name.split("to")
# first = values[0]
# second = values[1].split("-")[0]
# print(first)
# print(second)
# print(nameAndPositon[first])

def getOneImgClassifyProb(sess,imageToTensorPath):
    # with tf.Graph().as_default():
    # chkName = tf.train.latest_checkpoint('E:/tensorflow/selfmodelclassify/')
    # saver = tf.train.import_meta_graph(chkName + '.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('E:/tensorflow/selfmodelclassify/'))
    with open(imageToTensorPath,'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.strip().split(',')]
    bottleneck_values_reshape = np.reshape(bottleneck_values, [-1, 2048])
    graph = tf.get_default_graph()
    input = graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
    w1 = graph.get_tensor_by_name("final_training_ops/op_to_prob:0")
    feed_dict = {input: bottleneck_values_reshape}
    prob = sess.run(w1, feed_dict)
    return prob

def getListImgClassifyProb(sess,imageList):
    listLen = len(imageList)
    totalImgProb = np.zeros(22)
    for imgPath in imageList:
        oneImgProb = getOneImgClassifyProb(sess,imgPath)
        totalImgProb = np.add(totalImgProb,oneImgProb)
    return np.divide(totalImgProb,listLen)

def main(input_data_tensor_valuation_path,input_data_training_path):

    # get the position dict of classes
    nameAndPositon = collections.OrderedDict()
    class_sub_dirs = [x[0] for x in os.walk(input_data_training_path)]
    len_dirs = len(class_sub_dirs)
    is_root_dir = True
    for sub_dir,i in zip(class_sub_dirs,range(len_dirs)):
        if is_root_dir:
            is_root_dir = False
            continue
        base_name = os.path.basename(sub_dir)
        nameAndPositon[base_name] = str(i-1)
    print(nameAndPositon)
    print(nameAndPositon['apple'])
    print(type(nameAndPositon))
    # 22
    # 分类位置备注
    # ['apple', 'chinese', 'edges', 'hands', 'hedges', 'hgymany', 'hgyone', 'horse', 'many', 'map', 'monet', 'nosmile',
    #  'oil', 'one', 'orange', 'photo', 'shoes', 'smile', 'summer', 'vango', 'winter', 'zebra']


    # nameAndPositon = {'orange': '0', 'apple': '1', 'one': '2', 'zebra': '3','horse':'4','chinese': '5', 'oil': '6', 'many': '7'}
    sub_dirs = [x[0] for x in os.walk(input_data_tensor_valuation_path)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        base_name = os.path.basename(sub_dir)
        file_name_apart = base_name.split("to")
        first_name = file_name_apart[0]
        second_name = file_name_apart[1].split("-")[0]
        prefixs = ['fakeA','fakeB']
        file_list = {}
        for prefix in prefixs:
            file_glob = os.path.join(input_data_tensor_valuation_path,base_name,prefix+'*')
            file_list[prefix] = glob.glob(file_glob)

        fakeA_img_list = file_list['fakeA']
        fakeB_img_list = file_list['fakeB']
        print(base_name + "文件夹底下共" + str(len(fakeA_img_list)+len(fakeB_img_list))+"张图像")

        with tf.Session() as sess:
            chkName = tf.train.latest_checkpoint('E:/tensorflow/22classify/')
            saver = tf.train.import_meta_graph(chkName + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('E:/tensorflow/22classify/'))
            fakeA_img_average_prob = getListImgClassifyProb(sess,fakeA_img_list)
            fakeB_img_average_prob = getListImgClassifyProb(sess,fakeB_img_list)
            # print(fakeA_img_average_prob)
            # print(fakeB_img_average_prob)
            positionA = int(nameAndPositon[first_name])
            positionB = int(nameAndPositon[second_name])
            print(base_name+"文件中fakeA的平均准确率为："+str(fakeA_img_average_prob[0][positionB]))
            print(base_name+"文件中fakeB的平均准确率为："+str(fakeB_img_average_prob[0][positionA]))
            print("++++++++++++++++++++++++++++++++++")
            with open(input_data_tensor_valuation_path+'avgAccOutput.txt','a') as f:
                f.write(base_name+"文件中fakeA的平均准确率为："+str(fakeA_img_average_prob[0][positionB])+"\n"+
                        base_name + "文件中fakeB的平均准确率为：" + str(fakeB_img_average_prob[0][positionA])+"\n\n")


if __name__ == "__main__":
    # list1 = np.zeros(8)
    # list2 = [2.83964584e-03,7.56550930e-04,6.08784103e-06,3.81935941e-04,1.47359213e-04,7.49921858e-01,2.45923772e-01,2.27466589e-05]
    # print(list1)
    # addTo = np.add(list1, list2)
    # print(addTo)
    # print(np.divide(addTo,2))

    # input_data_tensor_valuation_path = 'F:/converImgtoTensor/tensor-valuation/'
    input_data_tensor_valuation_path = 'F:/converImgtoTensor/tensor-valuation/'
    input_data_training_path = 'F:/converImgtoTensor/tensor-training/'
    main(input_data_tensor_valuation_path,input_data_training_path)























# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform  import gfile


BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = '../data/model'
MODEL_FILE ='tensorflow_inception_graph.pb'
# CHECK_DIR = '../data/dataforclassify/test/'
CHECK_DIR = 'F:/converImgtoTensor/adding/tensor-valuation/'
# INPUT_DATA = '../data/dataforclassify/mix/'
INPUT_DATA = 'F:/converImgtoTensor/adding/data-valuation/'

# 模型保存路径
MODEL_SAVE_PATH = "E:/tensorflow/classify"
MODEL_NAME = "model.ckpt"

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    '''
    使用加载好的Inception-v3模型处理一张图片得到这张图片的特征向量
    :param sess:
    :param image_data:
    :param image_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def create_bottleneck(sess,image_path,bottleneck_path,jpeg_data_tensor,bottleneck_tensor):
        image_data = gfile.FastGFile(image_path,'rb').read()
        #通过Inception-v3计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        # print("bottleneck_values"+bottleneck_path)
        #将计算结果存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
        #返回得到的特征向量
        return tf.convert_to_tensor(bottleneck_values)



def main(InputData_path):

    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载读取的Inception-v3模型，返回数据输入对应的张量以及计算瓶颈层结果对应的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])

    # bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name="BottleneckInputPlaceholder")
    # ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name="GroundTruthInput")

    sub_dirs = [x[0] for x in os.walk(InputData_path)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        parent_path = os.path.join(CHECK_DIR,dir_name)
        os.mkdir(parent_path)
        extensions = ['jpg','jpeg']
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(InputData_path,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        num_images = len(file_list)
        print("++++++++++++++++++++++++++++")
        print(dir_name+"文件夹底下共"+str(num_images)+"张图像")
        kk = os.path.join(CHECK_DIR, sub_dir)
        # os.makedirs(os.path.join(CHECK_DIR,sub_dir))
        with tf.Session() as sess:
            for file_image in file_list:
                img_name = os.path.basename(os.path.splitext(file_image)[0])
                img_suffix = os.path.splitext(file_image)[1]
                bottleneck_path = os.path.join(CHECK_DIR,dir_name,img_name+img_suffix)+'.txt'
                # bottleneck_path = "D://yes//"+dir_name+"//"+img_name+'.txt'
                # bottleneck_path = 'r'+bottleneck_path
                # print("----"+bottleneck_path)
                # os.makedirs(bottleneck_path)
                bottleneck_value = create_bottleneck(sess,file_image,bottleneck_path,jpeg_data_tensor,bottleneck_tensor)
                sess.run(bottleneck_value)
                # print(bottleneck_value)

if __name__ == "__main__":
    main(INPUT_DATA)







































#从数据文件夹中读取所有的图片列表，并按训练，验证，测试数据分开
def create_image_lists(testing_percentage,validation_percnetage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg','jpeg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:continue

        #通过目录名获取类别名称
        label_name = dir_name.lower()
        data_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            data_images.append(base_name)
        result[label_name]={
            'dir':dir_name,
            'data':data_images
        }
    return result


def get_image_path(image_lists,image_dir,index):
    '''
    get a specific imag from the images_lists
    :param image_lists:the all images set
    :param index:the index of image in images
    :param image_dir:the category of images
    :return:the full path of the image
    '''
    #获取给定类别的所有图片
    label_lists = image_lists[dir_name]
    #获取所属数据集的所有图片
    mod_index = index % len(label_lists)
    #图片名称
    base_name = label_lists[mod_index]
    sub_dir = label_lists['dir']
    #最终地址为数据根目录地址加上类别文件加上图片名称
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path

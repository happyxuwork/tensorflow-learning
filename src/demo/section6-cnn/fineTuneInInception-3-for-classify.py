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
CHECK_DIR = '../data/bottleneck/mix/'
INPUT_DATA = '../data/dataforclassify/mix/'

CHECK_DIR = 'F:/converImgtoTensor/tensor/'
INPUT_DATA = 'F:/converImgtoTensor/data/'

# 模型保存路径
MODEL_SAVE_PATH = "E:/tensorflow/selfmodelclassify/"
MODEL_NAME = "model.ckpt"

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 10000
BATCH = 100


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
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            chance = np.random.randint(100)
            if chance < VALIDATION_PERCENTAGE:
                validation_images.append(base_name)
            elif chance < (VALIDATION_PERCENTAGE+TEST_PERCENTAGE):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name]={
            'dir':dir_name,
            'training':training_images,
            'testing':testing_images,
            'validation':validation_images,
        }
    return result

'''
通过类别名称、所属数据集和图片编号获取一张图片的地址
  Args:
    image_lists:所有图片信息
    image_dir:跟目录
    label_name:类别名称
    index:需要获取图片的编号
    category:指定需要的图片是处于训练集，验证集或是测试集
  Returns:
    full_path:图片的路径
'''
def get_image_path(image_lists,image_dir,label_name,index,category):
    #获取给定类别的所有图片
    label_lists = image_lists[label_name]
    #获取所属数据集的所有图片
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    #图片名称
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    #最终地址为数据根目录地址加上类别文件加上图片名称
    full_path = os.path.join(image_dir,sub_dir,base_name)
    return full_path


def get_bottleneck_path(image_lists,label_name,index,category):
    '''
    通过类别名称、所属数据集和图片编号获取经过Inception-v3模型处理之后的特征向量文件地址
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :return:
    '''
    return get_image_path(image_lists,CHECK_DIR,label_name,index,category)+'.txt'

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

def get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    '''
    获取一张图片进过Inception-v3模型处理之后的特征向量。会先去试图寻找已经保存下来的特征向量，如果找不到的话，则计算这个特征向量，然后保存
    :param sess:
    :param image_lists:
    :param label_name:
    :param index:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    #获取对应图片的特征向量的文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CHECK_DIR,sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)

    #如果特征向量不存在，则通过Inception-v3模型计算，并保存
    if not os.path.exists(bottleneck_path):
        #获取原始图片路径
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        #获取图片内容
        image_data = gfile.FastGFile(image_path,'rb').read()
        #通过Inception-v3计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        #将计算结果存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        #直接从文件中获取相应的特征向量
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        # bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        bottleneck_values = [float(x) for x in bottleneck_string.strip().split(',')]


    #返回得到的特征向量
    return bottleneck_values

def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
    '''

    :param sess:
    :param n_classes:
    :param image_lists:
    :param how_many:
    :param category:
    :param jpeg_data_tensor:
    :param bottleneck_tensor:
    :return:
    '''
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        #randrange(n_classes)---[0,end)
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes,dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    lable_name_list = list(image_lists.keys())
    #枚举所有类别和每个类别中的测试图片
    for label_index,label_name in enumerate(lable_name_list):
        category = 'testing'
        for index,unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor)
            ground_truth = np.zeros(n_classes,dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def main():
    image_lists = create_image_lists(TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())

    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #加载读取的Inception-v3模型，返回数据输入对应的张量以及计算瓶颈层结果对应的张量
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

    bottleneck_input = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name="BottleneckInputPlaceholder")
    ground_truth_input = tf.placeholder(tf.float32,[None,n_classes],name="GroundTruthInput")

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001),name='weight')
        biases = tf.Variable(tf.zeros([n_classes]),name='biases')
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits,name="op_to_prob")
        # print(final_tensor)

        # 损失
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 优化
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化TF持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化参数
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            # 训练
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # 验证
            if i % 1000 == 0 or i + 1 == STEPS:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=i)
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(sess, n_classes,
                                                                                                image_lists, BATCH,
                                                                                                'validation',
                                                                                                jpeg_data_tensor,
                                                                                                bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: validation_bottlenecks,
                                                                           ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % (
                    i, BATCH, validation_accuracy * 100))
                print(validation_bottlenecks[0])
                # print(sess.run(final_tensor,feed_dict={bottleneck_input:validation_bottlenecks}))

                # 测试
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                                                                   bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,
                                 feed_dict={bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    # classer()
     main()

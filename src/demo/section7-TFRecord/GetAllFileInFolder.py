# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''

import os


def getFileName(path):
    canreturn = ['.jpg', '.jpeg', '.gif']
    all = []
    if os.path.exists(path):
        f_list = os.listdir(path)
        for i in f_list:
            if os.path.splitext(i)[1] in canreturn:
                all.append(i)
    return all

# if __name__ == '__main__':
#     path = 'C:/Users/Administrator/Desktop/tensorflow-tutorial-master/tensorflow-tutorial-master/Deep_Learning_with_TensorFlow/datasets/flower_photos/daisy'
#     listd = getFileName(path)
#     for name in listd:
#         print(name)

# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
from scipy.misc import imsave
import numpy as np
import pickle as cPickle
import os

# 解压缩，返回解压后的字典
def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo,encoding='bytes')
    fo.close()
    return dict

# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
in_path = "F:/CIFAR10/"
out_path = "F:/CIFAR10/png/"
for j in range(1, 6):
    dataName = "data_batch_" + str(j)  # 读取当前目录下的data_batch12345文件，dataName其实也是data_batch文件的路径，本文和脚本文件在同一目录下。
    Xtr = unpickle(os.path.join(in_path,dataName))
    print(dataName + " is loading...")

    for i in range(0, 10000):
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
        img = img.transpose(1, 2, 0)  # 读取image
        picName = 'train/' + str(Xtr[b'labels'][i]) + '_' + str(i + (j - 1)*10000) + '.png'  # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        imsave(os.path.join(out_path,picName), img)
    print(dataName + " loaded.")

print("test_batch is loading...")

# 生成测试集图片
testXtr = unpickle(os.path.join(in_path,"test_batch"))
for i in range(0, 10000):
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(testXtr[b'labels'][i]) + '_' + str(i) + '.jpg'
    imsave(os.path.join(out_path,picName), img)
print("test_batch loaded.")
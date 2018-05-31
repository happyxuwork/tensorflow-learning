# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import numpy as np


c = np.load("fid_stats_cifar10_train.npz")
b = np.load("cifar10.npz")
print(c['mu'])
print(b['mu'])
# for i,j in zip(c['mu'].__len__(),b['mu'].__len__()):
#        print(c['mu'][i]+"    "+b['mu'][i])
print(c['sigma'])
print(b['sigma'])
# print(c['a'])
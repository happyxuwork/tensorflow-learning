# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import os
path = '../data/dataforclassify/test/dain/100080576_f52e8ee070_n.txt'
print(os.getcwd()) #打印出当前工作目录
with open(path, 'w') as bottleneck_file:
    bottleneck_file.write("adfasdf")
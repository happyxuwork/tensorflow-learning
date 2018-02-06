# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import collections
# 这样可以保证字典中添加的是有顺序的
nameAndPositon = collections.OrderedDict()
nameAndPositon['orange1'] = '1'
nameAndPositon['orange2'] = '2'
nameAndPositon['orange3'] = '3'
nameAndPositon['orange4'] = '4'
nameAndPositon['orange5'] = '5'
keys1 = list(nameAndPositon.keys())[1]
keys2 = list(nameAndPositon.keys())
print(keys1)
print(keys2)

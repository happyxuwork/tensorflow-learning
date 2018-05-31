# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
# a = [1,2,3,4,5]
#
# print(a[0:3])
# print(a[0:3])
import numpy as np
data = np.random.uniform(low=0., high=15., size=1000)
split_data = np.split(data, 10)
print(np.mean([np.exp(np.mean(x)) for x in split_data])) # 1608.25
print(np.exp(np.mean(data))) # 1477.25
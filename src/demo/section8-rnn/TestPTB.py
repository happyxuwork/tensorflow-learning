# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import reader
import os
import pprint
CURRENT_PATH = ".."
PARENT_PATH = "./"
DATA_PATH = "../data/PTB/data/"

abs1 = os.path.abspath(CURRENT_PATH)
abs2 = os.path.abspath(PARENT_PATH)
abs3 = os.path.abspath(DATA_PATH)
train_data, valid_data, test_data, vocabulary = reader.ptb_raw_data(DATA_PATH)

print(len(train_data))
pprint.pprint(train_data[:])



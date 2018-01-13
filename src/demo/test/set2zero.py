# -*- coding: UTF-8 -*-
__author__ = "xdg"

import  pandas as pd
import  numpy as np
import  random

# df =pd.read_csv("3415.csv",encoding="UTF-8",header=None)
# datamat=df.as_matrix()
# salecount=datamat[:,2]
# for i in range(len(salecount)):
# 	if(salecount[i]==0):
# 		prob=random.random()
# 		if prob>0.7:
# 			datamat[:,2][i]=1
#
# pd.DataFrame(datamat).to_csv("sample_2.csv",encoding="UTF-8")
import os
import glob


input_path = "./data-lstm/"
fasd = os.path.abspath(input_path)
# sub_dirs = [x[0] for x in os.walk(input_path)]
# is_root_dir = True
# for sub_dir in sub_dirs:
#     if is_root_dir:
#         is_root_dir = False
#         continue
dir_name = os.path.basename(input_path)
extensions = ['csv']
file_list = []

parent_path = os.path.dirname(input_path + os.path.sep + "../")
path = parent_path + "/extend/" + dir_name + "/"
os.makedirs(path)

for extension in extensions:
    file_glob = os.path.join(input_path, dir_name, '*.' + extension)
    # get all the images of one folder of the input_path
    file_list.extend(glob.glob(file_glob))
for file in file_list:
    file_name = os.path.basename(os.path.splitext(file)[0])
    file_suffix = os.path.splitext(file)[1]
    df =pd.read_csv(file,encoding="UTF-8",index_col=None)
    datamat=df.as_matrix()
    datamat=datamat[:120,-1]
    pd.DataFrame(datamat).to_csv(path+file_name+file_suffix,index=False,header=False, encoding="UTF-8")






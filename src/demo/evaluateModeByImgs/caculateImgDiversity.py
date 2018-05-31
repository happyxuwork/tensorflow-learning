# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import os
import numpy as np
from skimage import exposure,data
from skimage import io
import math
import glob
import random


def calculateODisBetweenTwoImages(imgOne,imgTwo):
    imgOne = io.imread(imgOne, as_grey=False)
    imgOneToOneD = imgOne.flatten()

    imgTwo = io.imread(imgTwo, as_grey=False)
    imgTwoToOneD = imgTwo.flatten()

    arrSub = np.subtract(imgOneToOneD,imgTwoToOneD)
    arrPow2 = np.power(arrSub,2)
    arrSum = np.sum(arrPow2)
    arrSqrt = np.sqrt(arrSum)
    return arrSqrt


def dateSetApart(inputPth):
    print("start caculate diversity ...")
    # sub_dirs = [x[1] for x in os.walk(inputPth)]
    sub_dirs = [x[1] for x in os.walk(inputPth)]
    # print(sub_dirs[0])
    sub_dirs[0].remove('realimg')
    arr_len = len(sub_dirs[0])
    result_val_arr = np.zeros(arr_len)
    result_winer_arr = np.zeros(arr_len)
    list_file_name = []
    file_dict = {}
    for sub_dir, k in zip(sub_dirs[0], range(arr_len)):
    # sub_dirs = [x[0] for x in os.walk(inputPth)]
    # is_root_dir = True
    # for sub_dir in sub_dirs:
    #     if is_root_dir:
    #         is_root_dir = False
    #         continue
        dir_name = os.path.basename(sub_dir)
        list_file_name.extend([dir_name])
        extensions = ['jpg']
        file_list_all = []
        for extension in extensions:
            file_glob = os.path.join(inputPth,dir_name,'*.'+extension)
            file_dict[dir_name+extension] = glob.glob(file_glob)
            file_list_all.extend(glob.glob(file_glob))
        print(dir_name+"文件底下共有图像"+str(len(file_list_all))+"张")
        folderName = dir_name
        list_one_fakeA = file_list_all
        diversity_one_fakeA = []
        for i in range(15):
            arrLen = len(list_one_fakeA)
            index1 = random.randint(0, (arrLen - 1))
            index2 = random.randint(0, (arrLen - 1))
            while index1 == index2:
                index2 = random.randint(0, (arrLen - 1))
            list_one_fakeA_copy = list_one_fakeA.copy()

            diversity_one_fakeA.append(calculateODisBetweenTwoImages(list_one_fakeA[index1], list_one_fakeA[index2]))

            # del list_one_fakeA_copy[index1]

        sum_diversity_one_fakeA = np.sum(diversity_one_fakeA)
        print(folderName + "集合的多样性值: " + str(sum_diversity_one_fakeA))
        result_val_arr[k] = sum_diversity_one_fakeA
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    max_index = np.where(np.max(result_val_arr) == result_val_arr)
    result_winer_arr[max_index] = 1
    print("Diversity caculated done...")
    # print(sub_dirs[0])
    return result_val_arr, result_winer_arr


def main(inputPath):
    result_val_arr, result_winer_arr = dateSetApart(inputPath)
    return result_val_arr,result_winer_arr

if __name__ == "__main__":
    # inputFolderPath = "F:/研究生/图像数据/数据/其它/calculate/补充/calculate-diversity/"
    # inputFolderPath = "F:/研究生/图像数据/数据/其它/calculate/calculate-input-diversity/zebra/"
    inputFolderPath = "F:/train1/"
    cal, arr = dateSetApart(inputFolderPath)
    print(cal)
    print(arr)
    # nearestImageDisInOneSet("../data-caculate-complex/fakeA_100_0.jpg","../data-caculate-complex/demo/")
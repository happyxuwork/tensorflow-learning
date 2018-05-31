# -*-encoding=UTF-8 -*-
'''
auther:xuqiang
'''
import tensorflow as tf
import pathlib
from scipy.misc import imread
from scipy import linalg
import numpy as np
import os
import glob
import convertImageToV3BacknetTensor
import fid
import classifyAcc
import inception_score
import caculateImgDiversity

inception_path = "G:/classify_image_graph_def.pb"
image_all_to_npz_path = "F:/alltoNPZ/"
generate_image_path = "F:/train1/"
image_all_to_tensor_path = "F:/alltoTensor"
input_data_training_path = 'F:/converImgtoTensor/tensor-training/'


if __name__=="__main__":
    #----------------------计算FID分数--------------------------------
    # 注意
    # 所有的图像文件放在同一个文件夹底下：包括真实的图形，多个生成的图像（用不同的文件夹区分）
    # 真实的图像训练集名称固定为realimg
    #还有重要的一点就是每个文件夹下的图像大小要一致
    # sub_dirs = [x[0] for x in os.walk(generate_image_path)]
    # del sub_dirs[0]
    # print(sub_dirs)
    # image_fake_path = sub_dirs
    # saveImgToNPZFile(inception_path,image_fake_path,image_all_to_npz_path)
    # npz_sub_dirs = [x[2] for x in os.walk(image_all_to_npz_path)]
    # 得到生成images相对与训练集的fid值，fid值越小，质量越高
    with tf.Session() as sess:
        # ----------------------计算FID分数--------------------------------
        # fid_val_arr,fid_winer_arr = fid.caculate_img_fid(sess,inception_path,generate_image_path,image_all_to_npz_path)
        # print(fid_val_arr)
        # print(fid_winer_arr)
        # -----------------------------------------------------------------

        # ----------------------计算分类准确性--------------------------------
        # 分类准确性度量：
        #第一步：将所有的图像转成BacknetTensor
        # convertImageToV3BacknetTensor.main(sess, inception_path, generate_image_path, image_all_to_tensor_path)
        #第二步：将得到的tensor送入到训练好的模型中，
        #取分类的平均准确性指标
        # acc_val_arr, acc_winer_arr = classifyAcc.main(sess,image_all_to_tensor_path,input_data_training_path)
        # print(acc_val_arr)
        # print(acc_winer_arr)
        # ----------------------------------------------------------------------


        # ----------------------计算Inception score--------------------------------
        # is_val_arr, is_winer_arr = inception_score.main(sess,inception_path,generate_image_path)
        # print(is_val_arr)
        # print(is_winer_arr)
        # -------------------------------------------------------------------------

        # ----------------------计算SWD距离--------------------------------
        # div_val_arr, div_winer_arr = caculateImgDiversity.main(generate_image_path)
        # print(div_val_arr)
        # print(div_winer_arr)
        # -------------------------------------------------------------------------


        # ----------------------计算多样性--------------------------------
        # div_val_arr, div_winer_arr = caculateImgDiversity.main(generate_image_path)
        # print(div_val_arr)
        # print(div_winer_arr)
        # -------------------------------------------------------------------------

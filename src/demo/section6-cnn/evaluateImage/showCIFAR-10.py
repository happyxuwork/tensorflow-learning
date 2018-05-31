import gzip
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
# import pywt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# def singlewaveletdenose(im,savepath):
#     # im = Image.open(dir)
#     data = im.getdata()
#     data = np.array(data)
#     print(data.shape)
#     data = data.reshape((32, 32, 3))
#     R = data[:, :, 0]
#     G = data[:, :, 1]
#     B = data[:, :, 2]
#     db1 = pywt.Wavelet('db1')
#     rA3, (rD3, rD2, rD1) = pywt.dwt2(R, db1)
#     gA3, (gD3, gD2, gD1) = pywt.dwt2(G, db1)
#     bA3, (bD3, bD2, bD1) = pywt.dwt2(B, db1)
#     irA3 = pywt.idwt2((rA3, (None, None, None)), db1).reshape((1, -1))[0]
#     igA3 = pywt.idwt2((gA3, (None, None, None)), db1).reshape((1, -1))[0]
#     ibA3 = pywt.idwt2((bA3, (None, None, None)), db1).reshape((1, -1))[0]
#     # rA3, (rH2, rV2, rD2), (rH1, rV1, rD1) = pywt.wavedec2(R, db1, level=2)
#     # gA3, (gH2, gV2, gD2), (gH1, gV1, gD1) = pywt.wavedec2(G, db1, level=2)
#     # bA3, (bH2, bV2, bD2), (bH1, bV1, bD1) = pywt.wavedec2(B, db1, level=2)
#     # irA3 = pywt.waverec2((rA3, (None, None, None),(None, None, None)), db1).reshape((1, -1))[0]
#     # igA3 = pywt.waverec2((gA3, (None, None, None),(None, None, None)), db1).reshape((1, -1))[0]
#     # ibA3 = pywt.waverec2((bA3, (None, None, None),(None, None, None)), db1).reshape((1, -1))[0]
#     imgdata = zip(irA3, igA3, ibA3)
#     imgdata = [list(x) for x in imgdata]
#     imgdata = np.array(imgdata).reshape((32,32,3)).astype(np.uint8)
#
#     # print(imgdata.shape)
#     img = Image.fromarray(imgdata).convert('RGB')
#     # return img
#     img.save(savepath, 'JPEG')

traindict=unpickle("E:/戴鹏/论文/CIFAR-10/cifar-10-batches-py/data_batch_1")
# print(traindict)
train_img= traindict[b'data']
train_label = traindict[b'labels']
train_img=np.array(train_img)
train_label=np.array(train_label)
train_img=train_img.reshape((10000,3,32,32)).transpose(0,2,3,1)
# train_img=train_img[0:2000]

i=0
for temp in train_img:
    new_img=Image.fromarray(temp)
    # singlewaveletdenose(new_img,"E:/戴鹏/论文/CIFAR-10/去噪后图片/"+str(i)+".jpg")
    new_img.save("E:/戴鹏/论文/CIFAR-10/完整图片/"+str(i)+".jpg")
    i=i+1
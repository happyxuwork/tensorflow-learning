# -*- coding: UTF-8 -*-
__author__ = "xdg"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import os
from  json import *
import  random
import  time


def load_data(file_name, sequence_length=10):  # 加参数split=0.75用来划分训练集和测试集
    df = pd.read_csv(file_name, sep=',', usecols=[1])
    data_all = np.array(df).astype(float)
    print ("数据集大小为",data_all.shape)
    #print ("数据集为",data_all)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    #split_boundary = int(reshaped_data.shape[0] * split)
    # train_x = x[: split_boundary]
    # test_x = x[split_boundary:]
    train_x = x[:-30]
    test_x = x[-30:]

    # train_y = y[: split_boundary]
    # test_y = y[split_boundary:]

    train_y = y[:-30]
    test_y = y[-30:]
    return train_x, train_y, test_x, test_y, scaler


def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size=512, nb_epoch=10, validation_split=0.1)
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    # print(predict)
    # print(test_y)
    # try:
    #     fig = plt.figure(1)
    #     # plt.plot(predict, 'r:')
    #     # plt.plot(test_y, 'g-')
    #     # plt.legend(['predict', 'true'])
    # except Exception as e:
    #     print(e)
    return predict, test_y

#找文件下的文件
def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print root
        # print(dirs)
        new_files=files
    return new_files



def predict(path):

    train_x, train_y, test_x, test_y, scaler = load_data(path)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    predict_y, test_y = train_model(train_x, train_y, test_x, test_y)
    predict_y = scaler.inverse_transform([[i] for i in predict_y])
    # print("反归一化预测结果：\n", predict_y)
    test_y = scaler.inverse_transform(test_y)
    # print("反归一化实际结果：\n", test_y)
    #输出曲线图
    # fig2 = plt.figure(2)
    # plt.plot(predict_y, 'g:')
    # plt.plot(test_y, 'r-')
    # plt.show()
    return predict_y

if __name__ == '__main__':

    path="G:/Git/304first/data/DateData/"
    #记录开始时间
    start=time.time()
    # 得到Big文件夹下的所有类文件名
    filenames = file_name(path+"/Big")
    # 得到Mid文件夹下的所有类文件名
    filenames2 = file_name(path+"/Mid")

    # 读取sample.csv文件，得到矩阵
    df = pd.read_csv("G:/Git/304first/data/alldata/sample.csv", header=0, index_col=None, encoding="UTF-8")
    data_matrix = df.as_matrix()
    data_matrix = np.array(data_matrix)
    # 获取要预测的类别
    Classlist = data_matrix[:, 0]
    ClassSet = list(set(Classlist))

    # 对于每个要预测的类别
    Classindex = 1
    AllclassPre={}
    for Class in ClassSet:
        print("第%d个类别" % Classindex, Class)
        predict_Datalist = []  # 预测数据
        file=str(Class)+".csv"
        if file in filenames2: #中类
            predict_Datalist = predict(path + "Mid/" + file)
            #pre_Datalist=[x[0] for x in predict_Datalist]
            pre_Datalist=[]
            for i in range(len(predict_Datalist)):
                pre_Datalist.append(round(predict_Datalist[i][0]))
            AllclassPre[str(Class)] = pre_Datalist
        elif file in filenames: #大类
            predict_Datalist = predict(path + "Big/" + file)
            #pre_Datalist =[x[0] for x in predict_Datalist]
            pre_Datalist=[]
            for i in range(len(predict_Datalist)):
                pre_Datalist.append(round(predict_Datalist[i][0]))
            AllclassPre[str(Class)] = pre_Datalist
        else: #都不在
            predict_Datalist= list(np.random.randint(10,size=30))
            AllclassPre[str(Class)] = predict_Datalist
        Classindex+=1
    print ("预测结束！")
    # k=JSONEncoder().encode(AllclassPre)
    # pre=JSONDecoder().decode(k)
    # print (pre)
    print (AllclassPre)


    end=time.time()
    print("程序结束！用时:",(end-start))






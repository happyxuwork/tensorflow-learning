# -*- coding: utf-8 -*-

from __future__ import print_function

import time
import warnings
import numpy as np
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

warnings.filterwarnings("ignore")

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.split(bytes("\n",encoding="utf-8"))
    data = data[:-1]
    # print('data len:',len(data))
    # print('sequence len:',seq_len)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])  #得到长度为seq_len+1的向量，最后一个作为label

    print('result len:',len(result))
    print('result shape:',np.array(result).shape)
    #result[:1]第一行
    # print(result[:1])

    # if normalise_window:
    #     #实际上就是进行一个归一化的过程（每一行的数，除以行首的数，得到商减一）
    #     result = normalise_windows(result)

    # print(result[:1])
    print('normalise_windows result shape:',np.array(result).shape)

    result = np.array(result)

    #划分train、test
    #row = round(0.75 * result.shape[0])
    row = 86
    #row = round(result.shape[0])
    train = result[:row, :]
    #进行下一步的合理性在哪里，表示疑惑？？？？
    #np.random.shuffle(train)

    #最后一个是label？？？神马label
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[row:, :-1]
    #步长为30
    x_test2 = data[-30:]


    #步长为10
    #x_test2=data[-10:]
    #x_test2=np.reshape(x_test2, (1, 3, 1))
    y_test = result[row:, -1]
    result1=result[:,:-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    result1=np.reshape(result1,(result1.shape[0],result1.shape[1],1))

    return [x_train, y_train, x_test, y_test,result1,x_test2]


def build_model(layers):  #layers [1,50,200,1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.1))

    # model.add(LSTM(layers[3], return_sequences=False))
    # model.add(Dropout(0.1))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

#直接全部预测
def predict_point_by_point(model, data):
    predicted = model.predict(data)
    # print('predicted shape:',np.array(predicted).shape)  #(412L,1L)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

#滚动预测
def predict_sequence_full(model, data, window_size):  #data X_test
    curr_frame = data[0]  #(50L,1L)
    predicted = []
    for i in range(len(data)):
        #x = np.array([[[1],[2],[3]], [[4],[5],[6]]])  x.shape (2, 3, 1) x[0,0] = array([1])  x[:,np.newaxis,:,:].shape  (2, 1, 3, 1)
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])  #np.array(curr_frame[newaxis,:,:]).shape (1L,50L,1L)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)   #numpy.insert(arr, obj, values, axis=None)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):  #window_size = seq_len
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def plot_results(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()
    #plt.savefig(filename+'.png')

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    plt.savefig('plot_results_multiple.png')

import os
import glob
if __name__=='__main__':
    global_start_time = time.time()
    epochs  = 200
    #步长为10
    #seq_len = 10
    # 步长为30
    seq_len = 30

    print('> Loading data... ')
    #将所有的文件依次读入
    input_path = "../data/extend/"
    # sub_dirs = [x[0] for x in os.walk(input_path)]
    # is_root_dir = True
    # for sub_dir in sub_dirs:
    #     if is_root_dir:
    #         is_root_dir = False
    #         continue
    dir_name = os.path.basename(input_path)
    extensions = ['csv']
    file_list = []
    for extension in extensions:
        file_glob = os.path.join(input_path, dir_name, '*.' + extension)
        # get all the images of one folder of the input_path
        file_list.extend(glob.glob(file_glob))
    preDict={}
    for file in file_list:
        file_name = os.path.basename(os.path.splitext(file)[0])
        X_train, y_train, X_test, y_test,result ,x_test2= load_data(file, seq_len, True)

        # print('X_train shape:', X_train.shape)  # (3709L, 50L, 1L)
        # print('y_train shape:', y_train.shape)  # (3709L,)
        # print('X_test shape:', X_test.shape)  # (412L, 50L, 1L)
        # print('y_test shape:', y_test.shape)  # (412L,)
        #
        # print('> Data Loaded. Compiling...')
        model = build_model([1, 200, 300, 1])
        # model.fit(X_train,y_train,batch_size=512,nb_epoch=epochs,validation_split=0.05)
        model.fit(X_train, y_train, batch_size=30, nb_epoch=epochs, validation_split=0.05)

        # multiple_predictions = predict_sequences_multiple(model, X_test, seq_len, prediction_len=3)
        # print(multiple_predictions)
        # print('multiple_predictions shape:',np.array(multiple_predictions).shape)   #(8L,50L)
        #
        #
        #
        # full_predictions = predict_sequence_full(model, X_test, seq_len)
        # print(full_predictions)
        # print('full_predictions shape:',np.array(full_predictions).shape)    #(412L,)



        # point_by_point_predictions = predict_point_by_point(model, X_test)
        # print(point_by_point_predictions)
        # print('point_by_point_predictions shape:', np.array(point_by_point_predictions).shape)  # (412L)
        #
        # print('Training duration (s) : ', time.time() - global_start_time)

        # plot_results_multiple(multiple_predictions, y_test, 30)
        # plot_results(full_predictions,y_test,'full_predictions')
        #plot_results(point_by_point_predictions, y_test, 'point_by_point_predictions')



        # point_by_point_predictions2 = predict_point_by_point(model,result)
        # print(point_by_point_predictions2)

        #定义列表存放所有的预测结果
        #xtest2=np.reshape(x_test2,(1))
        prelist=list(x_test2)
        for i in range(30):
            block = np.array(prelist[i:i+30])
            block = np.reshape(block, (1, 30, 1))
            pre_point = predict_point_by_point(model, block)
            prelist.extend(pre_point)
        #print (prelist[-30:])
        #plot_results(prelist[-30:], y_test, 'prePOINTs')
        preDict[file_name]=prelist[-30:]


    print(preDict)
    #pre={'10': [9.1826487, 9.1366463, 9.0686808, 9.0817347, 9.0396757, 9.0457306, 9.038578, 9.0750046, 9.0800419, 9.0464315, 9.0097675, 9.0081568, 9.0070486, 9.0065737, 9.0059853, 9.0058136, 9.0056276, 9.0055408, 9.005085, 9.0044899, 9.0041237, 9.0040798, 9.0040474, 9.0040283, 9.0040092, 9.0039968, 9.0039854, 9.0039749, 9.0039635, 9.0039539], '12': [24.167387, 24.18644, 24.197863, 24.202888, 24.205734, 24.207739, 24.209568, 24.211002, 24.212111, 24.212542, 24.211395, 24.211395, 24.211399, 24.211399, 24.211399, 24.211397, 24.211397, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399, 24.211399], '11': [3.0383589, 3.1777141, 2.6217718, 2.7337379, 3.0144749, 3.0218806, 3.2699642, 3.4856572, 3.4261208, 3.089263, 3.3549054, 3.3738468, 3.3859584, 3.4401236, 3.4863236, 3.5142002, 3.5430317, 3.5569744, 3.5587544, 3.5663049, 3.5967307, 3.6105454, 3.624078, 3.6376376, 3.6485381, 3.6572931, 3.6650014, 3.6715505, 3.6777956, 3.6843979]}

    # 换转成预测的矩阵
    # all_premat = []
    # for key in preDict:
    #     for i in range(30):
    #         if i + 1 > 9:
    #             date = "201505" + str(i + 1)  # 两位数   如20150211
    #         else:
    #             date = "201505" + "0" + str(i + 1)  # 个位数   如20150204
    #         all_premat.append([key, date, preDict[key][i]])
    # #print(all_premat)
    # predf = pd.DataFrame(all_premat, index=None, columns=["编码", "日期", "销量"])
    # predf.to_csv("submit.csv", encoding="UTF-8")










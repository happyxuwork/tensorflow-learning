# -*- coding: UTF-8 -*-
'''
@author: xuqiang
Tensorflow的数据读取有三种方式：

Preloaded data: 预加载数据
Feeding: Python产生数据，再把数据喂给后端。
Reading from file: 从文件中直接读取

TF的核心是用C++写的，这样的好处是运行快，缺点是调用不灵活。
而Python恰好相反，所以结合两种语言的优势。涉及计算的核心算子和运行框架是用C++写的，
并提供API给Python。Python调用这些API，设计训练模型(Graph)，再将设计好的Graph给后端去执行。
简而言之，Python的角色是Design，C++是Run。
'''

"""
Preload method
"""
import tensorflow as tf
def Preload():
    x1 = tf.constant([2,3,4])
    x2 = tf.constant([1,2,3])
    y = tf.add(x1,x2)
    with tf.Session() as sess:
        print(sess.run(y))

"""
Feeding methon
"""
def Feed():
    x1 = tf.placeholder(tf.int16)
    x2 = tf.placeholder(tf.int16)
    y = tf.add(x1,x2)
    li1 = [5,6,7]
    li2 = [4,5,6]
    with tf.Session() as sess:
        print(sess.run(y,feed_dict={x1:li1,x2:li2}))

"""
两种方法的区别
Preload:
将数据直接内嵌到Graph中，再把Graph传入Session中运行。
当数据量比较大时，Graph的传输会遇到效率问题。
Feeding:
用占位符替代数据，待运行的时候填充数据。

Reading From File
前两种方法很方便，但是遇到大型数据的时候就会很吃力，即使是Feeding，
中间环节的增加也是不小的开销，比如数据类型转换等等。
最优的方案就是在Graph定义好文件读取的方法，让TF自己去从文件中读取数据，
并解码成可使用的样本集。
"""
#单个Reader单个样本
def SRSS():
    filenames= ['../../../data/file/A.csv','../../../data/file/B.csv','../../../data/file/C.csv']
    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
    #定义Reader
    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)
    #定义Decoder
    example,label = tf.decode_csv(value,record_defaults=[['null'],['null']])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)#启动QuereRunner
        for i in range(10):
            print(example.eval())
        coord.request_stop()
        coord.join(threads)


#单个Reader多个样本
def SRMS():
    filenames= ['../../../data/file/A.csv','../../../data/file/B.csv','../../../data/file/C.csv']
    filename_queue = tf.train.string_input_producer(filenames,shuffle=False)
    #定义Reader
    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)
    #定义Decoder
    example,label = tf.decode_csv(value,record_defaults=[['null'],['null']])
    #使用一个tf.train.batch()会多加一个样本队列和一个QueueRunner,Decoder
    #会解析这个队列，再批量出队
    #虽然只有一个Reader,但是可以设置多个线程，相应增加线程会提高读取速度，但非越多越好
    example_batch,label_batch = tf.train.batch([example,label],batch_size=5)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()#创建一个协调器，管理线程
        threads = tf.train.start_queue_runners(coord=coord)#启动QuereRunner
        for i in range(10):
            print(example_batch.eval())#如何拿到一个tensor的值，使用.eval()即可
        coord.request_stop()
        coord.join(threads)

#多个Reader多个样本
def MRMS():
    filenames = ['../../../data/file/A.csv', '../../../data/file/B.csv', '../../../data/file/C.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    # 定义Reader
    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)
    #record_defaults = [['null'],['null']]
    #设置两个reader
    example_list = [tf.decode_csv(value,record_defaults=[['null'],['null']]) for _ in range(2)]
    example_batch,label_batch = tf.train.batch_join(example_list,batch_size=5)
    with tf.Session() as sess:
        #tf.initialize_local_variables()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(5):
            print(example_batch.eval())
        coord.request_stop()
        coord.join(threads)

"""
tf.train.batch与tf.train.shuffle_batch函数是单个Reader读取的，但是可以多线程
tf.train.batch_join与tf.train.shuffle_batch_join可设置多个Reader读取，每个
Reader使用一个线程，单个Reader时，2个线程达到速度的极限，多个Reader时，两个
Reader达到极限

"""

def iterateControll():
    filenames = ['../../../data/file/A.csv', '../../../data/file/B.csv', '../../../data/file/C.csv']
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False,num_epochs=3)#设置迭代次数
    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)
    record_defaults = [['null'],['null']]
    example_list = [tf.decode_csv(value,record_defaults=record_defaults) for _ in range(2)]
    example_batch,label_batch = tf.train.batch_join(example_list,batch_size=5)
    init_local_op = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_local_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                print(example_batch.eval())
        except tf.errors.OutOfRangeError:
            print("Epochs Complete!")
        finally:
            coord.request_stop()
        coord.join(threads)
        coord.request_stop()
        coord.join(threads)







if __name__ == "__main__":
    #Preload()
    #Feed()
    #SRSS()
    #SRMS()
    #MRMS()
    iterateControll()
    # example_list = [1 for _ in range(2)]
    # print(example_list)

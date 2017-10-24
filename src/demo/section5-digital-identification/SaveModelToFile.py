# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import tensorflow as tf

'''
持久化的使用
'''
# v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')
# v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')
#
# resule = v1 + v2
# init_op = tf.global_variables_initializer()
#
# #声明tf.train.Saver类用于保存模型
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess,"E:/tensorflow/model.ckpt")
# model.ckpt.meta--保存了计算图的结构
# model.ckpe--保存了tensorflow程序中的每一个变量的取值
# checkpoint--保存了一个目录下所有的模型文件列表

'''
restore the model from the disk
'''
# # load the persistence graph
# saver = tf.train.import_meta_graph("E:/tensorflow/model.ckpt.meta")
# with tf.Session() as sess:
#     saver.restore(sess,"E:/tensorflow/model.ckpt")
#     # get the name of tersor by the tensor name
#     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


'''
introduce movingaverage to the persistence model
'''
# v = tf.Variable(0,dtype=tf.float32,name="v")
# for variables in tf.global_variables():
#     print(variables.name)
# ema = tf.train.ExponentialMovingAverage(0.99)
# maintain_averages_op = ema.apply(tf.global_variables())
# for variables in tf.global_variables():
#     print(variables.name)
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run([v, ema.average(v)]))
#
#     sess.run(tf.assign(v,10))
#     sess.run(maintain_averages_op)
#     saver.save(sess,"E:/tensorflow/savemodel1/model.ckpt")
#     print(sess.run([v,ema.average(v)]))

'''
rename the variable name from the store
'''
# v = tf.Variable(0,dtype=tf.float32,name="v")
# saver = tf.train.Saver({"v/ExponentialMovingAverage":v})
# with tf.Session() as sess:
#     saver.restore(sess,"E:/tensorflow/savemodel1/model.ckpt")
#     print(sess.run(v))

'''
as we can see above 
it is difficult to rename the variable name from the store
however the tf.train.ExponentialMovingAverage provide the variables_to_restore 
to save the problem 
'''
# v = tf.Variable(0,dtype=tf.float32,name="v")
# ema = tf.train.ExponentialMovingAverage(0.99)
# print(ema.variables_to_restore())
# saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     saver.restore(sess,"E:/tensorflow/savemodel1/model.ckpt")
#     print(sess.run(v))

''''
stroe part variables to disk 
'''
# from tensorflow.python.framework import graph_util
# v1 = tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
# v2 = tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
# result = v1 + v2
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     #到处当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
#     graph_def = tf.get_default_graph().as_graph_def()
#     output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add'])
#     with tf.gfile.GFile("E:/tensorflow/savemodel2/combined_model.pb","wb") as f:
#         f.write(output_graph_def.SerializeToString())

'''
直接使用上述的加法的运算
'''
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "E:/tensorflow/savemodel2/combined_model.pb"
    # 读取保存的文件，并将其解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        result = tf.import_graph_def(graph_def, return_elements=["add:0"])
        print(sess.run(result))

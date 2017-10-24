# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/path/to/MNIST_data", one_hot=True)
print(type(mnist))
print("Training data size:", (mnist.train.num_examples))
print("Validating data size:", (mnist.validation.num_examples))
print("Testing data size:", (mnist.test.num_examples))

print("Example of Training data:", (mnist.train.images[0]))
print("Label of Training data:", (mnist.train.labels[0]))

# in order to convient to use the stocastic gradient decent
# the function input_data.read_data_set function provide a mnist.train.next_batch function to get a small batch size for specific dataset

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print("X shape:", xs.shape)
print("Y shape:", ys.shape)

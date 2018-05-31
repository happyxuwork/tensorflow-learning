# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys
import pprint

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
def get_inception_score(sess,images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    # print(img.shape)
    inps.append(np.expand_dims(img, 0))
  bs = 1
  # with tf.Session() as sess:
  preds = []
  n_batches = int(math.ceil(float(len(inps)) / float(bs)))
  for i in range(n_batches):
        # sys.stdout.write(".")
        # 用于刷新输出的缓存，只要由输出，马上输出
        # sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        # pprint.pprint(pred)
        preds.append(pred)
  preds = np.concatenate(preds, 0)
  scores = []
    # scoresnew= []
  for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      # py = np.mean(part, axis=0)
      # l = np.mean([entropy(part[i, :], py) for i in range(part.shape[0])])
      # np.mean([np.exp(np.mean(x)) for x in split_data])  # 1608.25
      scores.append(np.exp(kl))
      # scoresnew.append(kl)
  return np.mean(scores), np.std(scores)

# This function is called automatically.
def _init_inception(sess,inception_path):
  global softmax
  with tf.gfile.FastGFile(inception_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size.
  # with tf.Session() as sess:
  pool3 = sess.graph.get_tensor_by_name('pool_3:0')
  ops = pool3.graph.get_operations()
  for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
  w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
  logits = tf.matmul(tf.squeeze(pool3,[1,2]), w)
  softmax = tf.nn.softmax(logits)

def get_images(filename):
        return scipy.misc.imread(filename)


def main(sess,inception_path,generate_image_path):
    # 初始化graph
    print("initial the graph...")
    _init_inception(sess,inception_path)


    sub_dirs = [x[1] for x in os.walk(generate_image_path)]
    # print(sub_dirs[0])
    sub_dirs[0].remove('realimg')
    # print(sub_dirs[0])
    # npz_subs_dirs = [x[2] for x in os.walk(input_data_tensor_valuation_path)]
    print("caculate the Inception Score for each folder...")
    arr_len = len(sub_dirs[0])
    result_val_arr = np.zeros(arr_len)
    result_winer_arr = np.zeros(arr_len)
    for sub_dir,i in zip(sub_dirs[0],range(arr_len)):
        images_path = os.path.join(generate_image_path,sub_dir)
        filenames = glob.glob(os.path.join(images_path, '*.*'))
        images = [get_images(filename) for filename in filenames]
        mean, std = get_inception_score(images)
        result_val_arr[i] = mean
    max_index = np.where(np.max(result_val_arr)== result_val_arr)
    result_winer_arr[max_index] = 1
    print("Inception Score caculated done...")
    # print(sub_dirs[0])
    return result_val_arr,result_winer_arr







if __name__=='__main__':
    if softmax is None:
      _init_inception()
    filenames = glob.glob(os.path.join('./2', '*.*'))
    images = [get_images(filename) for filename in filenames]
    print(len(images))
    print(get_inception_score(images))

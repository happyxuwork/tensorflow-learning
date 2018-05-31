# -*- coding: UTF-8 -*-
'''
@author: xuqiang
'''
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from scipy import linalg
import pathlib
# a = [[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]]
# b = [[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7]]
# c = [[ 0,  1,  2,  3]]
#
# # 将a保存到a.npz文件中，并将a取名为ar
# # np.savez("1.npz",ar=a,br=b,cr=c)
# c = np.load("fid_stats_cifar10_train.npz")
# for i in range(c['mu'].__len__()):
#        print(c['mu'][i])
# print(c['sigma'].shape)
# # print(c['a'])
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              o._shape = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------



def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------



def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')

def savemAndsTonpzFile(inception_path,images_path):
       create_inception_graph(str(inception_path))
       with tf.Session() as sess:
              # path = "F:/研究生/图像数据/数据/其它/CIFAR10/out/train/"
              path = pathlib.Path(images_path)
              files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
              x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
              m, s = calculate_activation_statistics(x, sess)
              print(m+" "+s)
              np.savez("cifar10.npz",mu=m,sigma=s)

if __name__ == "__main__":
       i_path= "G:/classify_image_graph_def.pb"
       images_path="F:/研究生/图像数据/数据/其它/CIFAR10/out/train/"
       # images_path="F:/研究生/图像数据/数据/其它/CIFAR10/out/mode1_generate/"----因为这里面图像的大小不一致，所以会出错
       savemAndsTonpzFile(i_path,images_path)
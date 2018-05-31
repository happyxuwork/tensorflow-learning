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


# this is difficult for understand!!!!!!!!!!
def _get_inception_layer(sess):
    '''prepares inception net for batched usage and returns pool_3 layer'''
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    print("pool3" + str(pool3))
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


# def _get_inception_layer2(sess):
#     '''prepares inception net for batched usage and returns pool_3 layer'''
#     layername2 = 'FID_Inception_Net/pool_3/_reshape:0'
#     # layername2 = 'FID_Inception_Net/pool_3:1'
#     pool3_reshape = sess.graph.get_tensor_by_name(layername2)
#     # xx = sess.graph.get_tensor_by_name("FID_Inception_Net/softmax/logits/MatMul").inputs[1]
#     xx =  sess.graph.get_operation_by_name("FID_Inception_Net/pool_3/_reshape").inputs[1]
#     print("xx"+str(xx))
#
#     ops = pool3_reshape.graph.get_operations()
#     for op_idx, op in enumerate(ops):
#         for o in op.outputs:
#             shape = o.get_shape()
#             if shape._dims is not None:
#                 shape = [s.value for s in shape]
#                 new_shape = []
#                 for j, s in enumerate(shape):
#                     if s == 1 and j == 0:
#                         new_shape.append(None)
#                     else:
#                         new_shape.append(s)
#                 o._shape = tf.TensorShape(new_shape)
#     return pool3_reshape,xx

def get_activations(images, sess, batch_size=50, verbose=False):
    '''
    Caculates the activations of the pool_3 layer for all images
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
    '''
    # inception_layer = _get_inception_layer(sess)
    pool3 = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size
    pred_arr = np.empty((n_used_imgs, 2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i + 1, n_batches), end="", flush=True)
        start = i * batch_size
        end = start + batch_size
        batch = images[start:end]
        # pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred = sess.run(pool3, {'FID_Inception_Net/ExpandDims:0': batch})
        # pred_reshape = sess.run(pool3_reshape, {'FID_Inception_Net/pool_3:0': pred.reshape(batch_size,1,-1,2048)})
        # kk = tf.matmul(tf.squeeze(pred, [1, 2]), tf.cast(xx,tf.float32))
        # print(kk)
        # pred_reshape = sess.run(pool3_reshape, {'FID_Inception_Net/ExpandDims:0': batch})
        # print(pred_reshape)
        pred_arr[start:end] = pred.reshape(batch_size, -1)
    if verbose:
        print(" done")
    return pred_arr


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    '''Calculate of the statistics used by the FID
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
    '''
    act, _ = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# create the inception grap by exit trained
def create_inception_graph(path):
    '''
    :param path:
        trained mode path
    :return:
        the trained mode`s struct
    '''
    with tf.gfile.FastGFile(path, 'rb') as f:
        # define the default graph
        # graph_def = tf.GraphDef()
        graph_def = tf.GraphDef()
        # import the strucet of net form f
        # graph_def.ParseFromString(f.read())
        graph_def.ParseFromString(f.read())
        # rename the nee to name='XXX'
        # _ = tf.import_graph_def(graph_def,name='FID_Inception_Net')
        _ = tf.import_graph_def(graph_def, name='FID_Inception_Net')


def saveImgToNPZFile(sess, inception_path, image_fake_path, image_all_to_npz_path):
    '''
    :param inception_path:
        trained mode path--inception
    :param image_real_path:
        the real image path of sample
    :param image_fake_path:
        --list
        the fake image path of generator
    :return:
    '''
    if not os.path.exists(image_all_to_npz_path):
        os.makedirs(image_all_to_npz_path)
    create_inception_graph(str(inception_path))
    image_paths = image_fake_path
    # with tf.Session() as sess:
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        files = list(path.glob('*,jpg')) + list(path.glob('*.png'))
        # print(path)
        base_name = os.path.basename(image_path)
        extensions = ['jpg', 'png']
        file_list = []
        for extension in extensions:
            file_glob = os.path.join(image_path, '*.' + extension)
            # get all the images of one folder of the input_path
            file_list.extend(glob.glob(file_glob))
            # files = list(path.glob('*,jpg'))+list(path.glob('*.png'))
        if os.path.exists(os.path.join(image_all_to_npz_path, base_name + ".npz")):
            continue
        else:
            x = np.array([imread(str(fn)).astype(np.float32) for fn in file_list])
            m, s = calculate_activation_statistics(x, sess)
            # file_path = os.path.join(image_all_to_npz_path,base_name)
            file_name = os.path.join(image_all_to_npz_path, base_name + ".npz")
            np.savez(file_name, mu=m, sigma=s)


# -------------------------------------------------------------------------------
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        # warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# -------------------------------------------------------------------------------


def _handle_path(path):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    return m, s


def caculate_img_fid(sess, inception_path, generate_image_path, image_all_to_npz_path):
    sub_dirs = [x[0] for x in os.walk(generate_image_path)]
    del sub_dirs[0]
    # print(sub_dirs)
    image_fake_path = sub_dirs
    # 首先将所有的图像转成.npz文件保存（主要是计算图像的均值和协方差）
    print("convert the imgs to .pnz file....")
    saveImgToNPZFile(sess, inception_path, image_fake_path, image_all_to_npz_path)
    print("convert to .npz done...")

    npz_subs_dirs = [x[2] for x in os.walk(image_all_to_npz_path)]
    npz_sub_dirs = npz_subs_dirs[0]

    realimg_path = os.path.join(image_all_to_npz_path, 'realimg.npz')
    rm, rs = _handle_path(realimg_path)
    # 在目录中移除realimg
    npz_sub_dirs.remove('realimg.npz')
    len_sub_dirs = len(npz_sub_dirs)
    result = np.zeros(len_sub_dirs)
    return_result = np.zeros(len_sub_dirs)
    # print(type(result))
    print("caculate the imgs fid value...")
    for sub_dir, i in zip(npz_sub_dirs, range(len_sub_dirs)):
        fake_npz_path = os.path.join(image_all_to_npz_path, sub_dir)
        fm, fs = _handle_path(fake_npz_path)
        fid_val = calculate_frechet_distance(rm, rs, fm, fs)
        result[i] = fid_val
    min_value = max(result)
    max_index = np.where(result == min_value)
    return_result[max_index] = 1
    print("fid value caculate done...")
    return result, return_result


inception_path = "G:/classify_image_graph_def.pb"
image_all_to_npz_path = "F:/alltoNPZ/"
generate_image_path = "F:/train/"
image_all_to_tensor_path = "F:/alltoTensor"
if __name__ == "__main__":
    # ----------------------计算FID分数--------------------------------
    # 注意
    # 所有的图像文件放在同一个文件夹底下：包括真实的图形，多个生成的图像（用不同的文件夹区分）
    # 真实的图像训练集名称固定为realimg
    # 还有重要的一点就是每个文件夹下的图像大小要一致

    # sub_dirs = [x[0] for x in os.walk(generate_image_path)]
    # del sub_dirs[0]
    # print(sub_dirs)
    # image_fake_path = sub_dirs
    # saveImgToNPZFile(inception_path,image_fake_path,image_all_to_npz_path)
    # npz_sub_dirs = [x[2] for x in os.walk(image_all_to_npz_path)]
    # 得到生成images相对与训练集的fid值，fid值越小，质量越高
    with tf.Session() as sess:
        # fid_val,fid_winer_arr = caculate_img_fid(sess,inception_path,generate_image_path,image_all_to_npz_path)
        fid_val, fid_winer_arr = fid.caculate_img_fid(sess, inception_path, generate_image_path, image_all_to_npz_path)
        print(fid_val)
        print(fid_winer_arr)
        # ----------------------计算FID分数--------------------------------

        # ----------------------计算分类准确性--------------------------------
        # 分类准确性度量：
        # 第一步：将所有的图像转成BacknetTensor
        # 第二步：将得到的tensor送入到训练好的模型中，
        # 取分类的平均准确性指标
        # acc_val, acc_winer_arr = caculate_img_clasify_acc(sess,inception_path,generate_image_path,image_all_to_tensor_path)

        # ----------------------计算分类准确性--------------------------------



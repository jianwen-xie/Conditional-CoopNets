'''
From https://github.com/tsc2017/Frechet-Inception-Distance
Code derived from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
Usage:
    Call get_fid(images1, images2)
Args:
    images1, images2: Numpy arrays with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. 
    dtype of the images is recommended to be np.uint8 to save CPU memory.
Returns:
    Frechet Inception Distance between the two image distributions.
'''

import tensorflow as tf
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
print(tf.__version__)
if float('.'.join(tf.__version__.split('.')[:2])) < 1.15:
    tfgan = tf.contrib.gan
else:
    import tensorflow_gan as tfgan
from scipy import linalg
import warnings

# session=tf.compat.v1.InteractiveSession()
# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

# Run images through Inception.
inception_images = None
activations1 = None
activations2 = None

activations = None
fcd = None

def init_fid():
    global inception_images, activations1, activations2, activations, fcd
    inception_images = tf.placeholder(tf.float32, [None, None, None, 3])
    activations1 = tf.placeholder(tf.float32, [None, None], name = 'activations1')
    activations2 = tf.placeholder(tf.float32, [None, None], name = 'activations2')
    activations =inception_activations(inception_images)
    # fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

def calculate_activation_statistics(sess, images):
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
    act = get_inception_activations(sess, images)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


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
        warnings.warn(msg)
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


def inception_activations(images = inception_images, num_splits = 1):
    # images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = tf.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'pool_3:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 8,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations

# activations =inception_activations()

def get_inception_activations(sess, inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    act = np.zeros([inps.shape[0], 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE : (i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = sess.run(activations, feed_dict = {inception_images: inp})
    return act

def activations2distance(sess, act1, act2):
     return sess.run(fcd, feed_dict = {activations1: act1, activations2: act2})
        
def preprocess_images(images):
    new_images = []
    for idx, im in enumerate(images):
        im = np.clip((im + 1) * 127.5, 0, 255).astype(np.uint8)
        if im.shape[-1] == 1:
            im = np.tile(im, [1, 1, 3])
        new_images.append(im)
    return np.stack(new_images)

def get_fid(sess, images1, images2):
    assert(type(images1) == np.ndarray)
    assert(len(images1.shape) == 4)
    # assert(images1.shape[1] == 3)
    images1 = preprocess_images(images1)
    assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(type(images2) == np.ndarray)
    assert(len(images2.shape) == 4)
    # assert(images2.shape[1] == 3)
    images2 = preprocess_images(images2)
    assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    # print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()

    mu1, sigma1 = calculate_activation_statistics(sess, images1)
    mu2, sigma2 = calculate_activation_statistics(sess, images2)
    # print(act1.shape, act2.shape)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    # print('FID calculation time: %f s' % (time.time() - start_time))
    return fid
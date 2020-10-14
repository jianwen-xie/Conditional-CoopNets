from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from model_mnist import CoopNets
from data import *
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_integer('image_size', 28, 'Image size to rescale images')
tf.flags.DEFINE_integer('batch_size', 300, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 3000, 'Number of epochs to train')
tf.flags.DEFINE_integer('num_images', None, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 30, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 30, 'Column number of synthesized images')
tf.flags.DEFINE_integer('gpu_id', 1, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1', 0.5, 'Momentum term of adam')

# parameters for descriptorNet
tf.flags.DEFINE_float('d_lr', 0.0008, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 16, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.0008, 'Step size for descriptor Langevin dynamics') #0.0008

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0001, 'Initial learning rate for generator')
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 16, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')

# parameters for conditioned data
tf.flags.DEFINE_integer('attr_size', 10, 'Size of attributes')
tf.flags.DEFINE_integer('z_dim', 128, 'Size of attributes') #10

tf.flags.DEFINE_string('category', 'fashion_MNIST', 'MNIST or fashion_MNIST')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 10, 'Number of epochs to save output results')
tf.flags.DEFINE_integer('eval_step', 10, 'Number of epochs to save output results')
tf.flags.DEFINE_boolean('eval_fid_d', False, 'True when reporting FID for D')
tf.flags.DEFINE_boolean('eval_fid_g', False, 'True when reporting FID for G')

tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_boolean('debug', False, 'True if in testing mode')
tf.flags.DEFINE_string('ckpt', '/home/kenny/extend/PAMI_revision_condCoopNets/mnist_code/output/MNIST_2020-09-06_10-22-41_step_16_z_128/checkpoints/model.ckpt-1490', 'Checkpoint path to load')


def main(_):
    category = FLAGS.category

    RANDOM_SEED = 231
    np.random.seed(RANDOM_SEED)
    tf.set_random_seed(RANDOM_SEED)

    if FLAGS.category == 'MNIST':
        dataset_path = './dataset/mnist'
    elif FLAGS.category == 'fashion_MNIST':
        dataset_path = './dataset/fashion_mnist'
    else:
        raise NotImplementedError("wrong mnist type")

    postfix = time.strftime('%Y-%m-%d_%H-%M-%S') if not FLAGS.debug else 'debug'
    if FLAGS.debug:
        FLAGS.eval_step = 1
        FLAGS.log_step = 1
        FLAGS.num_images = 10000
    output_dir = os.path.join(FLAGS.output_dir, '{}_{}'.format(category, postfix))
    sample_dir = os.path.join(output_dir, 'synthesis')
    log_dir = os.path.join(output_dir, 'log')
    plots_dir = os.path.join(output_dir, 'plots')
    model_dir = os.path.join(output_dir, 'checkpoints')
    test_dir = os.path.join(output_dir, 'test')

    model = CoopNets(
        dataset_path=dataset_path,
        sample_dir=sample_dir, log_dir=log_dir, model_dir=model_dir, plots_dir=plots_dir, test_dir=test_dir
    )

    #ckpt = '%s/checkpoints/model.ckpt-%s' % (FLAGS.output_dir, FLAGS.ckpt)
    print(FLAGS.ckpt)
    # test = False
    # ckpt = './output/MNIST/checkpoints/model.ckpt-1260'
    gpu_options = tf.GPUOptions(visible_device_list=str(FLAGS.gpu_id), allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        if FLAGS.test:
            tf.gfile.MakeDirs(test_dir)

            model.test_generation(sess, FLAGS.ckpt)
        else:
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)
            tf.gfile.MakeDirs(sample_dir)
            tf.gfile.MakeDirs(model_dir)
            model.train(sess)


if __name__ == '__main__':
    tf.app.run()

from __future__ import division
from __future__ import print_function

import os
import time
import math
import numpy as np
from six.moves import xrange

from modules import *
from utils.custom_ops import *
from utils.data_io import saveSampleImages, labels_to_one_hot
import utils.plot
from scipy import io
from data import *
import fid_util2 as fid_util

FLAGS = tf.app.flags.FLAGS


class CoopNets(object):
    def __init__(self,
                 dataset_path=None,
                 sample_dir='./synthesis',
                 model_dir='./checkpoints',
                 log_dir='./log',
                 plots_dir='./plots',
                 test_dir='./test'):

        # self.type = FLAGS.net_type
        self.num_epochs = FLAGS.num_epochs
        self.batch_size = FLAGS.batch_size
        self.image_size = FLAGS.image_size
        self.nTileRow = FLAGS.nTileRow
        self.nTileCol = FLAGS.nTileCol
        self.num_chain = self.nTileRow * self.nTileCol
        self.beta1 = FLAGS.beta1

        self.d_lr = FLAGS.d_lr
        self.g_lr = FLAGS.g_lr
        self.delta1 = FLAGS.des_step_size
        self.sigma1 = FLAGS.des_refsig
        self.delta2 = FLAGS.gen_step_size
        self.sigma2 = FLAGS.gen_refsig
        self.t1 = FLAGS.des_sample_steps
        self.t2 = FLAGS.gen_sample_steps

        # self.code_size = code_size

        self.dataset_path = dataset_path

        self.log_step = FLAGS.log_step

        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.plots_dir = plots_dir
        self.test_dir = test_dir

        self.generator = generator_mnist
        self.descriptor = descriptor_attr2img_mnist

        self.attr_size = FLAGS.attr_size
        self.z_dim = FLAGS.z_dim

        assert FLAGS.category == 'MNIST' or FLAGS.category == 'fashion_MNIST'

        if FLAGS.category == 'MNIST':
            self.mnist_type = 1
        elif FLAGS.category == 'fashion_MNIST':
            self.mnist_type = 2


        self.grayscale = True
        if self.grayscale:
            self.channel = 1
        else:
            self.channel = 3

        self.syn = tf.placeholder(
            shape=[None, self.image_size, self.image_size, self.channel], dtype=tf.float32, name='syn')
        self.obs = tf.placeholder(
            shape=[None, self.image_size, self.image_size, self.channel], dtype=tf.float32, name='obs')
        self.attr = tf.placeholder(shape=[None, self.attr_size], dtype=tf.float32,
                                   name='attr')
        self.z = tf.placeholder(
            shape=[None, self.z_dim], dtype=tf.float32, name='z')

        self.lambda_L1 = 0

        self.verbose = False

        # np.random.seed(1)
        tf.set_random_seed(1)

    def build_model(self):

        global_step = tf.Variable(-1, trainable=False)
        self.decayed_learning_rate_D = tf.train.exponential_decay(self.d_lr, global_step, 100, 0.96, staircase=True)
        # self.decayed_learning_rate_d = self.d_lr
        self.decayed_learning_rate_G = tf.train.exponential_decay(self.g_lr, global_step, 100, 0.96, staircase=True)
        # self.decayed_learning_rate_vae = self.vae_lr
        self.update_lr = tf.assign_add(global_step, 1)


        self.gen_res = self.generator(self.attr, self.z)

        des_net = {}

        obs_res = self.descriptor(self.obs, self.attr, reuse=False)
        syn_res = self.descriptor(self.syn, self.attr, reuse=True)

        with open("%s/config.txt" % self.sample_dir, "w") as f:
            for k in self.__dict__:
                f.write(str(k) + ':' + str(self.__dict__[k]) + '\n')
            # f.write('\ndescriptor:\n')
            # for layer in sorted(des_net):
            #     f.write(str(layer) + ':' + str(des_net[layer]) + '\n')

        self.recon_err = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0)), 2))
        # self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean(self.recon_err)

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables()
                    if var.name.startswith('des')]

        self.des_loss = tf.reduce_sum(tf.subtract(tf.reduce_mean(
            syn_res, axis=0), tf.reduce_mean(obs_res, axis=0)))
        # self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(self.des_loss)

        des_optim = tf.train.AdamOptimizer(self.decayed_learning_rate_D, beta1=self.beta1)
        des_grads_vars = des_optim.compute_gradients(
            self.des_loss, var_list=des_vars)
        # des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients(des_grads_vars)

        # generator variables
        gen_vars = [var for var in tf.trainable_variables()
                    if var.name.startswith('gen')]

        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - self.gen_res)
                                       + self.lambda_L1 * tf.abs(self.syn - self.gen_res))

        # self.gen_loss = tf.reduce_sum(tf.square(self.obs - self.gen_res))
        # + tf.reduce_mean(tf.abs(self.syn - self.gen_res))

        # self.gen_loss_mean, self.gen_loss_update = tf.contrib.metrics.streaming_mean(self.gen_loss)

        gen_optim = tf.train.AdamOptimizer(self.decayed_learning_rate_G, beta1=self.beta1)
        gen_grads_vars = gen_optim.compute_gradients(
            self.gen_loss, var_list=gen_vars)
        # gen_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in gen_grads_vars if '/w' in var.name]
        self.apply_g_grads = gen_optim.apply_gradients(gen_grads_vars)

        # symbolic langevins
        self.langevin_conditional_descriptor = self.langevin_dynamics_conditional_descriptor(
            self.syn, self.attr, use_noise=False)
        self.langevin_conditional_descriptor_noise = self.langevin_dynamics_conditional_descriptor(
            self.syn, self.attr, use_noise=True)

        # self.langevin_conditional_generator = self.langevin_dynamics_conditional_generator(self.z, self.condition)

    def langevin_dynamics_conditional_descriptor(self, syn_arg, attr_arg, use_noise=False):
        def cond(i, syn, attr):
            return tf.less(i, self.t1)

        def body(i, syn, attr):
            noise = tf.random_normal(shape=tf.shape(syn), name='noise')
            syn_res = self.descriptor(syn, attr, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.delta1 * self.delta1 * \
                (syn / self.sigma1 / self.sigma1 - grad)
            if use_noise:
                syn = syn + self.delta1 * noise
            return tf.add(i, 1), syn, attr

        with tf.name_scope("langevin_dynamics_descriptor"):
            i = tf.constant(0)
            i, syn, condition = tf.while_loop(
                cond, body, [i, syn_arg, attr_arg])
            return syn

    def train(self, sess):
        self.build_model()

        train_data = MNISTDataSet(self.dataset_path, self.mnist_type, train=True, num_images=FLAGS.num_images,
                                  img_width=self.image_size, img_height=self.image_size, shuffle=True,
                                  low=-1, high=1)
        test_inds = np.random.choice(len(train_data), 10000)
        test_attr = train_data.attributes[test_inds]

        if FLAGS.eval_fid_d or FLAGS.eval_fid_g:
            fid_util.init_fid()
            fid_log_file = os.path.join(self.log_dir, 'fid.txt')


        # Prepare training data
        # with open("%s/data_info.json" % self.sample_dir, "w") as f:
        #     f.write(str(train_data))

        self.test_dir = os.path.join(self.sample_dir, 'test')

        num_batches = int(math.ceil(len(train_data) / self.batch_size))


        # initialize training
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        # sample_results = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, 3)

        saver = tf.train.Saver(max_to_keep=15)

        writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        tf.get_default_graph().finalize()

        # make graph immutable
        # tf.get_default_graph().finalize()

        sample_results = np.random.randn(
            train_data.num_images, train_data.img_height, train_data.img_width, self.channel)
        gen_results = np.random.randn(
            train_data.num_images, train_data.img_height, train_data.img_width, self.channel)
        saveSampleImages(train_data.images[test_inds], "%s/ref_target.png" % self.sample_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)

        img_summary = image_summary('ref_target', train_data.images[test_inds], row_num=self.nTileRow,
                                    col_num=self.nTileCol)

        writer.add_summary(img_summary)
        writer.flush()

        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
        utils.plot.log_dir = self.plots_dir
        # train
        for epoch in range(0, self.num_epochs):
            start_time = time.time()
            sess.run(self.update_lr)

            log = dict()
            d_loss_epoch, g_loss_epoch, mse_epoch = [], [], []
            for i in range(num_batches):

                index = slice(i * self.batch_size,
                              min(len(train_data), (i + 1) * self.batch_size))
                obs_img, attr = train_data[index]

                # Step G0: generate X ~ N(0, 1)
                z_vec = np.random.normal(
                    size=(len(obs_img), self.z_dim), scale=1.0)
                g_res = sess.run(self.gen_res, feed_dict={
                                 self.attr: attr, self.z: z_vec})

                # fc = sess.run(self.fc, feed_dict={self.z: z_vec, self.condition: src_img})
                # print(fc.max())
                # Step D1: obtain synthesized images Y
                if epoch < 100:

                    syn = sess.run(self.langevin_conditional_descriptor_noise,
                                   feed_dict={self.syn: g_res, self.attr: attr})
                else:
                    syn = sess.run(self.langevin_conditional_descriptor,
                                   feed_dict={self.syn: g_res, self.attr: attr})

                # Step G1: update X using Y as training image
                # if self.t2 > 0:
                #     z_vec = sess.run(self.langevin_conditional_generator, feed_dict={self.z: z_vec, self.obs: syn, self.condition: src_img})
                # Step D2: update D net
                d_loss = sess.run([self.des_loss, self.apply_d_grads],
                                  feed_dict={self.obs: obs_img, self.syn: syn, self.attr: attr})[0]
                # Step G2: update G net
                g_loss = sess.run([self.gen_loss, self.apply_g_grads],
                                  feed_dict={self.obs: syn, self.attr: attr, self.z: z_vec, self.syn: obs_img})[0]

                # Compute MSE
                mse = sess.run(self.recon_err, feed_dict={
                               self.obs: obs_img, self.syn: syn})
                d_loss_epoch.append(d_loss)
                g_loss_epoch.append(g_loss)
                mse_epoch.append(mse)

                sample_results[index] = syn
                gen_results[index] = g_res
                if self.verbose:
                    print('Epoch #{:d}, [{:2d}]/[{:2d}], descriptor loss: {:.4f}, generator loss: {:.4f}, '
                          'L2 distance: {:4.4f}'.format(epoch, i + 1, num_batches, d_loss.mean(), g_loss.mean(), mse))

                # if i == 0 and epoch == 0:
                #     saveSampleImages(obs_img, "%s/ref_target000.png" % self.sample_dir,
                #                      row_num=2, col_num=self.nTileCol)
                #
                #     with open('%s/data_info000.txt' % self.sample_dir, 'w') as f:
                #         f.write(str(attr))

            end_time = time.time()

            [decayed_lr_D, decayed_lr_G] = sess.run([self.decayed_learning_rate_D, self.decayed_learning_rate_G])

            log['d_loss_avg'], log['g_loss_avg'], log['mse_avg'] = np.mean(
                d_loss_epoch), np.mean(g_loss_epoch), np.mean(mse_epoch)

            print('Epoch #{:d}, avg.descriptor loss: {:.4f}, avg.generator loss: {:.4f}, avg.L2 distance: {:4.4f}, '
                  'time: {:.2f}s, learning rate: EBM {:.6f}, generator {:.6f}'.format(epoch, log['d_loss_avg'],
                  log['g_loss_avg'], log['mse_avg'], end_time - start_time, decayed_lr_D, decayed_lr_G))

            if np.isnan(log['mse_avg']) or log['mse_avg'] > 2:
                break

            for tag, value in log.items():
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag=tag, simple_value=value)])
                writer.add_summary(summary, epoch)

            if epoch % self.log_step == 0:
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" %
                           (self.model_dir, 'model.ckpt'), global_step=epoch)

                img_summary = image_summary('sample', sample_results, row_num=self.nTileRow,
                                            col_num=self.nTileCol)

                writer.add_summary(img_summary, epoch)
                writer.flush()


                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleImages(sample_results, "%s/des%03d.png" % (self.sample_dir, epoch), row_num=self.nTileRow,
                                 col_num=self.nTileCol)
                utils.plot.plot('%s/time' % self.plots_dir, end_time - start_time)

            if epoch % FLAGS.eval_step == 0:

                sample_des, sample_gen = self.val_mnist(sess, epoch, test_attr)

                if FLAGS.eval_fid_d or FLAGS.eval_fid_g:
                    print('Evaluating FID on Epoch {}'.format(epoch))
                    rand_ind = np.random.choice(len(sample_des), 10000)
                    message = 'Epoch #{:d}, '.format(epoch)

                    if FLAGS.eval_fid_d:
                        eval_data_des = sample_des[rand_ind]
                        fid_des = fid_util.get_fid(sess, eval_data_des, train_data.images[np.random.choice(len(train_data), len(eval_data_des))])
                        message = message + ' FID of slow-thinking = : {:.2f}'.format(fid_des)
                        utils.plot.plot('%s/fid des' % self.plots_dir, fid_des)

                    if FLAGS.eval_fid_g:
                        eval_data_gen = sample_gen[rand_ind]
                        fid_gen = fid_util.get_fid(sess, eval_data_gen, train_data.images[np.random.choice(len(train_data), len(eval_data_gen))])
                        message = message + ' FID of fast-thinking = : {:.2f}'.format(fid_gen)
                        utils.plot.plot('%s/fid gen' % self.plots_dir, fid_gen)

                    print(message)
                    fo = open(fid_log_file, 'a')
                    fo.write(message + "\n")
                    fo.close()

                utils.plot.flush()
            utils.plot.tick()
                # if FLAGS.eval_fid:


    def val_mnist(self, sess, epoch, test_attr=None):

        # labels = np.random.random_integers(0, 9, size=(10000))
        if test_attr is None:
            labels = np.concatenate([[i] * 100 for i in range(10)])
            # print(labels)
            # np.random.shuffle(labels)
            test_attr = labels_to_one_hot(labels, num_classes=10)

        num_batches = int(math.ceil(len(test_attr) / self.batch_size))

        # saveSampleImages(test_data.src_images, "%s/ref_src.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol)
        # saveSampleImages(test_data.tgt_images, "%s/ref_tgt.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol)
        d_results = np.random.randn(
            len(test_attr), self.image_size, self.image_size, self.channel)
        g_results = np.random.randn(
            len(test_attr), self.image_size, self.image_size, self.channel)

        for i in range(num_batches):
            
            index = slice(i * self.batch_size,
                          min(len(test_attr), (i + 1) * self.batch_size))

            attr = test_attr[index]
            z_vec = np.random.normal(
                size=(len(attr), self.z_dim), scale=1.0)
            # z_vec = latent_z[index]
            # print(z_vec)
            # print(src_img.shape)
            g_res = sess.run(self.gen_res, feed_dict={
                             self.attr: attr, self.z: z_vec})
            syn = sess.run(self.langevin_conditional_descriptor,
                           feed_dict={self.syn: g_res, self.attr: attr})

            d_results[index] = syn
            g_results[index] = g_res

        # save_dir = '%s/%06d' % (self.test_dir, epoch)
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        # os.makedirs(self.test_dir, exist_ok=True)
        saveSampleImages(g_results, "%s/eval_epoch%03d_gen.png" % (self.test_dir, epoch),
                         row_num=10, col_num=10, flip_color=True)
        saveSampleImages(d_results, "%s/eval_epoch%03d_des.png" % (self.test_dir, epoch),
                         row_num=10, col_num=10, flip_color=True)

        return d_results, g_results

    def test_generation(self, sess, ckpt, test_FID=True, useRandomLabel=True):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.attr, self.z)

        obs_res = self.descriptor(self.obs, self.attr, reuse=False)
        syn_res = self.descriptor(self.syn, self.attr, reuse=True)
        langevin_conditional_descriptor = self.langevin_dynamics_conditional_descriptor(
            self.syn, self.attr)


        saver = tf.train.Saver()

        print('Loading checkpoint {}.'.format(ckpt))
        saver.restore(sess, ckpt)


        d_results = []
        g_results = []

        if useRandomLabel:
            num_samples = 10000  # to compute the FID
        else:
            label = np.asarray(range(10)*10)
            attr_pool = labels_to_one_hot(label, num_classes=self.attr_size)
            attr_pool = np.tile(attr_pool, [10, 1])
            num_samples = len(attr_pool)

        num_batches = int(math.ceil(num_samples / self.batch_size))

        for i in range(num_batches):
            index = np.arange(i * self.batch_size,
                              min(num_samples, (i + 1) * self.batch_size))

            z_vec = np.random.normal(size=(len(index), self.z_dim), scale=1.0)

            if useRandomLabel:
                attr = np.random.randint(0, self.attr_size, len(index))
                attr = labels_to_one_hot(attr, num_classes=self.attr_size)
            else:
                attr = attr_pool[index]

            # print(src_img.shape)
            g_res = sess.run(gen_res, feed_dict={
                             self.attr: attr, self.z: z_vec})
            syn = sess.run(langevin_conditional_descriptor,
                           feed_dict={self.syn: g_res, self.attr: attr})

            d_results.append(syn)
            g_results.append(g_res)

        d_results = np.concatenate(d_results, axis=0)
        g_results = np.concatenate(g_results, axis=0)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        saveSampleImages(g_results, "%s/gen.png" % self.test_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)
        saveSampleImages(d_results, "%s/des.png" % self.test_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)

        if test_FID:

            print('Evaluating FID. Loading the training data...')
            train_data = MNISTDataSet(self.dataset_path, self.mnist_type, train=True, num_images=FLAGS.num_images,
                                      img_width=self.image_size, img_height=self.image_size, shuffle=True,
                                      low=-1, high=1)
            fid_util.init_fid()

            rand_ind = np.random.choice(len(d_results), 10000)  # we need 10000 examples to compute FID

            eval_data_des = d_results[rand_ind]
            fid_des = fid_util.get_fid(sess, eval_data_des,
                                       train_data.images[np.random.choice(len(train_data), len(eval_data_des))])

            eval_data_gen = g_results[rand_ind]
            fid_gen = fid_util.get_fid(sess, eval_data_gen,
                                       train_data.images[np.random.choice(len(train_data), len(eval_data_gen))])
            message = 'FID of slow-thinking = : {:.2f}, FID of fast-thinking = : {:.2f}'.format(fid_des, fid_gen)
            print(message)


    def test_infer_z(self, sess, ckpt):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.attr, self.z)

        obs_res = self.descriptor(self.obs, self.attr, reuse=False)
        syn_res = self.descriptor(self.syn, self.attr, reuse=True)
        langevin_conditional_descriptor = self.langevin_dynamics_conditional_descriptor(
            self.syn, self.attr)

        # labels = np.random.random_integers(0, 9, size=(10000))
        labels = np.asarray(range(10) * 10)
        print(labels)
        # np.random.shuffle(labels)
        test_attr = labels_to_one_hot(labels)

        test_attr = np.tile(test_attr, [10, 1])
        print(test_attr.shape)

        test_data = SVHNDataSet(self.dataset_path, img_width=self.image_size, img_height=self.image_size, shuffle=True,
                                low=-1, high=1, train=False)

        gt_imgs, gt_labels = test_data[np.random.randint(
            0, len(test_data), 100)]

        num_batches = int(math.ceil(len(test_attr) / self.batch_size))

        saver = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        # saveSampleImages(test_data.src_images, "%s/ref_src.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol)
        # saveSampleImages(test_data.tgt_images, "%s/ref_tgt.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol)
        d_results = np.random.randn(
            len(test_attr), self.image_size, self.image_size, self.channel)
        g_results = np.random.randn(
            len(test_attr), self.image_size, self.image_size, self.channel)

        for i in range(num_batches):
            index = slice(i * self.batch_size,
                          min(len(test_attr), (i + 1) * self.batch_size))

            gt_img, gt_label = test_data[np.random.randint(
                0, len(test_data), self.batch_size)]
            # z_vec = np.random.randn(min(sample_size, self.num_chain), self.z_size)
            attr = test_attr[index]
            z_vec = np.random.normal(size=(len(attr), self.z_dim), scale=1.0)
            # print(src_img.shape)
            g_res = sess.run(gen_res, feed_dict={
                             self.attr: attr, self.z: z_vec})
            syn = sess.run(langevin_conditional_descriptor,
                           feed_dict={self.syn: g_res, self.attr: attr})

            d_results[index] = syn
            g_results[index] = g_res

        saveSampleImages(g_results, "%s/gen.png" % self.test_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)
        saveSampleImages(d_results, "%s/des.png" % self.test_dir,
                         row_num=self.nTileRow, col_num=self.nTileCol)

    def test_mnist(self, sess, ckpt):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.attr, self.z)

        obs_res = self.descriptor(self.obs, self.attr, reuse=False)
        syn_res = self.descriptor(self.syn, self.attr, reuse=True)
        langevin_conditional_descriptor = self.langevin_dynamics_conditional_descriptor(
            self.syn, self.attr)

        # labels = np.random.random_integers(0, 9, size=(10000))
        labels = np.concatenate([[i] * 100 for i in range(10)])
        # print(labels)
        # np.random.shuffle(labels)
        test_attr = labels_to_one_hot(labels, num_classes=10)

        z_x = np.linspace(-0.7, 0.7, 10)
        z_y = np.linspace(-0.7, 0.7, 10)

        latent_z = np.zeros(shape=(100, 2), dtype=np.float32)

        for i in range(10):
            for j in range(10):
                latent_z[i*len(z_y)+j][0] = z_x[i]
                latent_z[i*len(z_y)+j][1] = z_y[j]
        # for i in range(100):

        # print(latent_z)

        latent_z = np.tile(latent_z, [10, 1])
        print(latent_z.shape)

        num_batches = int(math.ceil(len(test_attr) / self.batch_size))

        saver = tf.train.Saver()

        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        # saveSampleImages(test_data.src_images, "%s/ref_src.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol)
        # saveSampleImages(test_data.tgt_images, "%s/ref_tgt.png" % self.test_dir, row_num=self.nTileRow, col_num=self.nTileCol)
        d_results = np.random.randn(
            len(test_attr), self.image_size, self.image_size, self.channel)
        g_results = np.random.randn(
            len(test_attr), self.image_size, self.image_size, self.channel)

        for i in range(num_batches):
            index = slice(i * self.batch_size,
                          min(len(test_attr), (i + 1) * self.batch_size))

            # z_vec = np.random.randn(min(sample_size, self.num_chain), self.z_size)
            attr = test_attr[index]
            z_vec = latent_z[index]
            # print(z_vec)
            # print(src_img.shape)
            g_res = sess.run(gen_res, feed_dict={
                             self.attr: attr, self.z: z_vec})
            syn = sess.run(langevin_conditional_descriptor,
                           feed_dict={self.syn: g_res, self.attr: attr})

            d_results[index] = syn
            g_results[index] = g_res

        saveSampleImages(g_results, "%s/gen.png" %
                         self.test_dir, row_num=10, col_num=10, save_all=True)
        saveSampleImages(d_results, "%s/des.png" %
                         self.test_dir, row_num=10, col_num=10, save_all=True)

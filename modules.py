import tensorflow as tf
from utils.custom_ops import *


def descriptor_attr2img_mnist(inputs, attr, batch_layer=batch_norm, net={}, reuse=False):
    with tf.variable_scope('des', reuse=reuse):
        df_dim = 64
        # input ConvNet structure
        net['conv1'] = conv2d(inputs, df_dim, kernal=(5, 5), strides=(2, 2), padding="SAME", name="conv1")
        net['conv1'] = leaky_relu(net['conv1'])

        net['conv2'] = conv2d(net['conv1'], df_dim*2, kernal=(3, 3), strides=(2, 2), padding="SAME", name="conv2")
        net['conv2'] = leaky_relu(net['conv2'])

        # net['conv3'] = conv2d(net['conv2'], df_dim*4, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv3")
        # net['conv3'] = leaky_relu(net['conv3'])


        # net['conv4'] = conv2d(net['conv3'], df_dim*8, kernal=(4, 4), strides=(2, 2), padding="SAME", name="conv4")
       #  net['conv4'] = leaky_relu(net['conv4'])
        # (bn) x 4 x 4 x 512
        # (bn) x 312

        # net['attr'] = linear(attr, 128, name='linear')
        net['attr'] = tf.reshape(attr, [-1, 1, 1, attr.get_shape()[1]])
        net['attr'] = tf.tile(net['attr'], [1, 7, 7, 1], name='tile')
        net['enc'] = tf.concat([net['attr'], net['conv2']], axis=3, name='concat')

        net['conv3'] = conv2d(net['enc'], df_dim*4, kernal=(3, 3), strides=(1, 1), padding="SAME", name="conv3")
        net['conv3'] = leaky_relu(net['conv3'])

        net['fc'] = fully_connected(net['conv3'], 100, name="fc")

        return net['fc']


def generator_mnist(input_, z=None, batch_layer=batch_norm, reuse=False):
    gf_dim = 64

    with tf.variable_scope('gen', reuse=reuse):
        # attr = tf.layers.dense(attr, units=256, activation=leaky_relu,
        #                               kernel_initializer=tf.random_normal_initializer(stddev=0.01))

        if z is not None:
            input_ = tf.concat([input_, z], axis=1)
        input_ = tf.reshape(input_, [-1, 1, 1, input_.get_shape()[1]])

        # (1 x 1 x 512)
        convt1 = convt2d(input_, (None, 4, 4, gf_dim*4), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")
        convt1 = batch_layer(convt1, name="convt1_bn")
        # convt1 = tf.nn.dropout(convt1, 0.5)
        convt1 = tf.nn.relu(convt1)

        # (2 x 2 x 512)
        convt2 = convt2d(convt1, (None, 7, 7, gf_dim*2), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt2")
        convt2 = batch_layer(convt2, name="convt2_bn")
        # convt2 = tf.nn.dropout(convt2, 0.5)
        convt2 = tf.nn.relu(convt2)

        # (4 x 4 x 512)
        convt3 = convt2d(convt2, (None, 14, 14, gf_dim), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt3")
        convt3 = batch_layer(convt3, name="convt3_bn")
        # convt3 = tf.nn.dropout(convt3, 0.5)
        convt3 = tf.nn.relu(convt3)

        # (8 x 8 x 512)
        convt4 = convt2d(convt3, (None, 28, 28, 1), kernal=(4, 4), strides=(2, 2), padding="SAME", name="convt4")
        convt4 = tf.nn.tanh(convt4)

        return convt4
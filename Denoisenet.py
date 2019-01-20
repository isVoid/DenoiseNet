from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from layers import *

PATCH_SHAPE = (128 ,128, 1)

def feed_forward(X_train, y_train = None, scope = ""):
    """Denoise Net Model
        https://arxiv.org/pdf/1701.01687.pdf

        Input:
            X_train: noisy image patch of size (128, 128, 3)
            y_train: ground truth image, or long exposure image patch

        Output:
            If ground truth is not provided, test mode:
            return final layer of output.

            If ground truth is provided, train or eval mode:
            return final layer and loss.
    """

    if (len(X_train.shape) == 3):
        X_train = tf.expand_dims(X_train, -1)
        y_train = tf.expand_dims(y_train, -1)

    n_layers = 7
    conv_prev = X_train
    layer_lists = []
    last_channels = []

    for l in range(n_layers):
        if not l == 0:
            conv_prev = conv_prev[:, :, :, 0:-1]

        conv = tf.contrib.layers.conv2d(conv_prev,
                        num_outputs = 32,
                        kernel_size = 5,
                        stride = 1,
                        padding = 'SAME',
                        data_format = None,
                        activation_fn = tf.nn.relu,
                        scope = scope+"/"+str(l))
        layer_lists.append(conv)
        print (conv.name)
        lc = conv[:, :, :, -1]
        lc = tf.expand_dims(lc, -1)
        last_channels.append(lc)
        conv_prev = conv

    last_conv = layer_lists[-1]
    conv_last = tf.contrib.layers.conv2d(last_conv[:, :, :, 0:-1],
                        num_outputs = 32,
                        kernel_size = 7,
                        stride = 1,
                        padding = 'SAME',
                        data_format = None,
                        activation_fn = tf.nn.sigmoid,
                        scope = scope+"/"+"last")
    layer_lists.append(conv_last)
    print (conv_last.name)
    lc = conv_last[:, :, :, -1]
    lc = tf.expand_dims(lc, -1)
    last_channels.append(lc)

    denoised = tf.zeros_like(last_channels[0])
    for lc in last_channels:
        denoised = tf.add(denoised, lc)

    if y_train == None:
        return denoised

    _loss = loss(denoised, y_train)

    return denoised, _loss

def loss(output, ground_truth):
    """Loss function of deep denoise model

    Input:
        output: Image patch of feed_forward output, tensor shape (mini_batch_size, PATCH_SHAPE)
        ground_truth: Image patch of long exposure, tensor shape (mini_batch_size, PATCH_SHAPE)

    output:
        loss: mean squared error of output and ground_truth

    """

    return tf.losses.mean_squared_error(output, ground_truth)

# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import numpy as np

def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)
'''
def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)

def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name="conv2d_transpose")
'''
def conv2d(x, in_dim, out_dim, keep_prob_):
    with tf.name_scope("conv2d"):
        stddev = np.sqrt(2 / (3 ** 2 * out_dim))
        w = weight_variable([3, 3, in_dim, out_dim], stddev, name="w")
        b = bias_variable([out_dim], name='b')
        conv_2d = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)

def deconv2d(x, in_dim, out_dim):
    with tf.name_scope("deconv2d"):
        stddev = np.sqrt(2 / (3 ** 2 * out_dim))
        wd = weight_variable([2, 2, out_dim, in_dim], stddev, name="wd")
        bd = bias_variable([out_dim], name="bd")
        x_shape = tf.shape(x)
        #output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, out_dim])
        deconv_2d = tf.nn.conv2d_transpose(x, wd, output_shape, strides=[1, 2, 2, 1], padding='VALID', name="conv2d_transpose")
        deconv_2d_b = tf.nn.bias_add(deconv_2d, bd)
        return deconv_2d_b

def inception_conv(x, in_dim, out_dim, keep_prob_, training):
    with tf.name_scope("conv2d"):
        stddev = np.sqrt(2 / (3 ** 2 * out_dim))
        #x = tf.layers.batch_normalization(x, training=training)
        w0 = weight_variable([3, 3, in_dim, out_dim], stddev, name="w0")
        b0 = bias_variable([out_dim], name="b0")
        conv2d_0 = tf.nn.bias_add(tf.nn.conv2d(x, w0, strides=[1, 1, 1, 1], padding="SAME"), b0)

        w1 = weight_variable([1, 1, out_dim, out_dim], stddev, name="w1")
        b1 = bias_variable([out_dim], name="b1")
        conv2d_1 = tf.nn.bias_add(tf.nn.conv2d(conv2d_0, w1, strides=[1, 1, 1, 1], padding="SAME"), b1)
        conv2d_1 = tf.nn.relu(conv2d_1)
        w2 = weight_variable([3, 3, out_dim, out_dim], stddev, name="w2")
        b2 = bias_variable([out_dim], name="b2")
        conv2d_2 = tf.nn.bias_add(tf.nn.conv2d(conv2d_0, w2, strides=[1, 1, 1, 1], padding="SAME"), b2)
        conv2d_2 = tf.nn.relu(conv2d_2)
        w3 = weight_variable([5, 5, out_dim, out_dim], stddev, name="w3")
        b3 = bias_variable([out_dim], name="b3")
        conv2d_3 = tf.nn.bias_add(tf.nn.conv2d(conv2d_0, w3, strides=[1, 1, 1, 1], padding="SAME"), b3)
        conv2d_3 = tf.nn.relu(conv2d_3)
        w4 = weight_variable([3, 3, out_dim, out_dim], stddev, name="w4")
        b4 = bias_variable([out_dim], name="b4")
        conv2d_4 = tf.nn.bias_add(tf.nn.atrous_conv2d(conv2d_0, w4, rate=2, padding="SAME"), b4)
        conv2d_4 = tf.nn.relu(conv2d_4)
        w5 = weight_variable([3, 3, out_dim, out_dim], stddev, name="w5")
        b5 = bias_variable([out_dim], name="b5")
        conv2d_5 = tf.nn.bias_add(tf.nn.atrous_conv2d(conv2d_0, w5, rate=4, padding="SAME"), b5)
        conv2d_5 = tf.nn.relu(conv2d_5)

        conv2d = tf.concat([conv2d_1, conv2d_2, conv2d_3, conv2d_4, conv2d_5], axis=3)
        w = weight_variable([3, 3, out_dim*5, out_dim], stddev, name="w")
        b = bias_variable([out_dim], name="b")
        conv2d = tf.nn.bias_add(tf.nn.conv2d(conv2d, w, strides=[1, 1, 1, 1], padding="SAME"), b)
        conv2d = tf.nn.relu(conv2d)
        conv2d = conv2d + conv2d_0

        return conv2d

def dense_link(x, stride, out_dim):
    with tf.name_scope("dense_link"):
        stddev = np.sqrt(2 / (3 ** 2 * out_dim))
        w = weight_variable([3, 3, out_dim, out_dim], stddev, name="w")
        b = bias_variable([out_dim], name="b")
        conv = tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding="SAME"), b)
        return conv

def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keep_dims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keep_dims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")

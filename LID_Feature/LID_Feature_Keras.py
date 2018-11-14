#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Lambda
from keras.layers import AveragePooling2D, Input, Flatten, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import keras.backend as K
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import sys
from pathlib import *
import torch
import tensorflow as tf
from tensorflow.python import debug as tf_debug

if(len(sys.argv)!=1):
    order = int(sys.argv[1])
    batch_size = order
else:
    batch_size = 128  # orig paper trained all networks with batch_size=128

# Training parameters
# batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 20
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
# exp_name = 'BaseLine_resNet_Cifar10_BS%d_epochs%d' % (batch_size, epochs)
exp_name = 'LID_Feature_NoAct_20LID_resNet_Cifar10_BS%d_epochs%d_Shuffle' % (batch_size, epochs)

n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
work_path=Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Logs/LID_Feature')
# root_path = '/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Cifar10_Aug/Pics_Debug_5w+delete'
# if not (os.path.exists(root_path)): print("augmentation data not found!")
# x_train = np.load(root_path+"/aug_train_x.npy")
# y_train = np.load(root_path+"/aug_train_y.npy")
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-3
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name = None):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# 输入：X（计算点）,Y(参考点),batch_size
# 输出：mat. mat记录了X的距离最近的k个点与最远点的距离比，以及最远点的距离。
def LID_torch(X, Y, k):
    # import ipdb;ipdb.set_trace()
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:]) # XX指的是数据集中的其余点
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:]) # YY指的是reference point
    # XX = torch.from_numpy(XX)
    # YY = torch.from_numpy(YY)
    dist_mat = torch.pow(torch.sum(torch.pow(XX - YY, 2), dim=sum_axis), 0.5)
    dist_mat = torch.where(dist_mat < 1e-10, torch.full_like(dist_mat,1e10), dist_mat)
    sorted_mat = torch.sort(dist_mat, dim=1)
    r_max = sorted_mat[0][:, k-1].reshape(-1, 1)
    mask = (dist_mat <= r_max).float()

    mat = -1 / ( torch.log(sorted_mat[0][:,:k-1]/r_max))
    # est = -1 / (1 / k * torch.sum(torch.log(dist_mat) * mask, dim=1) - torch.log(r_max).reshape(-1))
    return mat

# X,Y格式为tensor
# def LID_keras(X, Y, k):
#     X_shape = X.shape.as_list()
#     Y_shape = Y.shape.as_list()
#     # k = tf.sqrt(X_shape[0])
#     sum_axis = tuple([i for i in range(2, len(X_shape) + 1)])
#     # XX = X.reshape(X_shape[0], 1, *X_shape[1:]) # XX指的是数据集中的其余点
#     # YY = Y.reshape(1, Y_shape[0], *Y_shape[1:]) # YY指的是reference point
#     # XX = tf.reshape(X,[batch_size, 1, -1])
#     # YY = tf.reshape(Y,[1, batch_size, -1])
#     XX = tf.expand_dims(X, 1)
#     YY = tf.expand_dims(Y, 0)
#     dist_mat = K.sqrt(K.sum(K.pow(XX - YY, 2), axis=sum_axis))
#     dist_mat += tf.cast((dist_mat < 1e-10), tf.float32)  * tf.constant(1e10)
#     sorted_mat = tf.nn.top_k(dist_mat, k=k, sorted=True)
#     r_max = tf.reshape(sorted_mat[0][:, k-1],[-1,1])
#     mat = -1 / ( K.log(sorted_mat[0][:,:k-1]/r_max))
#     return mat

def LID_keras(X, Y, k):

    X_shape = X.shape.as_list()
    Y_shape = Y.shape.as_list()
    # k = tf.sqrt(X_shape[0])
    sum_axis = tuple([i for i in range(2, len(X_shape) + 1)])
    XX = tf.expand_dims(X, 1)
    YY = tf.expand_dims(Y, 0)
    dist_mat = K.sqrt(K.sum(K.pow(XX - YY, 2), axis=sum_axis))
    dist_mat += tf.cast((dist_mat < 1e-10), tf.float32) * tf.constant(1e10)

    sorted_mat = -tf.nn.top_k(-dist_mat, k=k, sorted=True).values
    # r_max = sorted_mat[0]
    r_max = sorted_mat[:, k - 1]
    # r_max = tf.expand_dims(r_max, 0)

    import ipdb; ipdb.set_trace()
    print('shape:', r_max.shape)
    mat = -1 / (tf.log(sorted_mat / r_max))
    # mat = -1 / (K.log(sorted_mat[:, :k - 1] / (r_max)))

    # mat = -1 / (1 / k * tf.reduce_sum(K.log(sorted_mat),axis=1) - K.log(r_max))
    # mat = K.eval(mat)

    # import ipdb; ipdb.set_trace()
    # print(mat.shape)
    print('OYZH'*10)
    return mat

# global selected_layer_out
def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
x`
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)


    ####Add LID Feature Here:
    selected_layer_out = y
    # lid_feature = lid(y)
    # Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))
    lid_feature = Lambda(lambda x:
        LID_keras(x,x, k = int(np.sqrt(batch_size)))
                         )(selected_layer_out)


    print('x shape:', x.shape)
    print('y shape', y.shape)
    print('lid_feature shape;', lid_feature)

    lid_feature = Dense(20)(lid_feature)
    # import ipdb; ipdb.set_trace()
    # lid_feature = Flatten()(lid_feature)
    # y = Flatten()(y)
    # lid_feature = BatchNormalization()(lid_feature)
    #it depends whether to add activation layer here
    # lid_feature = Activation('relu')(lid_feature)


    y = Concatenate()([y, lid_feature])

    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define Model

#ResNet:
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy', 'top_k_categorical_accuracy'])
model.summary()

# Prepare model model saving directory.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
x_train_epoch = []
y_train_epoch = []
def renew_train_dataset():
    mask = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
    global x_train_epoch
    x_train_epoch = x_train[mask]
    global y_train_epoch
    y_train_epoch = y_train[mask]

def on_epoch_end(epoch, logs):
    print('End of epoch')
    import ipdb; ipdb.set_trace()
    K.print_tensor(selected_layer_out, 'lid_feature: ')
    renew_train_dataset()

on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
renew_train_dataset()
# K.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))
from keras.callbacks import TensorBoard
callbacks = [on_epoch_end_callback, lr_reducer, lr_scheduler,TensorBoard(
    log_dir= (work_path/'TB_Log'/exp_name).__str__())]

model.fit(x_train_epoch, y_train_epoch,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
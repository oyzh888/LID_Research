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
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os
from sklearn.decomposition import PCA
import lid
from lid import LID
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import sys
from pathlib import *

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

source_percent = 1
target_percent = 1
drop_alg='Resample_aug_high%d_to_%d' % (source_percent,target_percent)
# drop_alg='Resample_cls_aug_high%d_to_%d' % (source_percent,target_percent)
# drop_alg='baseline'
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
# exp_name = 'LID_%s_BS%d_epochs%d_Keras_Shuffle' % (drop_alg,batch_size, epochs)
# exp_name = 'LID_%s_resNet_Cifar10_BS%d_epochs%d_Baseline' % (drop_alg,batch_size, epochs)
exp_name = 'LID_%s_BS%d_epochs%d_Class_LID_Keras_Shuffle' % (drop_alg,batch_size, epochs)
# exp_name = 'LID_%s_BS%d_epochs%d_Class_LID_Keras_Shuffle_Duplicate1' % (drop_alg,batch_size, epochs)
n = 3
# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
work_path=Path('../../../Cifar10_LID_DataDrop')
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
# model.summary()
print("-"*20+exp_name+'-'*20)

# Prepare model model saving directory.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# lid load
lid_train = np.load("/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research_local/nparray/Cifar10_ground_truth_dataset50000_BS5000_Globale_Class_lid_K70.npy")
# lid_train = np.load("/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research_local/nparray/Cifar10_ground_truth_dataset50000_BS5000_lid_K70_TEST1.npy")
# # lid_selected_idx = np.argwhere(lid_train < np.percentile(lid_train,100 - drop_percent)).flatten()#Drop High
# lid_selected_idx = np.argwhere(lid_train > np.percentile(lid_train,drop_percent)).flatten()#Drop Low
# x_train,y_train=x_train[lid_selected_idx],y_train[lid_selected_idx]
#
#
#
# # Re-sample
# sample_mask = np.random.choice(len(x_train),int(train_num*drop_percent/100), replace=False)
# x_train = np.append(x_train,x_train[sample_mask], axis=0)
# y_train = np.append(y_train,y_train[sample_mask], axis=0)

# print('OYZH shape:', y_train.shape )
# import ipdb; ipdb.set_trace()
x_train_epoch = []
y_train_epoch = []

# x_train = np.append(x_train_aug, x_train_ori)


def renew_train_dataset():
    alpha = source_percent;
    beta = target_percent;  # alpha表示抽取lid最高样本的比例，beta表示前面抽取的样本在新的样本集中占据了多少比例.beta>alpha

    # 按照全局LID进行resample
    lid_high_idx = np.argwhere(lid_train > np.percentile(lid_train, 100 - alpha)).flatten()  # select high lid idx
    lid_low_idx = np.argwhere(lid_train <= np.percentile(lid_train, 100 - alpha)).flatten()  # select low lid idx

    lid_high_aug_idx = np.append(lid_high_idx,
                                 np.random.choice(lid_high_idx, int(train_num * ((beta - alpha) / 100)), replace=True))
    lid_low_aug_idx = np.random.choice(lid_low_idx, int(train_num * (1 - beta / 100)), replace=False)

    # 按照各类别LID进行resample
    # lid_sorted_idx = np.argsort(-lid_train) # 从大到小对LID排序，记录其下标。
    # y_train_lid_sorted = np.argmax(y_train[lid_sorted_idx],axis=1)
    # lid_high_aug_idx=[]
    # lid_low_aug_idx = []
    # for cls in range(num_classes):
    #     cls_lid_sorted_idx = lid_sorted_idx[y_train_lid_sorted==cls]
    #     cls_train_num=len(cls_lid_sorted_idx)
    #     lid_high_idx = cls_lid_sorted_idx[:int(cls_train_num*source_percent/100)]
    #     lid_low_idx = cls_lid_sorted_idx[int(cls_train_num*source_percent/100):]
    #     # print("before aug",len(lid_high_aug_idx),len(lid_low_aug_idx))
    #     lid_high_aug_idx.extend(lid_high_idx)
    #     lid_high_aug_idx.extend(np.random.choice(lid_high_idx, int(cls_train_num * ((beta - alpha) / 100)),replace=True))
    #     lid_low_aug_idx.extend(np.random.choice(lid_low_idx,int(cls_train_num * (1 - beta / 100)), replace=False))
        # print("after aug", len(lid_high_aug_idx), len(lid_low_aug_idx),"type",type(lid_low_aug_idx[0]))

    # print('lid_high_aug_idx', len(lid_high_aug_idx))
    # print('lid_low_aug_idx', len(lid_low_aug_idx))


    new_selected_index = np.append(lid_high_aug_idx,lid_low_aug_idx)
    new_selected_x_train = x_train[new_selected_index]
    new_selected_y_train = y_train[new_selected_index]

    not_selected_idx = np.delete(np.arange(train_num),new_selected_index)

    # print('not_selected_idx',not_selected_idx)
    # print('not_selected_idx shape', not_selected_idx.shape)
    # import ipdb;
    # ipdb.set_trace()
    mask = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
    global x_train_epoch
    x_train_epoch = new_selected_x_train[mask]
    global y_train_epoch
    y_train_epoch = new_selected_y_train[mask]

def on_epoch_end(epoch, logs):
    print('End of epoch')
    print("dataset size ",x_train.shape[0])
    renew_train_dataset()

on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
from keras.callbacks import TensorBoard
callbacks = [lr_reducer, lr_scheduler, on_epoch_end_callback, TensorBoard(
    log_dir= (work_path/'TB_Log'/exp_name).__str__())]
# baseline
# callbacks = [lr_reducer, lr_scheduler, TensorBoard(
#     log_dir= (work_path/'TB_Log'/exp_name).__str__())]
renew_train_dataset()
# import ipdb;ipdb.set_trace()
# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train_epoch, y_train_epoch,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
    # Baseline:
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           shuffle=True,
    #           callbacks=callbacks)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
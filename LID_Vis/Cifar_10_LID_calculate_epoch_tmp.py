#!/usr/bin/env python
# coding: utf-8
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
from lid import *
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import sys
from pathlib import *
import math
if(len(sys.argv)!=1):
    order = int(sys.argv[1])
    batch_size = order
else:
    batch_size = 128  # orig paper trained all networks with batch_size=128

# Training parameters
# batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 2
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
# exp_name = 'BaseLine_resNet_Cifar10_BS%d_epochs%d' % (batch_size, epochs)
exp_name = 'Debug_resNet_Cifar10_BS%d_epochs%d' % (batch_size, epochs)
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
work_path=Path('../Cifar10_epoch_Lid')
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
model.summary()
print(model_type)

# Prepare model model saving directory.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
x_train_epoch = []
y_train_epoch = []

from progressbar import *
def get_lid(outputs_predict):
    batch_size_lid = batch_size
    # import ipdb;
    # ipdb.set_trace()
    flatten_shape = outputs_predict.shape[0]
    outputs_predict_flatten = np.reshape(outputs_predict, newshape=(flatten_shape, -1))
    outputs_predict_lid = np.zeros(flatten_shape)
    lid_k = int(np.sqrt(batch_size_lid))
    batch_num = int(flatten_shape / batch_size_lid)
    mask_batch = []
    pbar = ProgressBar()
    for i in pbar(range(batch_num)):
        if((i+1)*batch_size_lid<flatten_shape):
            mask_batch = np.arange(i*batch_size_lid,(i+1)*batch_size_lid)  # 一个样本下标仅出现一次,顺序训练
        else:
            mask_batch = np.arange(i*batch_size_lid,flatten_shape)
        # import ipdb;ipdb.set_trace()
        # print(mask_batch)
        dis = LID(outputs_predict_flatten[mask_batch], outputs_predict_flatten[mask_batch], lid_k)
        dis = GPU_lid_eval_keras(outputs_predict_flatten[mask_batch], lid_k)

        outputs_predict_lid[mask_batch] = dis
    return outputs_predict_lid

def renew_train_dataset():
    mask = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
    global x_train_epoch
    x_train_epoch = x_train[mask]
    global y_train_epoch
    y_train_epoch = y_train[mask]

epoch_lids_train = []
epoch_lids_test = []
def on_epoch_end(epoch, logs):
    layer_name = 'average_pooling2d_1'
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)

    intermediate_output = intermediate_layer_model.predict(x_train)

    print(x_train.shape)
    print(intermediate_output.shape)
    lids_of_train = get_lid(intermediate_output)


    intermediate_output = intermediate_layer_model.predict(x_test)
    lids_of_test = get_lid(intermediate_output)

    epoch_lids_train.append(lids_of_train)
    epoch_lids_test.append(lids_of_test)
    print("LID Averge:",np.average(lids_of_train))
    print('End of epoch')
    renew_train_dataset()

on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
from keras.callbacks import TensorBoard
callbacks = [lr_reducer, lr_scheduler, on_epoch_end_callback, TensorBoard(
    log_dir= (work_path/'TB_Log'/exp_name).__str__())]

renew_train_dataset()
# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train_epoch, y_train_epoch,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=False,
              callbacks=callbacks)

path_to_create = work_path/'LID_Epoch_NP'/exp_name
path_to_create.mkdir(parents=True)

np.save(work_path/'LID_Epoch_NP'/exp_name/'train_lid.npy',epoch_lids_train)
np.save(work_path/'LID_Epoch_NP'/exp_name/'test_lid.npy',epoch_lids_test)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])